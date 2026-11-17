"""Durable plan/task/step state. Plan edits and ticket revocation are atomic."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import nullcontext

import db
import orders
import refund_tasks
import tickets
from planning import PlanDraft

MAX_CALLS = 40
MAX_REPLANS = 6
TERMINAL = {"succeeded", "failed", "cancelled", "blocked"}


class PlanConflict(ValueError):
    pass


def init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agent_plans (
            id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, requester_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL, request_id TEXT NOT NULL, draft_hash TEXT NOT NULL,
            goal TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
            calls_used INTEGER NOT NULL DEFAULT 0, max_calls INTEGER NOT NULL,
            replans INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
            last_scanned_at REAL NOT NULL DEFAULT 0,
            UNIQUE(tenant_id, requester_id, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS agent_plan_tasks (
            id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, task_key TEXT NOT NULL,
            scope_thread_id TEXT NOT NULL UNIQUE, UNIQUE(plan_id, task_key)
        );
        CREATE TABLE IF NOT EXISTS agent_plan_steps (
            plan_id TEXT NOT NULL, step_key TEXT NOT NULL, task_id TEXT NOT NULL,
            action TEXT NOT NULL, depends_on TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            run_token TEXT, lease_until REAL, next_run_at REAL NOT NULL DEFAULT 0,
            result TEXT, error_code TEXT, started_at REAL, duration_ms REAL,
            PRIMARY KEY(plan_id, step_key)
        );
        CREATE INDEX IF NOT EXISTS idx_plan_steps_ready ON agent_plan_steps(status, next_run_at);
        CREATE TABLE IF NOT EXISTS agent_plan_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL,
            request_id TEXT NOT NULL, task_id TEXT, step_id TEXT, plan_revision INTEGER NOT NULL,
            kind TEXT NOT NULL, detail TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_plan_events ON agent_plan_events(plan_id, id);
    """)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(agent_plans)")}
    if "last_scanned_at" not in columns:
        conn.execute("ALTER TABLE agent_plans ADD COLUMN last_scanned_at REAL NOT NULL DEFAULT 0")


def _scope(user, conversation_id):
    if not user.tenant_id or not user.username or not conversation_id:
        raise PlanConflict("缺少已认证的计划归属")
    return user.tenant_id, user.username, conversation_id


def _read(conn, user, conversation_id):
    row = conn.execute("""SELECT * FROM agent_plans
        WHERE tenant_id=? AND requester_id=? AND conversation_id=?""",
        _scope(user, conversation_id)).fetchone()
    return dict(row) if row else None


def event(conn, plan, kind, *, task_id=None, step_id=None, **detail):
    conn.execute("""INSERT INTO agent_plan_events
        (plan_id, request_id, task_id, step_id, plan_revision, kind, detail, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (plan["id"], plan["request_id"], task_id, step_id, plan["revision"], kind,
         json.dumps(detail, ensure_ascii=False), time.time()))


def get(user, conversation_id, *, _conn=None):
    with (nullcontext(_conn) if _conn is not None else db.session(tickets.DB_PATH)) as conn:
        plan = _read(conn, user, conversation_id)
        if not plan:
            return None
        branches = conn.execute("SELECT * FROM agent_plan_tasks WHERE plan_id=? ORDER BY rowid", (plan["id"],)).fetchall()
        plan["tasks"] = [dict(**dict(t), **{"state": refund_tasks.get(user, t["scope_thread_id"], _conn=conn)}) for t in branches]
        rows = conn.execute("SELECT * FROM agent_plan_steps WHERE plan_id=? ORDER BY rowid", (plan["id"],)).fetchall()
        plan["steps"] = []
        for row in rows:
            step = dict(row)
            step["depends_on"] = json.loads(step["depends_on"])
            step["result"] = json.loads(step["result"]) if step["result"] else None
            step.pop("run_token")
            plan["steps"].append(step)
    statuses = {s["status"] for s in plan["steps"]}
    by_key = {s["step_key"]: s for s in plan["steps"]}
    runnable = any(s["status"] in {"pending", "retry"} and all(
        by_key[d]["status"] == "succeeded" for d in s["depends_on"]) for s in plan["steps"])
    plan["status"] = (
        "completed" if statuses == {"succeeded"} else
        "finished_with_issues" if statuses <= TERMINAL else
        "running" if "running" in statuses else
        "ready" if runnable else
        "waiting_approval" if "waiting_approval" in statuses else "waiting_user"
    )
    return plan


def create(user, conversation_id, draft: PlanDraft, *, request_id=""):
    # Revalidate even callers passing model_construct() must not bypass the contract.
    draft = PlanDraft.model_validate(draft.model_dump())
    digest = hashlib.sha256(draft.model_dump_json().encode()).hexdigest()
    for task in draft.tasks:
        if task.order_id.strip():
            orders.get_order(user, task.order_id.strip())
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if tickets.conversation_closed(conn, *_scope(user, conversation_id)):
            raise PlanConflict("会话已关闭，不能创建退款计划")
        old = _read(conn, user, conversation_id)
        if old:
            if old["draft_hash"] == digest:
                return get(user, conversation_id, _conn=conn)
            raise PlanConflict("本会话已有计划；先读取计划，按任务标识与版本局部修改")
        legacy = refund_tasks.get(user, conversation_id, _conn=conn)
        if legacy["status"] in {"pending_approval", "approved", "executing", "ready", "collecting"}:
            raise PlanConflict("本会话已有单笔退款任务，请先完成或撤回，再创建多笔计划")
        pid = uuid.uuid4().hex
        conn.execute("""INSERT INTO agent_plans
            (id, tenant_id, requester_id, conversation_id, request_id, draft_hash, goal, max_calls, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pid, *_scope(user, conversation_id), request_id or uuid.uuid4().hex, digest, draft.goal, MAX_CALLS, time.time()))
        task_ids = {}
        for task in draft.tasks:
            tid = uuid.uuid4().hex
            task_ids[task.key] = tid
            scope = "task_" + tid
            conn.execute("INSERT INTO agent_plan_tasks VALUES (?, ?, ?, ?)", (tid, pid, task.key, scope))
            refund_tasks.update(user, scope, expected_revision=0, order_id=task.order_id,
                                reason=task.reason, intent=task.intent, source="planner", _conn=conn)
        for step in draft.steps:
            conn.execute("""INSERT INTO agent_plan_steps
                (plan_id, step_key, task_id, action, depends_on) VALUES (?, ?, ?, ?, ?)""",
                (pid, step.key, task_ids[step.task], step.action, json.dumps(step.depends_on)))
            if step.action == "request_refund":
                # '$' cannot occur in model-provided keys. These steps cannot be proposed by it.
                for suffix, action, dep in (("$approval", "await_approval", step.key),
                                            ("$execute", "execute_refund", step.key + "$approval")):
                    conn.execute("""INSERT INTO agent_plan_steps
                        (plan_id, step_key, task_id, action, depends_on) VALUES (?, ?, ?, ?, ?)""",
                        (pid, step.key + suffix, task_ids[step.task], action, json.dumps([dep])))
        event(conn, _read(conn, user, conversation_id), "plan_created", step_count=len(draft.steps) + 2 * len(draft.tasks))
    return get(user, conversation_id)


def edit(user, conversation_id, task_id, *, expected_revision, order_id=None,
         reason=None, intent=None, cancel=False, request_id=""):
    """Optimistic branch edit; keep successful independent/read-only steps."""
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if tickets.conversation_closed(conn, *_scope(user, conversation_id)):
            raise PlanConflict("会话已关闭，不能修改退款计划")
        plan = _read(conn, user, conversation_id)
        if not plan:
            raise PlanConflict("计划不存在")
        branch = conn.execute("SELECT * FROM agent_plan_tasks WHERE plan_id=? AND id=?", (plan["id"], task_id)).fetchone()
        if not branch:
            raise PlanConflict("任务不存在或不属于本计划")
        old = refund_tasks.get(user, branch["scope_thread_id"], _conn=conn)
        if old["status"] in {"executed", "approved", "executing"}:
            raise PlanConflict("该任务已批准或完成，不能重规划")
        if plan["replans"] >= MAX_REPLANS:
            raise PlanConflict("已达到计划修改次数上限，请转人工处理")
        if order_id and conn.execute("""SELECT 1 FROM agent_plan_tasks b JOIN refund_tasks t
            ON t.thread_id=b.scope_thread_id WHERE b.plan_id=? AND b.id!=? AND t.order_id=?""",
            (plan["id"], task_id, order_id.strip())).fetchone():
            raise PlanConflict("同一计划不能重复申请同一订单")
        new = refund_tasks.update(user, branch["scope_thread_id"], expected_revision=expected_revision,
                                 order_id=order_id, reason=reason, intent=intent, cancel=cancel,
                                 source="replanner", turn_id=request_id, _conn=conn)
        if old["revision"] == new["revision"]:
            return get(user, conversation_id, _conn=conn)
        rows = [dict(r) for r in conn.execute("SELECT * FROM agent_plan_steps WHERE plan_id=?", (plan["id"],))]
        order_changed = old["order_id"] != new["order_id"]
        affected = {s["step_key"] for s in rows if s["task_id"] == task_id and (
            cancel or order_changed or s["action"] not in {"lookup_order", "check_policy"} or s["status"] != "succeeded")}
        # Invalidate transitive dependants, including explicitly coupled read-only branches.
        while True:
            expanded = affected | {s["step_key"] for s in rows if set(json.loads(s["depends_on"])) & affected}
            if expanded == affected:
                break
            affected = expanded
        # Do not roll back a completed side effect in another branch.
        if any(s["step_key"] in affected and s["task_id"] != task_id and s["action"] == "request_refund"
               and s["status"] == "succeeded" for s in rows):
            raise PlanConflict("修改会影响另一笔已提交的申请，请先处理关联申请")
        plan["revision"] += 1
        plan["request_id"] = request_id or uuid.uuid4().hex
        conn.execute("UPDATE agent_plans SET revision=?, replans=replans+1, request_id=? WHERE id=?",
                     (plan["revision"], plan["request_id"], plan["id"]))
        for s in rows:
            if s["step_key"] not in affected:
                continue
            conn.execute("""UPDATE agent_plan_steps SET version=version+1, status=?, attempts=0,
                run_token=NULL, lease_until=NULL, next_run_at=0, result=NULL, error_code=NULL,
                started_at=NULL, duration_ms=NULL WHERE plan_id=? AND step_key=?""",
                ("cancelled" if cancel and s["task_id"] == task_id else "pending", plan["id"], s["step_key"]))
        event(conn, plan, "branch_replanned", task_id=task_id,
              invalidated=sorted(affected), task_revision=new["revision"], cancelled=cancel)
    return get(user, conversation_id)


def history(user, conversation_id, *, after=0, limit=100):
    with db.session(tickets.DB_PATH) as conn:
        plan = _read(conn, user, conversation_id)
        if not plan:
            return []
        rows = conn.execute("SELECT * FROM agent_plan_events WHERE plan_id=? AND id>? ORDER BY id LIMIT ?",
                            (plan["id"], max(after, 0), min(max(limit, 1), 200))).fetchall()
    return [dict(r) | {"detail": json.loads(r["detail"])} for r in rows]


def conversation_for(ticket):
    with db.session(tickets.DB_PATH) as conn:
        row = conn.execute("""SELECT p.conversation_id FROM agent_plans p JOIN agent_plan_tasks t ON p.id=t.plan_id
            WHERE t.scope_thread_id=? AND p.tenant_id=? AND p.requester_id=?""",
            (ticket["thread_id"], ticket["tenant_id"], ticket["requester_id"])).fetchone()
    return row[0] if row else ticket["thread_id"]


def task_for_scope(user, conversation_id, scope):
    with db.session(tickets.DB_PATH) as conn:
        row = conn.execute("""SELECT t.id FROM agent_plan_tasks t JOIN agent_plans p ON p.id=t.plan_id
            WHERE p.tenant_id=? AND p.requester_id=? AND p.conversation_id=? AND t.scope_thread_id=?""",
            (*_scope(user, conversation_id), scope)).fetchone()
    return row[0] if row else None
