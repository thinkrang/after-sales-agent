"""Durable conversational task state, separate from message history.

Task edits and pending-ticket invalidation share one SQLite transaction. The
revision is an optimistic lock, not an authorization token. User/tenant always
come from the authenticated server context.
"""
from __future__ import annotations

import json
from datetime import datetime

import db
import orders
import tickets
from auth import User


class TaskConflict(ValueError):
    pass


def init_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refund_tasks (
            tenant_id TEXT NOT NULL, requester_id TEXT NOT NULL,
            thread_id TEXT NOT NULL, revision INTEGER NOT NULL,
            order_id TEXT NOT NULL, reason TEXT NOT NULL,
            intent TEXT NOT NULL, cancelled INTEGER NOT NULL DEFAULT 0,
            ticket_id TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (tenant_id, requester_id, thread_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refund_task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL, requester_id TEXT NOT NULL,
            thread_id TEXT NOT NULL, revision INTEGER NOT NULL,
            snapshot TEXT NOT NULL, created_at TEXT NOT NULL
        )
    """)


def _scope(user: User, thread_id: str) -> tuple:
    if not user.username or not user.tenant_id or not thread_id:
        raise TaskConflict("缺少已认证的任务归属")
    return user.tenant_id, user.username, thread_id


def _read(conn, scope: tuple) -> dict | None:
    row = conn.execute(
        "SELECT * FROM refund_tasks WHERE tenant_id=? AND requester_id=? AND thread_id=?",
        scope,
    ).fetchone()
    return dict(row) if row else None


def get(user: User, thread_id: str) -> dict:
    scope = _scope(user, thread_id)
    with db.session(tickets.DB_PATH) as conn:
        task = _read(conn, scope) or dict(
            zip(("tenant_id", "requester_id", "thread_id"), scope),
            revision=0, order_id="", reason="", intent="consult",
            cancelled=0, ticket_id=None, updated_at=None,
        )
        ticket = conn.execute(
            "SELECT status, refund_id, required_role FROM refund_tickets WHERE id=? AND tenant_id=? AND requester_id=?",
            (task["ticket_id"], user.tenant_id, user.username),
        ).fetchone() if task["ticket_id"] else None
    task["missing_fields"] = [k for k in ("order_id", "reason") if not task[k]]
    task["status"] = (
        "cancelled" if task["cancelled"] else ticket["status"] if ticket else
        "consulting" if task["intent"] == "consult" else
        "collecting" if task["missing_fields"] else "ready"
    )
    task["refund_id"] = ticket["refund_id"] if ticket else None
    task["required_role"] = ticket["required_role"] if ticket else None
    task["next_action"] = (
        "complete" if task["status"] == "executed" else
        "await_approver" if task["status"] == "pending_approval" and task["required_role"] == "approver" else
        "await_customer" if task["status"] == "pending_approval" else
        "await_execution" if task["status"] in ("approved", "executing") else
        "ask_order" if task["status"] == "collecting" and not task["order_id"] else
        "ask_reason" if task["status"] == "collecting" else
        "submit_for_approval" if task["status"] == "ready" else "consult"
    )
    return task


def update(user: User, thread_id: str, *, expected_revision: int,
           order_id: str | None = None, reason: str | None = None,
           intent: str | None = None, cancel: bool = False,
           source: str = "service", turn_id: str | None = None) -> dict:
    scope = _scope(user, thread_id)
    if expected_revision < 0 or intent not in (None, "consult", "apply"):
        raise TaskConflict("任务版本或意图无效")
    if order_id is not None:
        order_id = order_id.strip()
        if len(order_id) > 64:
            raise TaskConflict("订单号过长")
        if order_id:
            orders.get_order(user, order_id)  # Validate before storing an entity.
    if reason is not None:
        reason = " ".join(reason.split())
        if len(reason) > 500:
            raise TaskConflict("退款原因过长")
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        old = _read(conn, scope)
        revision = old["revision"] if old else 0
        if expected_revision != revision:
            raise TaskConflict("任务已变化，请重新读取后再修改")
        current = old or dict(order_id="", reason="", intent="consult", cancelled=0)
        new_order = current["order_id"] if order_id is None else order_id
        changed_order = new_order != current["order_id"]
        new_reason = reason if reason is not None else ("" if changed_order else current["reason"])
        new_intent = intent or current["intent"]
        # Merely filling a field must not silently reopen a withdrawn task.
        # Reopening requires an explicit new apply intent.
        cancelled = int(cancel or (current["cancelled"] and intent != "apply"))
        values = (new_order, new_reason, new_intent, cancelled)
        if old and values == tuple(current[k] for k in ("order_id", "reason", "intent", "cancelled")):
            return get(user, thread_id)
        # Approval and edit race on the SAME database write lock. If approval
        # wins, editing fails; if edit wins, the old approval becomes rejected.
        active = conn.execute("""
            SELECT status FROM refund_tickets
            WHERE tenant_id=? AND requester_id=? AND thread_id=?
              AND status IN ('approved', 'executing') LIMIT 1
        """, scope).fetchone()
        if active:
            raise TaskConflict("退款已批准或正在执行，不能修改或撤回")
        if cancel and old and old.get("ticket_id"):
            completed = conn.execute(
                "SELECT 1 FROM refund_tickets WHERE id=? AND status='executed'",
                (old["ticket_id"],),
            ).fetchone()
            if completed:
                raise TaskConflict("退款已经执行，不能撤回")
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("""
            UPDATE refund_tickets SET status='rejected', decision='reject',
                decision_reason=?, operator=?, updated_at=?
            WHERE tenant_id=? AND requester_id=? AND thread_id=? AND status='pending_approval'
        """, ("用户撤回任务" if cancel else "任务已修改，原审批失效", user.username, now, *scope))
        conn.execute("""
            INSERT INTO refund_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(tenant_id, requester_id, thread_id) DO UPDATE SET
                revision=excluded.revision, order_id=excluded.order_id,
                reason=excluded.reason, intent=excluded.intent,
                cancelled=excluded.cancelled, ticket_id=NULL, updated_at=excluded.updated_at
        """, (*scope, revision + 1, *values, now))
        conn.execute("""
            INSERT INTO refund_task_events
            (tenant_id, requester_id, thread_id, revision, snapshot, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (*scope, revision + 1, json.dumps(dict(order_id=new_order, reason=new_reason,
                  intent=new_intent, cancelled=cancelled, source=source, turn_id=turn_id), ensure_ascii=False), now))
    return get(user, thread_id)


def history(user: User, thread_id: str, *, after: int = 0, limit: int = 50) -> list[dict]:
    with db.session(tickets.DB_PATH) as conn:
        rows = conn.execute("""
            SELECT id, revision, snapshot, created_at FROM refund_task_events
            WHERE tenant_id=? AND requester_id=? AND thread_id=? AND id>?
            ORDER BY id LIMIT ?
        """, (*_scope(user, thread_id), after, min(max(limit, 1), 100))).fetchall()
    return [dict(id=r["id"], revision=r["revision"], snapshot=json.loads(r["snapshot"]),
                 created_at=r["created_at"]) for r in rows]


def guard_action(conn, *, tenant_id: str, requester_id: str, thread_id: str,
                 order_id: str, reason: str, task_revision: int | None) -> dict | None:
    """Called INSIDE ticket creation's write transaction; legacy callers allowed."""
    task = _read(conn, (tenant_id, requester_id, thread_id))
    if task is None:
        if task_revision is not None:
            raise TaskConflict("任务不存在，请先登记退款信息")
        return None
    if task_revision != task["revision"]:
        raise TaskConflict("任务版本已变化，请重新读取并确认")
    if task["cancelled"] or task["intent"] != "apply":
        raise TaskConflict("当前任务已取消或仅为咨询，不能创建审批")
    if not task["reason"] or not task["order_id"]:
        raise TaskConflict("订单号和退款原因尚未补全")
    if task["order_id"] != order_id or task["reason"] != " ".join(reason.split()):
        raise TaskConflict("拟执行动作与当前任务信息不一致")
    return task


def bind_ticket(conn, task: dict | None, ticket_id: str) -> None:
    if task:
        conn.execute("UPDATE refund_tickets SET task_revision=? WHERE id=?", (task["revision"], ticket_id))
        conn.execute("""
            UPDATE refund_tasks SET ticket_id=?
            WHERE tenant_id=? AND requester_id=? AND thread_id=? AND revision=?
        """, (ticket_id, task["tenant_id"], task["requester_id"], task["thread_id"], task["revision"]))
