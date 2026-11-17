"""Bounded DAG executor: short SQLite claims, parallel reads, durable waits.

At-least-once scheduling, fenced result commits, idempotent business effects.
No model or network call runs while the scheduler holds a DB transaction.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import db
import orders
import plan_store as store
import refund_service
import refund_tasks
import tickets
from auth import User

MAX_ATTEMPTS = 3
LEASE_SECONDS = 30
READ_ACTIONS = {"lookup_order", "check_policy"}


class RetryableStep(RuntimeError):
    pass


def _transition(conn, plan, step, status, *, result=None, error_code=None):
    conn.execute("""UPDATE agent_plan_steps SET status=?, result=?, error_code=?,
        run_token=NULL, lease_until=NULL WHERE plan_id=? AND step_key=?""",
        (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
         error_code, plan["id"], step["step_key"]))
    store.event(conn, plan, "step_" + status, task_id=step["task_id"], step_id=step["step_key"],
                tool=step["action"], version=step["version"], error_code=error_code,
                approval_id=(result or {}).get("approval_id"), result_ref=step["step_key"] if result else None,
                result=result)
    step["status"] = status


def _claim(user, conversation_id, *, now):
    claims = []
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        plan = store._read(conn, user, conversation_id)
        if not plan or tickets.conversation_closed(conn, user.tenant_id, user.username, conversation_id):
            return []
        branches = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM agent_plan_tasks WHERE plan_id=?", (plan["id"],))}
        states = {tid: refund_tasks.get(user, b["scope_thread_id"], _conn=conn) for tid, b in branches.items()}
        steps = {r["step_key"]: dict(r) for r in conn.execute("SELECT * FROM agent_plan_steps WHERE plan_id=? ORDER BY rowid", (plan["id"],))}
        for step in steps.values():
            status = step["status"]
            if status in store.TERMINAL:
                continue
            task = states[step["task_id"]]
            if status == "running":
                if step["lease_until"] > now:
                    continue
                _transition(conn, plan, step, "retry", error_code="lease_expired")
            deps = [steps[d]["status"] for d in json.loads(step["depends_on"])]
            if any(d in {"failed", "cancelled", "blocked"} for d in deps):
                _transition(conn, plan, step, "blocked", error_code="dependency_failed")
                continue
            if not all(d == "succeeded" for d in deps):
                continue
            if task["cancelled"]:
                _transition(conn, plan, step, "cancelled")
                continue
            action = step["action"]
            if action == "request_refund" and task["ticket_id"] and task["status"] in {
                "pending_approval", "approved", "executing", "executed"
            }:
                # Recover a crash AFTER durable ticket creation but BEFORE step completion.
                _transition(conn, plan, step, "succeeded", result={"approval_id": task["ticket_id"],
                    "status": task["status"], "required_role": task["required_role"]})
                continue
            if action == "await_approval":
                ticket_status = task["status"]
                if ticket_status in {"approved", "executing", "executed"}:
                    _transition(conn, plan, step, "succeeded", result={"approval_id": task["ticket_id"], "status": ticket_status})
                elif ticket_status in {"rejected", "failed"}:
                    _transition(conn, plan, step, "failed", error_code=ticket_status)
                elif step["status"] != "waiting_approval":
                    _transition(conn, plan, step, "waiting_approval", result={"approval_id": task["ticket_id"]})
                continue
            if action == "execute_refund" and task["status"] == "executed":
                _transition(conn, plan, step, "succeeded", result={"approval_id": task["ticket_id"], "refund_id": task["refund_id"]})
                continue
            missing = not task["order_id"] or (
                action == "request_refund" and (task["intent"] != "apply" or not task["reason"]))
            if missing:
                if step["status"] != "waiting_user":
                    _transition(conn, plan, step, "waiting_user")
                continue
            if step["next_run_at"] > now:
                continue
            if step["attempts"] >= MAX_ATTEMPTS or plan["calls_used"] >= plan["max_calls"]:
                _transition(conn, plan, step, "failed", error_code="attempt_limit" if step["attempts"] >= MAX_ATTEMPTS else "plan_budget_exhausted")
                continue
            # Only independent read steps share a batch. Writes remain serial per runner.
            if claims and (action not in READ_ACTIONS or claims[0]["action"] not in READ_ACTIONS):
                continue
            token = uuid.uuid4().hex
            conn.execute("""UPDATE agent_plan_steps SET status='running', attempts=attempts+1,
                run_token=?, lease_until=?, started_at=?, error_code=NULL WHERE plan_id=? AND step_key=?""",
                (token, now + LEASE_SECONDS, now, plan["id"], step["step_key"]))
            conn.execute("UPDATE agent_plans SET calls_used=calls_used+1 WHERE id=?", (plan["id"],))
            plan["calls_used"] += 1
            step.update(status="running", run_token=token, attempts=step["attempts"]+1,
                        task=task, scope=branches[step["task_id"]]["scope_thread_id"])
            store.event(conn, plan, "step_started", task_id=step["task_id"], step_id=step["step_key"],
                        tool=action, version=step["version"], attempt=step["attempts"], run_id=token,
                        task_revision=task["revision"])
            claims.append(step)
            if len(claims) >= 2 or action not in READ_ACTIONS:
                break
    return claims


def _execute(user, step):
    task = step["task"]
    action = step["action"]
    if action in READ_ACTIONS:
        order = orders.get_order(user, task["order_id"])
        if action == "lookup_order":
            return {k: order[k] for k in ("order_id", "product_name", "amount_cents", "status")}
        policy = orders.refund_policy(order)
        if not policy["eligible"]:
            raise orders.RefundDenied(policy["reason"])
        return policy
    if action == "request_refund":
        ticket, _ = refund_service.prepare_ticket(user, thread_id=step["scope"], order_id=task["order_id"],
            reason=task["reason"], task_revision=task["revision"], approval_protocol="detached")
        return {"approval_id": ticket["id"], "status": ticket["status"], "required_role": ticket["required_role"]}
    if action == "execute_refund":
        ticket = tickets.get_ticket(task["ticket_id"], tenant_id=user.tenant_id, requester_id=user.username)
        result = refund_service.execute_for_ticket(ticket)
        if result["status"] in {"approved", "executing"}:
            raise RetryableStep("execution_pending")
        if result["status"] != "executed":
            raise orders.RefundDenied("执行失败")
        return {"approval_id": result["id"], "refund_id": result["refund_id"]}
    raise ValueError("unknown_action")


def _run_step(user, step):
    start = time.monotonic()
    try:
        result, error, retryable = _execute(user, step), None, False
    except (TimeoutError, ConnectionError, sqlite3.OperationalError, RetryableStep):
        result, error, retryable = None, "temporarily_unavailable", True
    except orders.OrderNotFound:
        result, error, retryable = None, "not_found", False
    except orders.RefundDenied:
        result, error, retryable = None, "denied", False
    except (refund_tasks.TaskConflict, refund_service.WorkflowConflict):
        result, error, retryable = None, "task_conflict", False
    except Exception:
        logging.exception("Plan step failed: %s/%s", step["plan_id"], step["step_key"])
        result, error, retryable = None, "internal_error", False
    duration = round((time.monotonic() - start) * 1000, 3)
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT * FROM agent_plan_steps WHERE plan_id=? AND step_key=?", (step["plan_id"], step["step_key"])).fetchone()
        plan = dict(conn.execute("SELECT * FROM agent_plans WHERE id=?", (step["plan_id"],)).fetchone())
        if current["run_token"] != step["run_token"] or current["version"] != step["version"]:
            store.event(conn, plan, "stale_result_discarded", task_id=step["task_id"], step_id=step["step_key"],
                        version=step["version"], run_id=step["run_token"])
            return
        status = "succeeded" if error is None else "retry" if retryable and step["attempts"] < MAX_ATTEMPTS else "failed"
        _transition(conn, plan, step, status, result=result, error_code=error)
        conn.execute("UPDATE agent_plan_steps SET duration_ms=?, next_run_at=? WHERE plan_id=? AND step_key=?",
                     (duration, time.time() + 2 ** (step["attempts"] - 1) if status == "retry" else 0, step["plan_id"], step["step_key"]))
        store.event(conn, plan, "step_attempt_finished", task_id=step["task_id"], step_id=step["step_key"],
                    tool=step["action"], version=step["version"], attempt=step["attempts"], duration_ms=duration,
                    error_code=error, retryable=retryable, run_id=step["run_token"],
                    approval_id=(result or {}).get("approval_id"), result_ref=step["step_key"] if result else None)


def run(user, conversation_id, *, max_waves=12):
    """Advance runnable work; return promptly at approval/user/backoff boundaries."""
    for _ in range(min(max_waves, 20)):
        claims = _claim(user, conversation_id, now=time.time())
        if not claims:
            break
        if len(claims) == 1:
            _run_step(user, claims[0])
        else:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="plan-read") as pool:
                list(pool.map(lambda step: _run_step(user, step), claims))
    return store.get(user, conversation_id)


def recover_once():
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("""SELECT DISTINCT p.* FROM agent_plans p JOIN agent_plan_steps s ON p.id=s.plan_id
            WHERE s.status NOT IN ('succeeded','failed','cancelled','blocked')
            AND NOT EXISTS (SELECT 1 FROM closed_refund_conversations c WHERE
                c.tenant_id=p.tenant_id AND c.requester_id=p.requester_id AND c.conversation_id=p.conversation_id)
            ORDER BY p.last_scanned_at, p.created_at LIMIT 100""").fetchall()
        # A large backlog of approval waits must not starve later runnable plans.
        conn.executemany("UPDATE agent_plans SET last_scanned_at=? WHERE id=?",
                         [(time.time(), r["id"]) for r in rows])
    for row in rows:
        user = User(row["requester_id"], row["requester_id"], row["tenant_id"], frozenset({"customer"}))
        run(user, row["conversation_id"])
    return len(rows)


def worker(stop):
    while not stop.is_set():
        try:
            recover_once()
        except Exception:
            logging.exception("计划恢复暂时失败，将在下次扫描重试")
        stop.wait(2)
