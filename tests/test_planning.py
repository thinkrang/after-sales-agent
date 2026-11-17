"""Real SQLite + refund gateway fault/race tests; no model or vector DB."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

import db
import orders
import orchestrator as engine
import plan_store as store
import refund_service
import refund_tasks
import tickets
from auth import User
from planning import PlanDraft, refund_plan

USER = User("acme-user", "客户", "acme", frozenset({"customer"}))
MANAGER = User("acme-manager", "主管", "acme", frozenset({"customer", "approver"}))
OTHER = User("globex-user", "其他租户", "globex", frozenset({"customer"}))


@pytest.fixture
def plan_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    yield tmp_path
    db.release_all()


def draft():
    return refund_plan("处理两笔退款，一笔等待时继续另一笔", [
        dict(key="low", order_id="A300", reason="尺码不合适", intent="apply"),
        dict(key="high", order_id="A400", reason="质量问题", intent="apply"),
    ])


def start():
    store.create(USER, "chat", draft(), request_id="request-1")
    return engine.run(USER, "chat")


def branch(plan, key):
    return next(t for t in plan["tasks"] if t["task_key"] == key)


def step(plan, key):
    return next(s for s in plan["steps"] if s["step_key"] == key)


def count_refunds():
    with db.session(orders.ORDERS_DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]


def approve(plan, key, actor):
    state = branch(plan, key)["state"]
    refund_service.decide(actor, approval_id=state["ticket_id"], decision="approve")
    return engine.run(USER, "chat")


def test_independent_approvals_then_exactly_one_refund_per_order(plan_db):
    plan = start()
    assert plan["status"] == "waiting_approval"
    assert len({t["scope_thread_id"] for t in plan["tasks"]}) == 2
    assert count_refunds() == 0
    with pytest.raises(refund_service.ApprovalForbidden):
        approve(plan, "high", USER)
    plan = approve(plan, "low", USER)
    assert branch(plan, "low")["state"]["status"] == "executed"
    assert branch(plan, "high")["state"]["status"] == "pending_approval"
    plan = approve(plan, "high", MANAGER)
    assert plan["status"] == "completed"
    assert count_refunds() == 2
    engine.recover_once()
    engine.run(USER, "chat")
    assert count_refunds() == 2
    assert all(e["request_id"] == "request-1" for e in store.history(USER, "chat"))


@pytest.mark.parametrize("mutation", ["cycle", "missing", "unsafe_tool", "missing_guard", "duplicate", "foreign_task", "extra_authority"])
def test_untrusted_plans_are_rejected(plan_db, mutation):
    raw = draft().model_dump()
    if mutation == "cycle":
        raw["steps"][0]["depends_on"] = ["low_policy"]
    elif mutation == "missing":
        raw["steps"][0]["depends_on"] = ["absent"]
    elif mutation == "unsafe_tool":
        raw["steps"][2]["action"] = "execute_refund"
    elif mutation == "missing_guard":
        raw["steps"][2]["depends_on"] = []
    elif mutation == "duplicate":
        raw["tasks"][1]["order_id"] = "A300"
    elif mutation == "foreign_task":
        raw["steps"][0]["task"] = "foreign"
    else:
        raw["tasks"][0]["approved"] = True
    with pytest.raises(ValidationError):
        PlanDraft.model_validate(raw)


def test_plan_ownership_is_checked_before_any_mutation(plan_db):
    with pytest.raises(orders.OrderNotFound):
        store.create(OTHER, "chat", draft())
    assert store.get(OTHER, "chat") is None
    plan = start()
    assert store.get(OTHER, "chat") is None
    assert store.history(OTHER, "chat") == []
    with pytest.raises(store.PlanConflict):
        store.edit(OTHER, "chat", branch(plan, "low")["id"], expected_revision=1, cancel=True)
    assert count_refunds() == 0


def test_reason_edit_preserves_reads_and_other_completed_branch(plan_db):
    plan = approve(start(), "low", USER)
    high = branch(plan, "high")
    old_ticket = high["state"]["ticket_id"]
    old_calls = plan["calls_used"]
    store.edit(USER, "chat", high["id"], expected_revision=1, reason="包装破损", request_id="request-2")
    assert tickets.get_ticket(old_ticket)["status"] == "rejected"
    with pytest.raises(refund_service.WorkflowConflict):
        refund_service.decide(MANAGER, approval_id=old_ticket, decision="approve")
    plan = engine.run(USER, "chat")
    assert step(plan, "high_order")["version"] == 1
    assert step(plan, "high_policy")["version"] == 1
    assert step(plan, "high_request")["version"] == 2
    assert plan["calls_used"] == old_calls + 1
    assert branch(plan, "low")["state"]["status"] == "executed"
    assert branch(plan, "high")["state"]["ticket_id"] != old_ticket
    old_results = [e for e in store.history(USER, "chat") if e["step_id"] == "high_request"
                   and e["kind"] == "step_succeeded" and e["detail"]["version"] == 1]
    assert old_results[0]["detail"]["result"]["approval_id"] == old_ticket
    assert count_refunds() == 1
    with pytest.raises(refund_tasks.TaskConflict):
        store.edit(USER, "chat", high["id"], expected_revision=1, cancel=True)
    with pytest.raises(store.PlanConflict):
        store.edit(USER, "chat", branch(plan, "low")["id"], expected_revision=1, cancel=True)


def test_change_order_clears_reason_and_waits_without_blocking_sibling(plan_db):
    plan = start()
    high = branch(plan, "high")
    store.edit(USER, "chat", high["id"], expected_revision=1, order_id="A600")
    plan = engine.run(USER, "chat")
    assert branch(plan, "high")["state"]["reason"] == ""
    assert step(plan, "high_request")["status"] == "waiting_user"
    plan = approve(plan, "low", USER)
    assert count_refunds() == 1
    store.edit(USER, "chat", high["id"], expected_revision=2, reason="质量问题")
    plan = engine.run(USER, "chat")
    assert branch(plan, "high")["state"]["status"] == "pending_approval"


def test_consultation_and_missing_order_cannot_create_approval(plan_db):
    p = refund_plan("咨询两笔", [dict(key="a", order_id="A300", intent="consult"), dict(key="b", reason="质量问题", intent="apply")])
    store.create(USER, "chat", p)
    plan = engine.run(USER, "chat")
    assert plan["status"] == "waiting_user"
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets").fetchone()[0] == 0


def test_cancel_one_pending_branch_only(plan_db):
    plan = start()
    high = branch(plan, "high")
    store.edit(USER, "chat", high["id"], expected_revision=1, cancel=True)
    plan = approve(store.get(USER, "chat"), "low", USER)
    assert branch(plan, "high")["state"]["status"] == "cancelled"
    assert count_refunds() == 1


def test_parallel_reads_have_actual_overlap(plan_db, monkeypatch):
    store.create(USER, "chat", draft())
    barrier = threading.Barrier(2)
    original = engine._execute
    def overlap(user, item):
        if item["action"] == "lookup_order":
            barrier.wait(timeout=5)
        return original(user, item)
    monkeypatch.setattr(engine, "_execute", overlap)
    plan = engine.run(USER, "chat")
    assert step(plan, "low_order")["status"] == step(plan, "high_order")["status"] == "succeeded"


def test_two_workers_claim_each_step_once(plan_db):
    store.create(USER, "chat", draft())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: engine.run(USER, "chat"), range(2)))
    plan = engine.run(USER, "chat")
    assert plan["status"] == "waiting_approval"
    assert plan["calls_used"] == 6
    assert all(s["attempts"] <= 1 for s in plan["steps"])
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets").fetchone()[0] == 2


def test_ticket_and_gateway_results_survive_lost_step_commit(plan_db):
    store.create(USER, "chat", draft())
    engine.run(USER, "chat", max_waves=2)
    claimed = engine._claim(USER, "chat", now=time.time())[0]
    result = engine._execute(USER, claimed)  # process dies before _run_step commits
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("UPDATE agent_plan_steps SET lease_until=0 WHERE status='running'")
    plan = engine.run(USER, "chat")
    assert branch(plan, "low")["state"]["ticket_id"] == result["approval_id"]
    refund_service.decide(USER, approval_id=result["approval_id"], decision="approve")
    claimed = engine._claim(USER, "chat", now=time.time())[0]
    assert claimed["action"] == "execute_refund"
    engine._execute(USER, claimed)  # gateway committed, scheduler acknowledgement lost
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("UPDATE agent_plan_steps SET lease_until=0 WHERE status='running'")
    engine.recover_once()
    assert count_refunds() == 1
    assert step(store.get(USER, "chat"), "low_request$execute")["status"] == "succeeded"


def test_stale_claim_cannot_create_approval_after_edit(plan_db):
    store.create(USER, "chat", draft())
    plan = engine.run(USER, "chat", max_waves=2)
    claimed = engine._claim(USER, "chat", now=time.time())[0]
    store.edit(USER, "chat", branch(plan, "low")["id"], expected_revision=1, reason="改成质量问题")
    engine._run_step(USER, claimed)
    plan = engine.run(USER, "chat")
    assert branch(plan, "low")["state"]["reason"] == "改成质量问题"
    assert step(plan, "low_request")["version"] == 2
    assert any(e["kind"] == "stale_result_discarded" for e in store.history(USER, "chat"))
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets WHERE reason='尺码不合适'").fetchone()[0] == 0


def test_transient_retry_backoff_and_lifetime_budget(plan_db, monkeypatch):
    store.create(USER, "chat", draft())
    original = engine._execute
    def unavailable(user, item):
        if item["task"]["order_id"] == "A400":
            raise TimeoutError()
        return original(user, item)
    monkeypatch.setattr(engine, "_execute", unavailable)
    plan = engine.run(USER, "chat")
    assert step(plan, "high_order")["status"] == "retry"
    assert step(plan, "high_order")["attempts"] == 1
    engine.run(USER, "chat")
    assert step(store.get(USER, "chat"), "high_order")["attempts"] == 1
    for _ in range(2):
        with db.session(tickets.DB_PATH) as conn:
            conn.execute("UPDATE agent_plan_steps SET next_run_at=0")
        plan = engine.run(USER, "chat")
    assert step(plan, "high_order")["status"] == "failed"
    assert step(plan, "high_order")["attempts"] == 3
    assert branch(plan, "low")["state"]["status"] == "pending_approval"
    calls = plan["calls_used"]
    store.edit(USER, "chat", branch(plan, "high")["id"], expected_revision=1, reason="包装破损")
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("UPDATE agent_plans SET max_calls=?", (calls,))
    plan = engine.run(USER, "chat")
    assert plan["calls_used"] == calls
    assert step(plan, "high_order")["error_code"] == "plan_budget_exhausted"


def test_idempotent_create_and_replan_limit(plan_db):
    original = store.create(USER, "chat", draft())
    assert store.create(USER, "chat", draft())["id"] == original["id"]
    for i in range(store.MAX_REPLANS):
        store.edit(USER, "chat", branch(original, "high")["id"], expected_revision=i+1, reason=f"新原因{i}")
    with pytest.raises(store.PlanConflict):
        store.edit(USER, "chat", branch(original, "high")["id"], expected_revision=store.MAX_REPLANS+1, reason="超限")


def test_approval_and_replanning_race_share_transaction_boundary(plan_db):
    plan = start()
    high = branch(plan, "high")
    barrier = threading.Barrier(2)
    def decide():
        barrier.wait(timeout=5)
        try:
            refund_service.decide(MANAGER, approval_id=high["state"]["ticket_id"], decision="approve")
            return "approved"
        except refund_service.WorkflowConflict:
            return "lost"
    def edit():
        barrier.wait(timeout=5)
        try:
            store.edit(USER, "chat", high["id"], expected_revision=1, reason="包装破损")
            return "edited"
        except (store.PlanConflict, refund_tasks.TaskConflict):
            return "lost"
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(decide), pool.submit(edit)
        results = {a.result(), b.result()}
    assert results in ({"approved", "lost"}, {"edited", "lost"})
    plan = engine.run(USER, "chat")
    assert count_refunds() == (1 if "approved" in results else 0)
    assert branch(plan, "low")["state"]["status"] == "pending_approval"


def test_recovery_backlog_of_waiting_plans_cannot_starve_new_work(plan_db):
    waiting = refund_plan("等待用户补充订单", [dict(key="a", intent="apply")])
    for index in range(100):
        store.create(USER, f"waiting-{index}", waiting)
    store.create(USER, "last", draft())
    engine.recover_once()
    assert store.get(USER, "last")["calls_used"] == 0
    engine.recover_once()
    assert store.get(USER, "last")["status"] == "waiting_approval"
