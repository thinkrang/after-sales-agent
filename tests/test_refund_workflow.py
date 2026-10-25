"""Refund authorization, approval binding and idempotent execution tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import db
import orders
import refund_service
import tickets
from auth import User


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    yield
    db.release_all()


@pytest.fixture
def manager():
    return User(
        "acme-manager",
        "ACME 售后主管",
        "acme",
        frozenset({"customer", "approver", "knowledge_admin"}),
    )


@pytest.fixture
def other_user():
    return User(
        "globex-user",
        "GLOBEX 客户",
        "globex",
        frozenset({"customer"}),
    )


def test_order_is_scoped_to_authenticated_user(manager, other_user, workflow):
    assert orders.get_order(manager, "A100")["order_id"] == "A100"
    with pytest.raises(orders.OrderNotFound):
        orders.get_order(other_user, "A100")


def test_direct_execution_without_approval_fails(manager, workflow):
    refund_service.prepare_ticket(
        manager,
        thread_id="thread-1",
        order_id="A100",
        reason="尺码不合适",
    )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-1",
            order_id="A100",
            reason="尺码不合适",
        )
    assert tickets.find_executed(
        "A100", tenant_id="acme", requester_id="acme-manager"
    ) is None


def test_approval_is_bound_to_exact_order_reason_and_thread(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-2",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-2",
        decision="approve",
    )

    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-2",
            order_id="A100",
            reason="商品有质量问题",
        )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="another-thread",
            order_id="A100",
            reason="尺码不合适",
        )


def test_approved_refund_executes_and_retry_returns_same_refund(
    manager, workflow
):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-3",
        decision="approve",
    )

    first = refund_service.execute(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    second = refund_service.execute(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )

    assert first["status"] == tickets.EXECUTED
    assert first["refund_id"] == second["refund_id"]
    assert first["execution_id"] == first["id"]
    matched, _ = refund_service.find_action_ticket(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    assert matched["id"] == first["id"]
    assert matched["status"] == tickets.EXECUTED


def test_rejected_refund_cannot_execute(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-4",
        order_id="A200",
        reason="不想要了",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-4",
        decision="reject",
    )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-4",
            order_id="A200",
            reason="不想要了",
        )


def test_duplicate_approval_submission_is_rejected(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-5",
        order_id="A200",
        reason="不想要了",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-5",
        decision="approve",
    )
    with pytest.raises(refund_service.WorkflowConflict):
        refund_service.decide(
            manager,
            approval_id=pending["id"],
            thread_id="thread-5",
            decision="approve",
        )


def test_execution_recovers_after_gateway_succeeded_but_ticket_not_updated(
    manager, workflow
):
    pending, order = refund_service.prepare_ticket(
        manager,
        thread_id="thread-6",
        order_id="A200",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-6",
        decision="approve",
    )
    payload_hash = refund_service.action_payload_hash(
        user=manager,
        thread_id="thread-6",
        order_id="A200",
        amount_cents=order["amount_cents"],
        reason="尺码不合适",
    )
    claimed = tickets.acquire_execution(
        "thread-6",
        tenant_id="acme",
        requester_id="acme-manager",
        order_id="A200",
        payload_hash=payload_hash,
    )
    assert claimed["status"] == tickets.EXECUTING

    # Simulate the downstream refund succeeding before the process dies.
    execution = orders.execute_refund(
        user=manager,
        order_id="A200",
        amount_cents=order["amount_cents"],
        idempotency_key=pending["id"],
    )
    assert execution["refund_id"]

    repaired = refund_service.reconcile_executing(
        manager,
        thread_id="thread-6",
        order_id="A200",
        reason="尺码不合适",
    )
    assert repaired["status"] == tickets.EXECUTED
    assert repaired["refund_id"] == execution["refund_id"]


def test_reconciliation_worker_finishes_approved_ticket(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-recover-approved",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-recover-approved",
        decision="approve",
    )

    recoverable = tickets.list_recoverable()
    assert [row["id"] for row in recoverable] == [pending["id"]]

    recovered = refund_service.reconcile_ticket(recoverable[0])
    assert recovered["status"] == tickets.EXECUTED
    assert recovered["refund_id"]


def test_concurrent_downstream_retry_creates_one_refund(manager, workflow):
    key = "fixed-idempotency-key"

    def run_once(_):
        return orders.execute_refund(
            user=manager,
            order_id="A100",
            amount_cents=12900,
            idempotency_key=key,
        )["refund_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        refund_ids = list(pool.map(run_once, range(16)))

    assert len(set(refund_ids)) == 1
