"""Task identity, revision and approval races; no model or vector database."""
from concurrent.futures import ThreadPoolExecutor

import pytest

import db
import orders
import refund_service
import refund_tasks as tasks
import tickets
from auth import User
from task_eval import SYSTEM_CASES


@pytest.fixture
def task_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    yield
    db.release_all()


@pytest.fixture
def user():
    return User("acme-manager", "客户", "acme", frozenset({"customer", "approver"}))


@pytest.mark.parametrize("case", SYSTEM_CASES, ids=lambda c: c.__name__)
def test_complete_service_tasks(task_db, case):
    case()


def test_stale_revision_cannot_overwrite_or_create_ticket(task_db, user):
    task = tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    tasks.update(user, "t", expected_revision=1, reason="尺码不合适")
    with pytest.raises(tasks.TaskConflict):
        tasks.update(user, "t", expected_revision=1, cancel=True)
    with pytest.raises(tasks.TaskConflict):
        refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason=task["reason"], task_revision=1)
    assert tasks.get(user, "t")["revision"] == 2


def test_same_payload_after_edit_still_needs_new_revision(task_db, user):
    tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    tasks.update(user, "t", expected_revision=1, reason="尺码不合适")
    tasks.update(user, "t", expected_revision=2, reason="质量问题")
    with pytest.raises(tasks.TaskConflict):
        refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason="质量问题", task_revision=1)


def test_owner_and_thread_isolation(task_db, user):
    tasks.update(user, "same", expected_revision=0, order_id="A100")
    other = User("globex-user", "其他客户", "globex", frozenset({"customer"}))
    assert tasks.get(other, "same")["revision"] == 0
    assert tasks.get(user, "other-thread")["revision"] == 0
    with pytest.raises(orders.OrderNotFound):
        tasks.update(other, "same", expected_revision=0, order_id="A100")
    assert tasks.get(other, "same")["revision"] == 0


def test_noop_does_not_invalidate_approval(task_db, user):
    tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason="质量问题", task_revision=1)
    task = tasks.update(user, "t", expected_revision=1, reason=" 质量问题 ")
    assert task["revision"] == 1 and task["ticket_id"] == ticket["id"]
    assert task["status"] == "pending_approval"


def test_approval_vs_cancel_has_one_winner(task_db, user):
    tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason="质量问题", task_revision=1)

    def approve():
        try:
            refund_service.decide(user, approval_id=ticket["id"], thread_id="t", decision="approve")
            return True
        except refund_service.WorkflowConflict:
            return False

    def cancel():
        try:
            tasks.update(user, "t", expected_revision=1, cancel=True)
            return True
        except tasks.TaskConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda f: f(), [approve, cancel]))
    assert a != b
    final = tickets.get_ticket(ticket["id"])
    assert (final["status"], tasks.get(user, "t")["cancelled"]) in {
        ("approved", 0), ("rejected", 1)
    }


def test_high_risk_requester_can_withdraw_but_not_approve(task_db):
    user = User("acme-user", "客户", "acme", frozenset({"customer"}))
    tasks.update(user, "t", expected_revision=0, order_id="A400", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="t", order_id="A400", reason="质量问题", task_revision=1)
    with pytest.raises(refund_service.ApprovalForbidden):
        refund_service.decide(user, approval_id=ticket["id"], thread_id="t", decision="approve")
    tasks.update(user, "t", expected_revision=1, cancel=True)
    assert tickets.get_ticket(ticket["id"])["status"] == "rejected"


def test_consulting_invalidates_pending_and_records_audit(task_db, user):
    tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason="质量问题", task_revision=1)
    assert tasks.update(user, "t", expected_revision=1, intent="consult")["status"] == "consulting"
    assert tickets.get_ticket(ticket["id"])["status"] == "rejected"
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_task_events").fetchone()[0] == 2


def test_cancelled_task_requires_explicit_apply_to_reopen(task_db, user):
    tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    tasks.update(user, "t", expected_revision=1, cancel=True)
    task = tasks.update(user, "t", expected_revision=2, reason="尺码不合适")
    assert task["status"] == "cancelled"
    task = tasks.update(user, "t", expected_revision=3, intent="apply")
    assert task["status"] == "ready" and not task["cancelled"]


def test_completed_task_cannot_be_marked_cancelled(task_db, user):
    task = tasks.update(user, "t", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="t", order_id="A100", reason="质量问题", task_revision=1)
    refund_service.decide(user, approval_id=ticket["id"], thread_id="t", decision="approve")
    refund_service.execute_for_ticket(ticket)
    with pytest.raises(tasks.TaskConflict):
        tasks.update(user, "t", expected_revision=task["revision"], cancel=True)
    assert tasks.get(user, "t")["status"] == "executed"
