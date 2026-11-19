from concurrent.futures import ThreadPoolExecutor

import pytest

import db
import orders
import refund_service
import refund_tasks
import tickets
from auth import User
from test_refund_workflow import workflow


USER = User("acme-user", "客户", "acme", frozenset({"customer"}))


def intake(order_id="A400", reason="质量问题", **kwargs):
    return refund_service.request_order_refund(
        USER, thread_id="intake", order_id=order_id, reason=reason, **kwargs,
    )


def test_concurrent_intake_returns_one_pending_approval(workflow):
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(lambda _: intake(), range(8)))
    assert len({row["id"] for row in rows}) == 1
    assert {row["status"] for row in rows} == {"pending_approval"}
    assert {row["required_role"] for row in rows} == {"approver"}
    assert len(refund_tasks.history(USER, "intake")) == 1
    assert intake(expected_revision=0)["id"] == rows[0]["id"]
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 0
    with pytest.raises(refund_service.ApprovalForbidden):
        refund_service.decide(USER, approval_id=rows[0]["id"], thread_id="intake", decision="approve")


def test_wrong_owner_and_ineligible_order_leave_no_draft(workflow):
    with pytest.raises(orders.OrderNotFound):
        intake("A100")
    with db.session(orders.ORDERS_DB) as conn:
        conn.execute("UPDATE orders SET refundable_until='2000-01-01 00:00:00' WHERE order_id='A300'")
    with pytest.raises(orders.RefundDenied, match="超过七天"):
        intake("A300")
    assert refund_tasks.get(USER, "intake")["revision"] == 0
    assert refund_tasks.history(USER, "intake") == []
    assert tickets.list_tickets() == []


def test_failed_correction_keeps_old_approval_and_revision(workflow, monkeypatch):
    old = intake()
    with pytest.raises(refund_tasks.TaskConflict):
        intake("A300", "尺码不合适")
    create = tickets.create_ticket

    def fail_after_insert(*args, **kwargs):
        create(*args, **kwargs)
        raise OSError("simulated failure before commit")

    with monkeypatch.context() as patch:
        patch.setattr(tickets, "create_ticket", fail_after_insert)
        with pytest.raises(OSError):
            intake("A300", "尺码不合适", expected_revision=1)
    assert refund_tasks.get(USER, "intake")["ticket_id"] == old["id"]
    assert tickets.get_ticket(old["id"])["status"] == "pending_approval"
    assert len(tickets.list_tickets()) == 1
    new = intake("A300", "尺码不合适", expected_revision=1)
    assert new["required_role"] == "customer" and new["task_revision"] == 2
    assert intake("A300", "尺码不合适", expected_revision=1)["id"] == new["id"]
    assert tickets.get_ticket(old["id"])["status"] == "rejected"
    with pytest.raises(refund_tasks.TaskConflict):
        intake("A400", expected_revision=1)
