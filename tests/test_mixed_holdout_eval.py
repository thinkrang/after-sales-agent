"""Exercise the holdout runner and graders on separate scripted fixtures."""
import sys

import pytest

import voucher_disputes
from mixed_holdout_eval import run_case
from test_task_http import script, task_api


@pytest.mark.parametrize(("fault", "order", "reason", "role"), [
    ("order_read_timeout", "A300", "尺码不合适", "customer"),
    ("refund_receipt_lost", "A400", "质量问题", "approver"),
])
def test_holdout_harness_grades_actual_fault_recovery(task_api, monkeypatch, fault, order, reason, role):
    api, _, _, model = task_api
    monkeypatch.setitem(sys.modules, "api", api)
    voucher_disputes.seed_demo()
    user = api.authenticate("acme-user", "demo1234")
    script(model, ("observe_voucher_branch", {"voucher_id": "V500"}),
           ("choose_voucher_step", {"expected_revision": 1, "action": "explain"}),
           ("get_order_info", {"order_id": order}),
           ("update_refund_task", {"expected_revision": 0, "order_id": order, "reason": reason, "intent": "apply"}),
           ("submit_refund_request", {"task_revision": 1, "order_id": order, "reason": reason}),
           ("get_refund_task", {}))
    result = run_case({"id": "grader-smoke", "fault": fault, "turns": [
        {"text": "故障观察器验证", "tickets": [[order, "pending_approval", reason, role]],
         "voucher": ["V500", "explain", "expired_voucher"]},
    ]}, user, model, arrival="2026-11-17T04:30:00+00:00")
    assert result["passed"], result
    assert all(result["fault_checks"].values())
    assert result["final_writes"]["refund_orders"] == [order]
