"""Exercise the holdout runner and graders on separate scripted fixtures."""
from datetime import datetime
import json
from pathlib import Path
import sys

import pytest

import voucher_disputes
from evaluation import mixed_holdout_eval as evaluator
from evaluation.mixed_holdout_eval import grade_turn, load_dataset, run_case, run_event, write_report
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


def test_review_event_uses_current_revision_and_merchant_update_reopens_case(task_api):
    api, _, _, _ = task_api
    voucher_disputes.seed_demo()
    user = api.authenticate("acme-user", "demo1234")
    case = voucher_disputes.submit(user, "V200", "合成争议", store_code="CENTRAL",
                                   visit_at=datetime.fromisoformat("2026-11-17T04:30:00+00:00"))

    run_event("resolved_then_merchant", user, case)
    current = voucher_disputes.detail(user, case["id"])

    assert current["status"] == "pending_recheck" and current["outcome"] is None
    assert current["assessment"]["reason_code"] == "matching_redemption"
    assert current["revision"] == case["revision"] + 2
    assert sum(row["source"] == "merchant" for row in current["assessment"]["evidence"]) == 1


def test_optional_voucher_expectation_distinguishes_ungraded_from_must_be_absent():
    actual = {"tickets": [], "voucher": {"voucher_id": "V200"}, "executions": 0, "disputes": 0}
    assert "voucher_route" not in grade_turn({"tickets": []}, actual, None)
    assert grade_turn({"tickets": [], "voucher": None}, actual, None)["voucher_route"] is False


def test_dataset_rejects_unsafe_ids_and_events_without_a_dispute(tmp_path):
    path = tmp_path / "dataset.json"
    for case in ({"id": "../other", "turns": [{}]},
                 {"id": "valid", "turns": [{"event": "review_redeemed"}]}):
        path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dataset(path)


@pytest.mark.parametrize("recovers", [True, False])
def test_report_replacement_retries_without_losing_checkpoint(tmp_path, monkeypatch, recovers):
    output = tmp_path / "report.json"
    output.write_text('{"rows": [1]}', encoding="utf-8")
    replace = Path.replace
    delays = []

    def locked(path, target):
        if not recovers or not delays:
            raise PermissionError("simulated sharing violation")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(evaluator.time, "sleep", delays.append)
    if recovers:
        write_report(output, {"rows": [1, 2]})
        assert json.loads(output.read_text(encoding="utf-8")) == {"rows": [1, 2]}
        assert delays == [0.05]
    else:
        with pytest.raises(PermissionError):
            write_report(output, {"rows": [1, 2]})
        assert json.loads(output.read_text(encoding="utf-8")) == {"rows": [1]}
        assert json.loads(output.with_suffix(".json.partial").read_text(encoding="utf-8")) == {"rows": [1, 2]}
        assert len(delays) == 4
