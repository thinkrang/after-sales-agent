"""Run the actual evaluator against the graph using deterministic model replies."""
import json
import sys

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
import pytest

from evaluation import plan_eval
import plan_store
from evaluation.plan_eval import run_case
from evaluation.plan_recovery_eval import RECOVERY_CASES
from planning import refund_plan
from test_task_http import task_api


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def make_model(case, variant, user, sid, behavior):
    def call(name, sequence, args=None):
        return AIMessage(content="", tool_calls=[
            {"id": f"call_{sequence}", "name": name, "args": args or {}, "type": "tool_call"},
        ])

    def responses():
        items = [{"key": "low", "order_id": "A300", "reason": "尺码不合适", "intent": "apply"}]
        if case == "edit_while_waiting":
            items.append({"key": "high", "order_id": "A400", "reason": "质量问题", "intent": "apply"})
        yield call("get_refund_plan", 1)
        if variant == "fixed":
            name = "create_fixed_refund_plan"
            args = {"batch": {"goal": "申请退款", "tasks": items}}
        else:
            name = "create_refund_plan"
            args = {"draft": refund_plan("申请退款", items).model_dump()}
        yield call(name, 2, args)
        if case == "lost_receipt":
            if behavior == "repeat_write":
                yield call(name, 3, args)
            if behavior != "skip_read":
                yield call("get_refund_plan", 4)
            yield AIMessage(content="A300：已退款" if behavior == "wrong_status" else "A300：待确认")
            return
        yield AIMessage(content="A300 待确认，A400 待主管审批")
        yield call("get_refund_plan", 3)
        tasks = plan_store.get(user, sid)["tasks"]
        order = "A400" if behavior == "wrong_branch" else "A300"
        task = next(t for t in tasks if t["state"]["order_id"] == order)
        yield call("edit_refund_plan_task", 4, {"task_id": task["id"],
                   "expected_revision": task["state"]["revision"], "reason": "包装破损"})
        yield AIMessage(content="已更新退款原因")
    return ScriptedModel(messages=responses())


def evaluate(task_api, monkeypatch, tmp_path, case, variant, behavior="normal"):
    api, *_ = task_api
    monkeypatch.setitem(sys.modules, "api", api)
    selected = next(item for item in RECOVERY_CASES if item["id"] == case)
    return run_case(selected, variant, tmp_path / "eval",
                    model_factory=lambda user, sid: make_model(case, variant, user, sid, behavior))


@pytest.mark.parametrize("variant", ["fixed", "plan"])
@pytest.mark.parametrize("case", ["lost_receipt", "edit_while_waiting"])
def test_recovery_cases_complete_for_both_variants(task_api, monkeypatch, tmp_path, case, variant):
    report = evaluate(task_api, monkeypatch, tmp_path, case, variant)
    assert report["outcome"] == "passed", report
    assert report["mode"] == "scripted"
    assert report["model_calls"] == (4 if case == "lost_receipt" else 6)
    assert report["unsafe_refunds_before_approval"] == 0
    assert not report["executions"]
    assert all(report["recovery_checks"].values())


@pytest.mark.parametrize(("case", "behavior", "check"), [
    ("lost_receipt", "skip_read", "read_verified_after_fault"),
    ("lost_receipt", "repeat_write", "no_repeat_write_after_fault"),
    ("lost_receipt", "wrong_status", "reported_pending"),
    ("edit_while_waiting", "wrong_branch", "only_target_revised"),
])
def test_recovery_scorer_rejects_incorrect_actions(task_api, monkeypatch, tmp_path, case, behavior, check):
    report = evaluate(task_api, monkeypatch, tmp_path, case, "plan", behavior)
    assert report["outcome"] == "failed"
    assert report["failure_stage"] == "postcondition", report
    assert not report["recovery_checks"][check]


def test_offline_report_marks_unrun_pairs_after_provider_error(monkeypatch, tmp_path):
    report_path = tmp_path / "paired.json"
    monkeypatch.setattr(sys, "argv", ["plan_eval.py", "--suite", "recovery", "--with-model",
                                    "--repeat", "3", "--output", str(report_path)])
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PLAN_WORKER_ENABLED", "0")
    calls = []

    def blocked(case, variant, root, *, real_model):
        assert real_model
        calls.append((case["id"], variant))
        return {"case": case["id"], "variant": variant, "outcome": "blocked", "passed": False,
                "error": "synthetic provider HTTP 402", "failure_stage": "provider",
                "unsafe_refunds_before_approval": 0, "elapsed_ms": 1}

    monkeypatch.setattr(plan_eval, "run_case", blocked)
    assert plan_eval.main() == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == [("lost_receipt", "fixed")]
    assert report["planned_runs"] == 12 and report["complete"] is False
    assert report["pairs"][0]["outcomes"] == {"fixed": "blocked", "plan": "not_run"}
    assert all(set(pair["outcomes"].values()) == {"not_run"} for pair in report["pairs"][1:])
