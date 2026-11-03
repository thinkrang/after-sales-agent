import copy

import pytest

import task_eval
from task_scenarios import CASES, cases, fingerprint
from task_eval_model import reason_matches


def test_scenarios_are_unique_and_partitions_do_not_overlap():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    assert {c["id"] for c in cases("dev")}.isdisjoint(c["id"] for c in cases("holdout"))
    assert len(cases("dev")) == 8 and len(cases("holdout")) == 4
    assert len(fingerprint()) == 64
    for case in CASES:
        assert case["steps"]
        assert case["decision"] in (None, "approve", "reject", "manager")
        if case["decision"]:
            assert case["expect_order"] and case["expect_reason"]


def report(variant="structured"):
    result = task_eval.summary([
        dict(case="slot_fill", split="dev", attempt=1, passed=True, elapsed_ms=10),
        dict(case="cancel", split="holdout", attempt=1, passed=False, failure_stage="intent", elapsed_ms=30),
    ], "model")
    result.update(model="test-model", variant=variant)
    return result


def test_summary_keeps_failures_in_denominator_and_split():
    r = report()
    assert r["success_rate"] == 0.5 and r["total"] == 2
    assert r["by_split"]["holdout"] == {"total": 1, "passed": 0}
    assert r["by_case"]["cancel"]["failure_stages"] == {"intent": 1}
    assert r["failure_stages"] == {"intent": 1}


def test_negated_reason_does_not_pass_by_substring():
    assert reason_matches("商品质量问题", "质量问题")
    assert not reason_matches("不是质量问题", "质量问题")
    assert not reason_matches("尺码不合适", "质量问题")


def test_paired_comparison_includes_regressions():
    current = report()
    baseline = report("messages")
    baseline["results"][0]["passed"] = False
    baseline["results"][1]["passed"] = True
    diff = task_eval.compare(current, baseline)
    assert diff["paired_runs"] == 2
    assert diff["gained"] == [("slot_fill", 1)]
    assert diff["lost"] == [("cancel", 1)]


@pytest.mark.parametrize("change", ["dataset", "model", "subset", "duplicate", "same_variant"])
def test_comparison_rejects_incompatible_reports(change):
    current, baseline = report(), report("messages")
    if change == "dataset":
        baseline["dataset_sha256"] = "other"
    elif change == "model":
        baseline["model"] = "other"
    elif change == "subset":
        baseline["results"].pop()
    elif change == "duplicate":
        baseline["results"].append(copy.deepcopy(baseline["results"][0]))
    else:
        baseline["variant"] = "structured"
    with pytest.raises(ValueError):
        task_eval.compare(current, baseline)
