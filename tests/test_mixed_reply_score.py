"""Audit packet isolation, coverage, and evidence checks for reply scoring."""

import hashlib
import json
import sys

import pytest

from evaluation.mixed_reply_score import build_packet, load_turns, main, read_json, score_audit


PAID = "voucher_refund_already_paid"
WAIT = "voucher_wait_for_review"


@pytest.fixture
def example():
    dataset = {"id": "sample", "protocol": "v2", "cases": [
        {"id": "needs_support", "turns": [{"text": "怎么处理券？", "assertions": {PAID: False}, "rubric": "评测分支"}]},
        {"id": "read_only", "turns": [
            {"text": "还有进度吗？", "assertions": {WAIT: True}},
            {"text": "到店时间是 {arrival}。", "assertions": {PAID: False}},
        ]},
    ]}
    raw = {"dataset": "sample", "protocol": "v2", "dataset_sha256": "hash", "complete": True,
           "arrival": "15:00", "rows": [
               {"case": "needs_support", "passed": True, "rows": [
                   {"user": "怎么处理券？", "answer": "已经退款。", "checks": {"business": True}},
               ]},
               {"case": "read_only", "passed": False, "rows": [
                   {"user": "还有进度吗？", "answer": "当前待人工复核。", "checks": {"business": False}},
                   {"user": "到店时间是 15:00。", "answer": "这个说法不确定。", "checks": {"business": True}},
               ]},
           ]}
    return dataset, raw


def test_packet_excludes_backend_expectations(example):
    dataset, raw = example
    packet = build_packet(load_turns(dataset, raw, "hash"))
    assert [row["turn_id"] for row in packet["rows"]] == ["T01", "T02", "T03"]
    assert all(set(row) == {"turn_id", "user", "answer", "claims"} for row in packet["rows"])
    assert "needs_support" not in repr(packet) and "评测分支" not in repr(packet)
    assert list(packet["rows"][0]["claims"]) == [PAID]
    assert isinstance(packet["rows"][0]["claims"][PAID], str)


def test_business_and_reply_scores_are_independent_and_null_is_unresolved(example):
    dataset, raw = example
    audit = {"rows": [
        {"turn_id": "T03", "claims": {PAID: {"observed": None, "evidence": "不确定"}}},
        {"turn_id": "T01", "claims": {PAID: {"observed": True, "evidence": "已经退款"}}},
        {"turn_id": "T02", "claims": {WAIT: {"observed": True, "evidence": "待人工复核"}}},
    ]}
    report = score_audit(load_turns(dataset, raw, "hash"), audit, raw)
    assert report["business"] == {"passed_cases": 1, "total_cases": 2, "passed_turns": 2, "total_turns": 3}
    assert report["reply"]["passed_cases"] == 0 and report["reply"]["passed_turns"] == 1
    assert report["reply"]["unresolved_claims"] == report["reply"]["unresolved_turns"] == 1
    assert report["reply"]["severe_false_positive_claims"] == 1
    assert report["rows"][0]["severe_false_positives"] == [PAID]
    assert report["rows"][1]["reply_passed"] and not report["rows"][1]["business_passed"]
    assert report["rows"][2]["claims"][PAID]["result"] == "unresolved"


@pytest.mark.parametrize(("annotation", "error"), [
    ({"observed": True, "evidence": "退款已到账"}, "exact answer substring"),
    ({"observed": True, "evidence": ""}, "requires evidence"),
    ({"observed": 1, "evidence": "已经退款"}, "true, false, or null"),
    ({"evidence": "已经退款"}, "expected observed and evidence"),
])
def test_incorrect_evidence_and_annotation_shape_are_rejected(example, annotation, error):
    dataset, raw = example
    audit = {"rows": [
        {"turn_id": "T01", "claims": {PAID: annotation}},
        {"turn_id": "T02", "claims": {WAIT: {"observed": False, "evidence": ""}}},
        {"turn_id": "T03", "claims": {PAID: {"observed": False, "evidence": ""}}},
    ]}
    with pytest.raises(ValueError, match=error):
        score_audit(load_turns(dataset, raw, "hash"), audit, raw)


def test_hash_completion_and_raw_turn_alignment_are_required(example):
    dataset, raw = example
    with pytest.raises(ValueError, match="dataset hash"):
        load_turns(dataset, raw, "wrong")
    with pytest.raises(ValueError, match="must be complete"):
        load_turns(dataset, dict(raw, complete=False), "hash")
    raw["rows"][1]["rows"][1]["user"] = "不属于这组用例"
    with pytest.raises(ValueError, match="prompt does not match"):
        load_turns(dataset, raw, "hash")


def test_duplicate_and_missing_audit_turns_are_rejected(example):
    dataset, raw = example
    turns = load_turns(dataset, raw, "hash")
    row = {"turn_id": "T01", "claims": {PAID: {"observed": False, "evidence": ""}}}
    with pytest.raises(ValueError, match="duplicate audit turn"):
        score_audit(turns, {"rows": [row, row]}, raw)
    with pytest.raises(ValueError, match="exactly cover"):
        score_audit(turns, {"rows": [row]}, raw)


def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text('{"observed":true,"observed":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_json(path)


def test_packet_cli_verifies_raw_bytes_and_preserves_previous_output(tmp_path, monkeypatch, example):
    dataset, raw = example
    input_path, run_path, output_path = (tmp_path / name for name in ("fixture.json", "run.json", "packet.json"))
    input_path.write_text(json.dumps(dataset), encoding="utf-8")
    raw["dataset_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    run_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mixed_reply_score.py", "packet", "--dataset", str(input_path),
                                      "--run", str(run_path), "--output", str(output_path)])
    main()
    previous = output_path.read_bytes()
    assert len(json.loads(previous)["rows"]) == 3
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and output_path.read_bytes() == previous
