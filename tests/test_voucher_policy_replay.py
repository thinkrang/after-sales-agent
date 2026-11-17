"""Business-level checks on the paired replay and fixed decision time."""
import json

import pytest

from scripts.replay_voucher_policy import FIXTURES, replay


def test_policy_change_flips_recommendations_but_keeps_manual_route():
    dataset = json.loads(FIXTURES.read_text(encoding="utf-8"))
    original = json.dumps(dataset, sort_keys=True)
    report = replay(dataset, "voucher-v1", "voucher-v2")
    assert json.dumps(dataset, sort_keys=True) == original
    records = {row["id"]: row for row in report["cases"]}
    row = records["R04"]
    assert row["redemption_delta_seconds"] == 180
    assert row["baseline"]["recommendation"] == "redemption_supported"
    assert row["candidate"]["recommendation"] == "manual_review"
    assert row["baseline"]["case_status"] == row["candidate"]["case_status"] == "pending_review"
    assert records["R02"]["candidate"]["recommendation"] == "redemption_supported"
    assert records["R06"]["baseline"]["recommendation"] == "redemption_supported"
    assert records["R12"]["candidate"]["reason"] == "expired_voucher"
    assert records["R14"]["candidate"]["reason"] == "redemption_time_mismatch"
    assert records["R16-before"]["candidate"]["reason"] == "status_mismatch"
    assert records["R16-after"]["candidate"]["reason"] == "matching_redemption"
    assert report["summary"] == {
        "recommendation_changes": 4, "initial_route_changes": 0,
        "baseline": {"reference_matches": 15, "unsupported_redemption_recommendations": 2,
                     "missed_redemption_recommendations": 1, "initial_manual_cases": 17},
        "candidate": {"reference_matches": 15, "unsupported_redemption_recommendations": 0,
                      "missed_redemption_recommendations": 3, "initial_manual_cases": 17},
    }
    assert report["review_ids"] == ["R03", "R04", "R05", "R06", "R07"]


def test_replay_rejects_duplicate_cases_and_naive_decision_time():
    dataset = json.loads(FIXTURES.read_text(encoding="utf-8"))
    datasets = [
        (dict(dataset, cases=[dataset["cases"][0]] * 2), "重复"),
        (dict(dataset, cases=[dict(dataset["cases"][0], checked_at="2026-11-17T13:00:00")]), "timezone"),
    ]
    for invalid, reason in datasets:
        with pytest.raises(ValueError, match=reason):
            replay(invalid, "voucher-v1", "voucher-v2")
