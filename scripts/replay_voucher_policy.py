"""Compare two voucher rules against the same frozen evidence snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voucher_policy import POLICIES, evaluate_redemption, initial_dispute_result, parse_aware_datetime


FIXTURES = ROOT / "docs" / "evaluation" / "voucher-policy-fixtures.json"
RECOMMENDATIONS = {"eligible_to_apply", "redemption_supported", "manual_review"}
DECISIONS = {"eligible_to_apply": "支持申请退款", "redemption_supported": "核销记录吻合", "manual_review": "需要人工核查"}


def _validate_case(case, seen):
    if not case["id"] or case["id"] in seen:
        raise ValueError(f"样本编号缺失或重复: {case['id']}")
    seen.add(case["id"])
    if not isinstance(case["on_site"], bool):
        raise ValueError(f"{case['id']}: on_site must be true or false")
    label = case["label"]
    if label["recommendation"] not in RECOMMENDATIONS or not label["rationale"].strip():
        raise ValueError(f"{case['id']}: invalid reference label")
    sources = [row["source"] for row in case["evidence"]]
    if len(sources) != len(set(sources)) or not set(sources) <= {"merchant", "platform"}:
        raise ValueError(f"{case['id']}: invalid or duplicate source")
    if not case["voucher"]["voucher_id"] or not isinstance(case["voucher"]["expires_at"], str):
        raise ValueError(f"{case['id']}: invalid voucher")
    return parse_aware_datetime(case["checked_at"])


def _evaluate_case(case, version, checked_at):
    # The evaluator sees only the business inputs. Reference labels and
    # reviewer notes are read by the scorer after both evaluations.
    voucher = dict(case["voucher"], policy_version=version)
    recommendation, reason = evaluate_redemption(voucher, case["evidence"], checked_at)
    status, _ = initial_dispute_result(recommendation, on_site=case["on_site"])
    return {"recommendation": recommendation, "reason": reason, "case_status": status}


def _time_difference(evidence):
    if len(evidence) != 2 or any(row["status"] != "redeemed" for row in evidence):
        return None
    try:
        platform_time, merchant_time = (parse_aware_datetime(row["redeemed_at"]) for row in evidence)
    except ValueError:
        return None
    return abs((platform_time - merchant_time).total_seconds())


def _summarize(rows, side):
    matches = false_support = missed_support = manual = 0
    for row in rows:
        result = row[side]
        expected = row["reference"]["recommendation"]
        actual = result["recommendation"]
        matches += actual == expected
        false_support += actual == "redemption_supported" and expected != "redemption_supported"
        missed_support += expected == "redemption_supported" and actual != "redemption_supported"
        manual += result["case_status"] == "pending_review"
    return {"reference_matches": matches, "unsupported_redemption_recommendations": false_support,
            "missed_redemption_recommendations": missed_support, "initial_manual_cases": manual}


def replay(dataset, baseline, candidate):
    if baseline == candidate or any(version not in POLICIES for version in (baseline, candidate)):
        raise ValueError("请选择两个不同的已知规则版本")
    if not dataset["cases"]:
        raise ValueError("回放数据集为空")
    seen, rows = set(), []
    for case in dataset["cases"]:
        checked_at = _validate_case(case, seen)
        first = _evaluate_case(case, baseline, checked_at)
        second = _evaluate_case(case, candidate, checked_at)
        label = case["label"]
        changed = first["recommendation"] != second["recommendation"]
        flags = []
        if changed:
            flags.append("recommendation_changed")
        if first["reason"] != second["reason"]:
            flags.append("reason_changed")
        if second["recommendation"] != label["recommendation"]:
            flags.append("candidate_disagrees_with_reference")
        rows.append({"id": case["id"], "scenario": case["scenario"], "voucher_id": case["voucher"]["voucher_id"],
                     "checked_at": case["checked_at"], "on_site": case["on_site"],
                     "redemption_delta_seconds": _time_difference(case["evidence"]),
                     "reference": label, "baseline": first, "candidate": second, "review_flags": flags})
    canonical = json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    policy_source = (ROOT / "voucher_policy.py").read_text(encoding="utf-8")
    return {
        "dataset": dataset["dataset_id"],
        "label_source": dataset["label_source"],
        "dataset_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "policy_source_sha256": hashlib.sha256(policy_source.encode("utf-8")).hexdigest(),
        "variants": {side: {"version": version, "tolerance_seconds": int(POLICIES[version].total_seconds())}
                     for side, version in (("baseline", baseline), ("candidate", candidate))},
        "snapshot_count": len(rows),
        "summary": {
            "recommendation_changes": sum(r["baseline"]["recommendation"] != r["candidate"]["recommendation"] for r in rows),
            "initial_route_changes": sum(r["baseline"]["case_status"] != r["candidate"]["case_status"] for r in rows),
            "baseline": _summarize(rows, "baseline"),
            "candidate": _summarize(rows, "candidate"),
        },
        "review_ids": [r["id"] for r in rows if r["review_flags"]],
        "cases": rows,
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report):
    base = report["variants"]["baseline"]
    candidate = report["variants"]["candidate"]
    summary = report["summary"]
    count = report["snapshot_count"]
    lines = [
        "# 核销规则离线回放", "",
        f"数据集 {report['dataset']} 包含 {count} 份合成快照。两版使用相同的证据和固定判定时间。",
        "参考标签由模拟复核事实预先给出，指标仅描述本样本集。", "",
        "| 指标 | 基线 | 候选 |",
        "| --- | ---: | ---: |",
        f"| 规则 | {base['version']} | {candidate['version']} |",
        f"| 时间容差 | {base['tolerance_seconds']} 秒 | {candidate['tolerance_seconds']} 秒 |",
    ]
    for key, label in (
        ("reference_matches", "与参考标注一致"),
        ("unsupported_redemption_recommendations", "误提示核销证据吻合"),
        ("missed_redemption_recommendations", "漏提示核销证据吻合"),
        ("initial_manual_cases", "初始进入人工复核"),
    ):
        lines.append(f"| {label} | {summary['baseline'][key]}/{count} | {summary['candidate'][key]}/{count} |")
    lines.extend([
        "",
        f"核查建议变化 {summary['recommendation_changes']}/{count}。初始分流变化 {summary['initial_route_changes']}/{count}。",
        "抽查样本 " + "、".join(report["review_ids"]) + "。", "",
        "| 快照 | 场景 | 两端时差 | 基线建议与原因 | 候选建议与原因 | 参考标注 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ])
    for row in report["cases"]:
        delta = row["redemption_delta_seconds"]
        first, second, ref = row["baseline"], row["candidate"], row["reference"]
        cells = [row["id"], row["scenario"], f"{delta:g} 秒" if delta is not None else "—",
                 f'{DECISIONS[first["recommendation"]]} · {first["reason"]}',
                 f'{DECISIONS[second["recommendation"]]} · {second["reason"]}',
                 f'{DECISIONS[ref["recommendation"]]} · {ref["rationale"]}']
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--baseline", choices=sorted(POLICIES), default="voucher-v1")
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="voucher-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work" / "voucher-policy-replay")
    args = parser.parse_args()
    try:
        report = replay(json.loads(args.fixtures.read_text(encoding="utf-8")), args.baseline, args.candidate)
    except (OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "replay.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    markdown = args.output_dir / "replay.md"
    markdown.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(f"REPORT {markdown.resolve()}")


if __name__ == "__main__":
    main()
