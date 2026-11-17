"""真实接口冒烟：区分核查器可用、证据充分和证据不足；产生两次模型调用费用。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence
from task_eval import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [dict(source="probe-policy.txt", content="签收后7天内可申请无理由退货，商品须完好且不影响二次销售。")]
    rows = []
    for question, expected in [("签收后几天内可申请无理由退货？", True), ("支持比特币付款吗？", False)]:
        verdict = evidence.verify_evidence(question, results)
        rows.append(dict(question=question, expected=expected, verdict=verdict.model_dump(),
                         passed=verdict.sufficient is expected and "不可用" not in verdict.missing))
    report = dict(provenance=provenance(), method="function_calling", rows=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(not all(r["passed"] for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
