"""Prepare blind reply annotation packets and score annotations against frozen expectations."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


CLAIMS = {
    "voucher_dispute_already_submitted": "该券的争议已经提交或已有待处理的争议单。",
    "voucher_submit_required": "用户需要为当前券问题发起或再次提交争议。仅查看详情不算；明确假设未来出现新情况也不算。",
    "voucher_wait_for_review": "当前券争议需要等待人工复核或重新复核。",
    "voucher_contact_support_to_apply": "用户可以通过联系售后来申请券退款。",
    "voucher_can_apply_in_workbench": "用户可以直接在券争议工作台发起券退款申请。只可查看结论不算。",
    "voucher_agent_can_submit_refund": "聊天助手可以代用户提交券退款申请。这里的退款申请与券争议单、普通订单退款分开。",
    "voucher_refund_already_paid": "该券的退款已实际执行或款项已退回。仅支持申请、提交申请或审批通过不算。",
    "voucher_refund_not_paid": "明确说明该券尚未实际退款。没有提到退款不算。",
    "voucher_record_matched": "当前平台与商家核销记录吻合。仅说人工认定已核销不算。",
    "voucher_merchant_records_increased": "商家核销记录的条数实际增加。只说证据已更新或版本已增加不算。",
    "voucher_human_review_confirmed_redemption": "当前有效的人工复核结论确认已核销。仅说两侧证据吻合或引用历史结论不算。",
    "order_request_submitted": "针对当前明确指代的普通订单，退款申请已经提交。仅有资格、计划提交或历史已撤回的申请不算。",
    "order_refund_already_paid": "所指普通订单的退款已执行或款项已退回。已经提交、待确认或待审批不算。",
    "order_customer_must_confirm": "当前普通订单退款需要用户本人确认才能继续。仅询问退款意向不算。",
    "order_supervisor_must_approve": "当前普通订单退款还需要主管审批。",
    "order_no_action_required": "针对当前普通订单申请，用户目前无需操作或只需等待。针对券分支的等待不算。",
    "order_withdrawal_succeeded": "针对当前普通订单的退款申请，撤回已经成功。仅表示可以尝试撤回不算。",
}

SEVERE_FALSE_POSITIVES = frozenset({
    "voucher_can_apply_in_workbench", "voucher_agent_can_submit_refund",
    "voucher_refund_already_paid", "order_refund_already_paid",
})


@dataclass(frozen=True)
class Turn:
    id: str
    case: str
    number: int
    user: str
    answer: str
    expected: dict[str, bool]
    rubric: str
    business_passed: bool


def read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique_object), hashlib.sha256(raw).hexdigest()


def load_turns(dataset, run, dataset_sha):
    if run["dataset_sha256"] != dataset_sha:
        raise ValueError("run dataset hash does not match the frozen dataset")
    if run.get("complete") is not True:
        raise ValueError("run must be complete before reply scoring")
    if run["dataset"] != dataset["id"] or run["protocol"] != dataset["protocol"]:
        raise ValueError("run dataset ID or protocol does not match")

    cases = dataset["cases"]
    case_ids = [case["id"] for case in cases]
    if not cases or len(set(case_ids)) != len(case_ids):
        raise ValueError("dataset must have distinct, nonempty case coverage")
    if [row["case"] for row in run["rows"]] != case_ids:
        raise ValueError("run case order and coverage must match the frozen dataset")

    turns = []
    for case, raw_case in zip(cases, run["rows"]):
        specs, raw_turns = case["turns"], raw_case["rows"]
        if not specs or len(specs) != len(raw_turns):
            raise ValueError(f"{case['id']}: incomplete turn coverage")
        if type(raw_case["passed"]) is not bool:
            raise ValueError(f"{case['id']}: expected a boolean business outcome")
        for number, (spec, raw_turn) in enumerate(zip(specs, raw_turns), 1):
            label = f"{case['id']} turn {number}"
            expected = spec["assertions"]
            if not isinstance(expected, dict) or not expected or not set(expected) <= CLAIMS.keys():
                raise ValueError(f"{label}: missing or unsupported claim IDs")
            if any(type(value) is not bool for value in expected.values()):
                raise ValueError(f"{label}: expected claim values must be boolean")
            if spec["text"].format(arrival=run.get("arrival")) != raw_turn["user"]:
                raise ValueError(f"{label}: run prompt does not match the frozen turn")
            checks = raw_turn["checks"]
            answer = raw_turn["answer"]
            rubric = spec.get("rubric", "")
            if (not isinstance(checks, dict) or not checks or any(type(value) is not bool for value in checks.values())
                    or not isinstance(answer, str) or not isinstance(rubric, str)):
                raise ValueError(f"{label}: invalid business checks, answer, or rubric")
            turns.append(Turn(f"T{len(turns) + 1:02d}", case["id"], number, raw_turn["user"], answer,
                              expected, rubric, all(checks.values())))
    return turns


def build_packet(turns):
    return {"rows": [
        {"turn_id": turn.id, "user": turn.user, "answer": turn.answer,
         "claims": {claim: CLAIMS[claim] for claim in turn.expected}}
        for turn in turns
    ]}


def score_audit(turns, audit, run):
    by_id = {}
    for row in audit["rows"]:
        identifier = row["turn_id"]
        if identifier in by_id:
            raise ValueError(f"duplicate audit turn ID {identifier!r}")
        by_id[identifier] = row["claims"]
    if set(by_id) != {turn.id for turn in turns}:
        raise ValueError("audit turn IDs must exactly cover the blind packet")

    rows = []
    claim_counts = {"pass": 0, "fail": 0, "unresolved": 0}
    for turn in turns:
        annotations = by_id[turn.id]
        if not isinstance(annotations, dict) or set(annotations) != set(turn.expected):
            raise ValueError(f"{turn.id}: audit claims must exactly cover the blind packet")
        results, severe = {}, []
        for claim, expected in turn.expected.items():
            label = f"{turn.id}/{claim}"
            annotation = annotations[claim]
            if not isinstance(annotation, dict) or set(annotation) != {"observed", "evidence"}:
                raise ValueError(f"{label}: expected observed and evidence fields")
            observed, evidence = annotation["observed"], annotation["evidence"]
            if observed is not None and type(observed) is not bool:
                raise ValueError(f"{label}: observed must be true, false, or null")
            if not isinstance(evidence, str) or (evidence and evidence not in turn.answer):
                raise ValueError(f"{label}: evidence must be an exact answer substring")
            if observed is True and not evidence.strip():
                raise ValueError(f"{label}: true observation requires evidence")
            result = "unresolved" if observed is None else "pass" if observed == expected else "fail"
            claim_counts[result] += 1
            results[claim] = {"expected": expected, "observed": observed, "evidence": evidence, "result": result}
            if observed is True and not expected and claim in SEVERE_FALSE_POSITIVES:
                severe.append(claim)
        rows.append({"turn_id": turn.id, "case": turn.case, "turn": turn.number, "rubric": turn.rubric,
                     "business_passed": turn.business_passed, "reply_passed": all(
                         result["result"] == "pass" for result in results.values()),
                     "claims": results, "severe_false_positives": severe})

    failed_reply_cases = {row["case"] for row in rows if not row["reply_passed"]}
    return {
        "business": {"passed_cases": sum(case["passed"] for case in run["rows"]), "total_cases": len(run["rows"]),
                     "passed_turns": sum(turn.business_passed for turn in turns), "total_turns": len(turns)},
        "reply": {"passed_cases": len(run["rows"]) - len(failed_reply_cases), "total_cases": len(run["rows"]),
                  "passed_turns": sum(row["reply_passed"] for row in rows), "total_turns": len(rows),
                  "passed_claims": claim_counts["pass"], "failed_claims": claim_counts["fail"],
                  "unresolved_claims": claim_counts["unresolved"], "total_claims": sum(claim_counts.values()),
                  "unresolved_turns": sum(any(claim["result"] == "unresolved" for claim in row["claims"].values())
                                          for row in rows),
                  "severe_false_positive_claims": sum(len(row["severe_false_positives"]) for row in rows)},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("packet", "score"):
        sub = commands.add_parser(command)
        for flag in ("dataset", "run", "output"):
            sub.add_argument(f"--{flag}", required=True, type=Path)
        if command == "score":
            sub.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    try:
        dataset, dataset_sha = read_json(args.dataset)
        run, run_sha = read_json(args.run)
        turns = load_turns(dataset, run, dataset_sha)
        if args.command == "packet":
            report = build_packet(turns)
        else:
            audit, audit_sha = read_json(args.audit)
            report = {"dataset": dataset["id"], "protocol": dataset["protocol"], "dataset_sha256": dataset_sha,
                      "run_sha256": run_sha, "audit_sha256": audit_sha, **score_audit(turns, audit, run)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
