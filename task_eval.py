"""Refund task evaluation with database postconditions (no LLM by default).

python task_eval.py --repeat 3 --output work/task-system.json
python task_eval.py --mode model --repeat 3 --output work/task-model.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import db
import orders
import refund_service
import refund_tasks
import tickets
from auth import User

USER = User("acme-manager", "客户", "acme", frozenset({"customer", "approver"}))


def prepare(task: dict) -> dict:
    return refund_service.prepare_ticket(
        USER, thread_id=task["thread_id"], order_id=task["order_id"],
        reason=task["reason"], task_revision=task["revision"],
    )[0]


def change(thread="task-eval", **kwargs) -> dict:
    return refund_tasks.update(USER, thread, expected_revision=refund_tasks.get(USER, thread)["revision"], **kwargs)


def approved(task: dict) -> dict:
    ticket = prepare(task)
    refund_service.decide(USER, approval_id=ticket["id"], thread_id=task["thread_id"], decision="approve")
    return ticket


def execution_count() -> int:
    with db.session(orders.ORDERS_DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]


def expect_conflict(fn) -> None:
    try:
        fn()
    except (refund_tasks.TaskConflict, refund_service.ApprovalRequired, refund_service.WorkflowConflict):
        return
    raise AssertionError("旧动作被错误放行")


def consultation():
    task = change(order_id="A100", reason="质量问题", intent="consult")
    expect_conflict(lambda: prepare(task))
    assert execution_count() == 0


def slot_fill():
    task = change(intent="apply")
    assert task["missing_fields"] == ["order_id", "reason"]
    expect_conflict(lambda: prepare(task))
    task = change(order_id="A100")
    assert task["missing_fields"] == ["reason"]
    task = change(reason="质量问题")
    ticket = approved(task)
    final = refund_service.execute_for_ticket(ticket)
    assert final["order_id"] == "A100" and final["status"] == "executed"
    assert refund_tasks.get(USER, task["thread_id"])["refund_id"] == final["refund_id"]
    assert execution_count() == 1


def change_order():
    old = change(order_id="A100", reason="质量问题", intent="apply")
    ticket = prepare(old)
    task = change(order_id="A200")
    assert task["reason"] == "" and task["status"] == "collecting"
    assert tickets.get_ticket(ticket["id"])["status"] == "rejected"
    expect_conflict(lambda: prepare(old))
    task = change(reason="尺码不合适")
    final = refund_service.execute_for_ticket(approved(task))
    assert final["order_id"] == "A200" and final["reason"] == "尺码不合适"
    assert orders.get_order(USER, "A100")["status"] == "delivered"
    assert execution_count() == 1


def change_reason():
    old = change(order_id="A100", reason="质量问题", intent="apply")
    ticket = prepare(old)
    task = change(reason="尺码不合适")
    assert tickets.get_ticket(ticket["id"])["status"] == "rejected"
    expect_conflict(lambda: prepare(old))
    assert prepare(task)["reason"] == "尺码不合适"
    assert execution_count() == 0


def cancel_pending():
    task = change(order_id="A100", reason="质量问题", intent="apply")
    ticket = prepare(task)
    task = change(cancel=True)
    assert task["status"] == "cancelled"
    assert not tickets.approve_ticket(ticket["id"])
    expect_conflict(lambda: prepare(task))
    assert execution_count() == 0


def approved_cannot_edit():
    ticket = approved(change(order_id="A100", reason="质量问题", intent="apply"))
    expect_conflict(lambda: change(cancel=True))
    expect_conflict(lambda: change(order_id="A200"))
    assert tickets.get_ticket(ticket["id"])["status"] == "approved"


def retry_after_lost_response():
    ticket = approved(change(order_id="A100", reason="质量问题", intent="apply"))
    original = orders.execute_refund

    def lost_response(**kwargs):
        original(**kwargs)
        raise TimeoutError("下游成功但响应丢失")

    with patch.object(orders, "execute_refund", lost_response):
        try:
            refund_service.execute_for_ticket(ticket)
        except TimeoutError:
            pass
    assert tickets.get_ticket(ticket["id"])["status"] == "executing"
    final = refund_service.reconcile_ticket(tickets.get_ticket(ticket["id"]))
    again = refund_service.execute_for_ticket(ticket)
    assert final["status"] == "executed" and final["refund_id"] == again["refund_id"]
    assert execution_count() == 1


def restart_read():
    task = change(order_id="A100", reason="质量问题", intent="apply")
    ticket = prepare(task)
    db.release_all()
    tickets.init_tickets_table()  # Reopen/migrate, without deleting state.
    recovered = refund_tasks.get(USER, task["thread_id"])
    assert recovered["revision"] == task["revision"]
    assert recovered["ticket_id"] == ticket["id"] and recovered["status"] == "pending_approval"


SYSTEM_CASES = [consultation, slot_fill, change_order, change_reason,
                cancel_pending, approved_cannot_edit, retry_after_lost_response, restart_read]


def run_system(repeat=1) -> list[dict]:
    results = []
    for attempt in range(repeat):
        for case in SYSTEM_CASES:
            with tempfile.TemporaryDirectory(prefix="refund-task-eval-") as temp:
                with patch.object(tickets, "DB_PATH", Path(temp) / "tickets.sqlite"), \
                     patch.object(orders, "ORDERS_DB", Path(temp) / "orders.sqlite"):
                    start = time.perf_counter()
                    error = None
                    try:
                        tickets.init_tickets_table()
                        orders.init_order_db()
                        case()
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    finally:
                        db.release_all()
                    results.append(dict(case=case.__name__, attempt=attempt + 1,
                                        passed=error is None, error=error,
                                        elapsed_ms=round((time.perf_counter() - start) * 1000, 2)))
    return results


def summary(rows, mode):
    from task_scenarios import VERSION, fingerprint
    durations = sorted(r["elapsed_ms"] for r in rows)
    groups = {}
    for row in rows:
        key = row["case"]
        group = groups.setdefault(key, {"runs": 0, "passed": 0, "failure_stages": Counter()})
        group["runs"] += 1
        group["passed"] += bool(row["passed"])
        if not row["passed"]:
            group["failure_stages"][row.get("failure_stage") or "runtime"] += 1
    split_results = {}
    for split in ("dev", "holdout"):
        sub = [r for r in rows if r.get("split") == split]
        if sub:
            split_results[split] = dict(total=len(sub), passed=sum(r["passed"] for r in sub))
    return dict(
        mode=mode, dataset_version=VERSION if mode == "model" else "refund-tasks-v1",
        dataset_sha256=fingerprint() if mode == "model" else None, total=len(rows),
        passed=sum(r["passed"] for r in rows),
        success_rate=sum(r["passed"] for r in rows) / len(rows) if rows else 0,
        p50_ms=statistics.median(durations) if durations else None,
        p95_ms=durations[max(0, (95 * len(durations) + 99) // 100 - 1)] if durations else None,
        by_case=groups, by_split=split_results,
        failure_stages=dict(Counter(r.get("failure_stage") or "runtime" for r in rows if not r["passed"])),
        limitations=("合成开发场景，指定 holdout 非独立采集；重复运行不代表生产分布。"
                     "system 模式验证业务约束，不代表模型能力。模型耗时不含服务启动预热。"
                     "messages 对照去掉结构化任务工具及 Skill 提示，保留对话记忆和相同审批/执行服务，"
                     "是组件组合对照，不能归因于某一条提示或某一个字段。"),
        results=rows,
    )


def compare(report, baseline):
    """Paired comparison only: no silent dropping unmatched/failed runs."""
    if report["mode"] != baseline["mode"] or report.get("dataset_sha256") != baseline.get("dataset_sha256"):
        raise ValueError("对照报告的模式或数据集指纹不一致")
    if report.get("model") != baseline.get("model"):
        raise ValueError("对照模型标识不一致")
    if report.get("variant") == baseline.get("variant"):
        raise ValueError("必须选择不同的对照变体")
    current = {(r["case"], r["attempt"]): r for r in report["results"]}
    previous = {(r["case"], r["attempt"]): r for r in baseline["results"]}
    if current.keys() != previous.keys() or len(current) != len(report["results"]) or len(previous) != len(baseline["results"]):
        raise ValueError("场景或重复次数不一致，不能选择性比较")
    gained = [k for k in current if current[k]["passed"] and not previous[k]["passed"]]
    lost = [k for k in current if previous[k]["passed"] and not current[k]["passed"]]
    return dict(baseline_variant=baseline.get("variant"), paired_runs=len(current),
                gained=gained, lost=lost,
                success_delta_points=100*(sum(r["passed"] for r in current.values())-sum(r["passed"] for r in previous.values()))/len(current) if current else None,
                mean_latency_delta_ms=statistics.mean(current[k]["elapsed_ms"]-previous[k]["elapsed_ms"] for k in current) if current else None)


def provenance():
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = list(root.glob("*.py")) + list((root/"skills").rglob("*.py")) + list((root/"skills").rglob("*.md")) + [root/"requirements.txt"]
    for path in sorted(sources):
        digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                source_sha256=digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("system", "model"), default="system")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--variant", choices=("structured", "messages"), default="structured")
    parser.add_argument("--baseline", type=Path, help="同一数据集/模型/重复次数的对照报告")
    parser.add_argument("--max-model-calls", type=int, default=40)
    parser.add_argument("--case", action="append", dest="case_ids", help="只跑指定场景，可重复传入")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须大于 0")
    if args.max_model_calls < 1:
        parser.error("--max-model-calls 必须大于 0")
    if args.mode == "model":
        from task_eval_model import run_model
        run_provenance = provenance()
        def progress(rows):
            if args.output:
                partial = summary(rows, "model")
                partial.update(complete=False, variant=args.variant, provenance=run_provenance,
                               model=rows[0].get("model"))
                path = args.output.with_suffix(".progress.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        rows = run_model(args.repeat, split=args.split, variant=args.variant,
                         max_model_calls=args.max_model_calls, case_ids=args.case_ids, progress=progress)
    else:
        rows = run_system(args.repeat)
    report = summary(rows, args.mode)
    report.update(complete=True, variant=args.variant, provenance=run_provenance if args.mode == "model" else provenance(),
                  model=rows[0].get("model") if rows else None)
    if args.baseline:
        report["comparison"] = compare(report, json.loads(args.baseline.read_text(encoding="utf-8")))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if rows and all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
