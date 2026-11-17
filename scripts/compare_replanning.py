"""Compare full branch recomputation and local replanning using isolated SQLite."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_case(root, change, variant):
    import db
    import orders
    import orchestrator
    import plan_store
    import refund_service
    import tickets
    from auth import User
    from planning import refund_plan

    root.mkdir()
    tickets.DB_PATH = root / "tickets.sqlite"
    orders.ORDERS_DB = root / "orders.sqlite"
    tickets.init_tickets_table()
    orders.init_order_db()
    user = User("acme-user", "客户", "acme", frozenset({"customer"}))
    sid = "replan-eval"
    draft = refund_plan("两笔退款", [
        {"key": "low", "order_id": "A300", "reason": "尺码不合适", "intent": "apply"},
        {"key": "high", "order_id": "A400", "reason": "质量问题", "intent": "apply"},
    ])
    plan_store.create(user, sid, draft)
    initial = orchestrator.run(user, sid)
    low, high = initial["tasks"]
    low_id = low["state"]["ticket_id"]
    old_high_id = high["state"]["ticket_id"]
    refund_service.decide(user, approval_id=low_id, decision="approve")
    baseline = orchestrator.run(user, sid)
    old_refund_id = baseline["tasks"][0]["state"]["refund_id"]
    assert old_refund_id

    changes = {
        "reason": {"reason": "包装破损"},
        "order": {"order_id": "A600", "reason": "商品破损"},
        "withdraw": {"cancel": True},
    }
    plan_store.edit(user, sid, high["id"], expected_revision=high["state"]["revision"], **changes[change])
    if variant == "full_branch" and change != "withdraw":
        # Disable reuse of successful reads in the affected branch only.
        # Both variants retain ownership, approval, and invalidation checks.
        with db.session(tickets.DB_PATH) as conn:
            conn.execute("""UPDATE agent_plan_steps SET status='pending', result=NULL,
                attempts=0, version=version+1, duration_ms=NULL, started_at=NULL
                WHERE task_id=? AND action IN ('lookup_order','check_policy') AND status='succeeded'""",
                (high["id"],))
    final = orchestrator.run(user, sid)
    final_low, final_high = (task["state"] for task in final["tasks"])
    assert final_low["ticket_id"] == low_id and final_low["refund_id"] == old_refund_id
    assert tickets.get_ticket(old_high_id)["status"] == tickets.REJECTED
    if change == "withdraw":
        assert final_high["status"] == "cancelled" and final_high["ticket_id"] is None
    else:
        assert final_high["status"] == tickets.PENDING
        assert final_high["order_id"] == ("A600" if change == "order" else "A400")
        assert final_high["reason"] == changes[change]["reason"]
        assert final_high["ticket_id"] != old_high_id
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
    reruns = [event["detail"]["tool"] for event in plan_store.history(user, sid)
              if event["kind"] == "step_started" and event["plan_revision"] == final["revision"]]
    count = final["calls_used"] - baseline["calls_used"]
    assert len(reruns) == count
    return {"change": change, "variant": variant, "business_step_attempts": count,
            "executed_actions": reruns, "completed_sibling_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be between 1 and 10")
    with tempfile.TemporaryDirectory(prefix="replan-ablation-") as directory:
        os.environ["AGENT_DATA_DIR"] = directory
        import db
        from task_eval import provenance

        rows = []
        try:
            for repeat in range(args.repeat):
                for change in ("reason", "order", "withdraw"):
                    for variant in ("full_branch", "local"):
                        root = Path(directory) / f"{repeat}-{change}-{variant}"
                        rows.append({"repeat": repeat + 1, **run_case(root, change, variant)})
                        db.release_all()
        finally:
            db.release_all()
        report = {"provenance": provenance(), "repeats_per_variant": args.repeat,
                  "unit": "business step attempts after edit", "rows": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"REPORT {args.output}")


if __name__ == "__main__":
    main()
