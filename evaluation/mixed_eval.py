"""Compare a fixed checklist with observation-driven routing on the same services."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


CASES = {"matching": "V300", "conflict_with_update": "V200"}
EVAL_SYSTEM_PROMPT = (
    "你是售后客服。退款只依据明确申请，先查订单资格和当前任务，登记后才能提交待确认。"
    "严格按工具实际结果回复。券的到店时间没明确给出时不能猜测。"
)


def run_case(user, case, variant):
    import db
    import mixed_voucher
    import orders
    import refund_service
    import refund_tasks
    import tickets
    import voucher_disputes as vouchers

    sid = "mixed-eval"
    voucher_id = CASES[case]
    store = "CENTRAL" if voucher_id == "V200" else "PARK"
    arrival = datetime.now(timezone.utc).replace(microsecond=0)
    metrics = dict(clarifications=0, voucher_observations=0, refund_refresh_reads=0)
    actions = []

    def observe(row=None, *, include_visit=False):
        metrics["voucher_observations"] += 1
        return mixed_voucher.observe(user, sid, voucher_id, store_code=store,
                                     visit_at=arrival.isoformat() if include_visit else None,
                                     expected_revision=row["revision"] if row else None)

    def select(row):
        choice = mixed_voucher.route(row["facts"])
        actions.append(choice)
        return mixed_voucher.choose(user, sid, row["revision"], choice)

    draft = refund_tasks.update(user, sid, expected_revision=0, order_id="A300", reason="尺码不合适", intent="apply")
    initial, _ = refund_service.prepare_ticket(user, thread_id=sid, order_id="A300", reason="尺码不合适",
                                              task_revision=draft["revision"], approval_protocol="detached")
    if variant == "fixed":
        metrics["clarifications"] += 1
        row = select(observe(include_visit=True))
    else:
        row = select(observe())
        if row["decision"] == "ask_visit":
            metrics["clarifications"] += 1
            row = select(observe(row, include_visit=True))

    stale_write_rejected = None
    if case == "conflict_with_update":
        assert row["decision"] == "guide_dispute"
        dispute = vouchers.submit(user, voucher_id, "到店用不了", store_code=store, visit_at=arrival)
        row = select(observe(row))
        assert row["decision"] == "wait_review"
        platform = next(item for item in dispute["assessment"]["evidence"] if item["source"] == "platform")
        ingestion = vouchers.ingest_merchant_evidence(user.tenant_id, "eval-merchant", dict(
            voucher_id=voucher_id, source_version=2, record_id="pos-record", status="redeemed",
            redeem_ref=platform["redeem_ref"], redeemed_at=platform["redeemed_at"]))
        assert ingestion["case_rechecked"]
        try:
            mixed_voucher.choose(user, sid, row["revision"], "wait_review")
        except vouchers.Conflict:
            stale_write_rejected = True
        assert stale_write_rejected
        if variant == "fixed":
            refund_tasks.get(user, sid)
            metrics["refund_refresh_reads"] += 1
        row = select(observe(row))
        assert row["facts"]["reason_code"] == "matching_redemption" and row["upstream"]["merchant"] == 2
    else:
        assert row["decision"] == "explain"

    final = tickets.get_ticket(initial["id"])
    applications = len(tickets.list_tickets(tenant_id=user.tenant_id, requester_id=user.username, limit=20))
    with db.session(orders.ORDERS_DB) as conn:
        executions = conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]
    assert initial == final and final["status"] == tickets.PENDING
    assert applications == 1 and executions == 0
    return dict(case=case, variant=variant, **metrics, voucher_actions=actions, refund_pending_initial=True,
                refund_preserved=True, refund_applications=applications, refund_executions=executions,
                stale_decision_rejected=stale_write_rejected)


def live_smoke(user):
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from agent_context import ContextBoundary, Limits
    import api
    import db
    import mixed_voucher
    import refund_tasks
    import orders
    import tickets
    import voucher_disputes
    import time
    import uuid

    model = ChatOpenAI(model="deepseek-chat", base_url="https://api.deepseek.com",
                       api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0, timeout=35, max_retries=0,
                       extra_body={"thinking": {"type": "disabled"}})
    graph = create_agent(model=model, checkpointer=InMemorySaver(), middleware=[ContextBoundary(Limits(model_calls=8))],
        tools=[api.get_voucher_branch, api.observe_voucher_branch, api.choose_voucher_step, api.get_refund_task,
               api.update_refund_task, api.get_order_info, api.submit_refund_request],
        system_prompt=EVAL_SYSTEM_PROMPT + mixed_voucher.INSTRUCTIONS)
    sid = "mixed-model"
    arrival = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    requests = ["V200 在中心店 CENTRAL 用不了。A300 尺码不合适，请申请退款。",
                f"V200 的实际到店时间是 {arrival}。"]
    rows = []
    seen = 0
    for message in requests:
        result = graph.invoke({"messages": [{"role": "user", "content": message}]}, {"configurable": {
            "thread_id": sid, "tenant": user.tenant_id, "user_id": user.username, "roles": sorted(user.roles),
            "turn_id": uuid.uuid4().hex, "turn_started": time.monotonic()}})
        new = result["messages"][seen:]
        seen = len(result["messages"])
        refund = refund_tasks.get(user, sid)
        rows.append({"user": message, "answer": new[-1].text, "voucher": mixed_voucher.get(user, sid),
                     "refund": {key: refund[key] for key in ("order_id", "ticket_id", "status", "revision", "required_role")},
                     "tools": [{"name": call["name"], "args": call["args"]} for item in new
                               if isinstance(item, AIMessage) for call in item.tool_calls]})
    first, second = rows
    with db.session(orders.ORDERS_DB) as conn:
        executions = conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]
    checks = {"voucher_step_requests_only_missing_visit": bool(first["voucher"] and first["voucher"]["decision"] == "ask_visit"
                       and first["voucher"]["facts"]["missing"] == ["visit_at"]),
              "refund_pending_first_turn": first["refund"]["order_id"] == "A300"
                       and first["refund"]["status"] == "pending_approval",
              "offered_dispute_after_visit": bool(second["voucher"] and second["voucher"]["decision"] == "guide_dispute"),
              "refund_preserved": first["refund"] == second["refund"],
              "voucher_not_looked_up_as_order": not any(call["name"] == "get_order_info" and
                  call["args"].get("order_id") == "V200" for row in rows for call in row["tools"]),
              "one_refund_application": len(tickets.list_tickets(tenant_id=user.tenant_id, limit=20)) == 1,
              "no_execution_or_dispute_without_confirmation": executions == 0 and not voucher_disputes.list_cases(user)}
    return {"mode": "real_model_smoke", "model": model.model_name, "checks": checks,
            "passed": all(checks.values()), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--output", type=Path, default=Path("work/mixed-agent/comparison.json"))
    parser.add_argument("--with-model", action="store_true", help="Run one paid mixed-chat smoke instead of the comparison")
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory(prefix="mixed-eval-") as directory:
        os.environ["AGENT_DATA_DIR"] = directory
        from auth import User
        import db
        import orders
        import tickets
        import voucher_disputes
        from evaluation.task_eval import provenance

        user = User("acme-user", "客户", "acme", frozenset({"customer"}))
        if args.with_model:
            tickets.init_tickets_table()
            orders.init_order_db()
            voucher_disputes.seed_demo()
            try:
                report = dict(provenance=provenance(), **live_smoke(user))
            finally:
                db.release_all()
        for repeat in range(1, args.repeat + 1) if not args.with_model else ():
            for case in CASES:
                for variant in ("fixed", "adaptive"):
                    root = Path(directory) / f"{repeat}-{case}-{variant}"
                    root.mkdir()
                    tickets.DB_PATH, orders.ORDERS_DB = root / "tickets.sqlite", root / "orders.sqlite"
                    try:
                        tickets.init_tickets_table()
                        orders.init_order_db()
                        voucher_disputes.seed_demo()
                        rows.append(dict(repeat=repeat, **run_case(user, case, variant)))
                    finally:
                        db.release_all()
        if not args.with_model:
            report = {"provenance": provenance(), "mode": "deterministic_service_comparison",
                  "scope": "Two synthetic scenarios, isolated SQLite, simulated payments",
                  "baseline": "Ask for visit information up front; refresh both branch statuses on a merchant update",
                  "candidate": "Ask from observed facts; refresh only the affected voucher branch",
                  "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report {args.output}")
    return int(not report.get("passed", True))


if __name__ == "__main__":
    raise SystemExit(main())
