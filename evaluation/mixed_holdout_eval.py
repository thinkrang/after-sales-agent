"""Frozen mixed-task dialogues, scored against fresh SQLite state and tool traces."""
import argparse
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from unittest.mock import patch
import uuid


DATASET = Path(__file__).resolve().parents[1] / "docs/evaluation/mixed-holdout-v1.json"
MODEL = "deepseek-chat"
REFUND_TOOLS = ("update_refund_task", "submit_refund_request", "request_order_refund", "withdraw_refund_request")
EVENTS = {"merchant_matches", "review_resolved", "review_eligible", "review_redeemed", "resolved_then_merchant"}
ACTION_CODES = {
    "order_customer_must_confirm": "refund_customer_confirm",
    "order_supervisor_must_approve": "refund_await_supervisor",
    "order_withdrawal_succeeded": "refund_withdrawn",
    "voucher_submit_required": "voucher_submit_dispute",
    "voucher_wait_for_review": "voucher_wait_review",
    "voucher_contact_support_to_apply": "voucher_contact_support",
}


def evaluation_prompt(intake):
    from evaluation.mixed_eval import EVAL_SYSTEM_PROMPT
    import mixed_voucher

    if not intake:
        return EVAL_SYSTEM_PROMPT + mixed_voucher.INSTRUCTIONS
    from customer_guidance import REPLY_INSTRUCTIONS
    from refund_service import SINGLE_REFUND_INSTRUCTIONS

    return ("你是中文售后客服。严格按业务工具的实际结果答复。券到店资料缺失时不能猜。"
            "写操作回执不确定时先读取当前状态，本轮不要重复写入。"
            + mixed_voucher.INSTRUCTIONS + SINGLE_REFUND_INSTRUCTIONS + REPLY_INSTRUCTIONS)


def grade_next_steps(assertions, steps):
    observed = {step["code"] for step in steps}
    return {claim: (code in observed) == assertions[claim]
            for claim, code in ACTION_CODES.items() if claim in assertions}


def grade_turn(spec, actual, previous):
    from evaluation.task_eval_model import reason_matches

    expected = sorted(spec["tickets"])
    actual_rows = sorted(actual["tickets"], key=lambda row: (row["order_id"], row["status"], row["reason"]))
    matched = (len(expected) == len(actual_rows) and
               all((order, status, role) == (row["order_id"], row["status"], row["required_role"])
                   and reason_matches(row["reason"], reason)
                   for (order, status, reason, role), row in zip(expected, actual_rows)))
    voucher = actual["voucher"] or {}
    facts = voucher.get("facts") or {}
    checks = {
        "refund_tickets": matched,
        "no_unconfirmed_money_or_dispute": actual["executions"] == 0
        and actual["disputes"] == spec.get("seeded_disputes", 0),
    }
    if "voucher" in spec:
        checks["voucher_route"] = (not voucher if spec["voucher"] is None else
            [voucher.get("voucher_id"), voucher.get("decision"), facts.get("reason_code")] == spec["voucher"]
            and not voucher.get("needs_recheck"))
    for field in ("visit_step", "case_outcome"):
        if field in spec:
            checks[field] = facts.get(field) == spec[field]
    if "merchant_version" in spec:
        checks["merchant_version"] = voucher.get("upstream", {}).get("merchant") == spec["merchant_version"]
    if spec.get("preserve_refund"):
        checks["same_refund_ticket_unchanged"] = actual["tickets"] == previous["tickets"]
        checks["merchant_refresh_avoids_refund_tools"] = not any(step["name"] in {
            "get_order_info", "get_refund_task", *REFUND_TOOLS,
        } for step in actual["tools"])
    return checks


def grade_fault(kind, rows, count):
    events = [event for row in rows for event in row["runtime"]]
    checks = {"fault_injected_once": count == 1}
    if kind == "order_read_timeout":
        first = next((event for event in events if event["event"] == "tool_attempt"
                      and event["tool"] == "get_order_info" and event["code"] == "transient"), {})
        checks["read_retried_after_timeout"] = bool(first) and any(
            event["event"] == "tool_attempt" and event["tool"] == "get_order_info"
            and event["ordinal"] == first["ordinal"] and event["attempt"] == 2 and event["code"] is None for event in events)
    if kind == "refund_receipt_lost":
        steps = [step for row in rows for step in row["tools"]]
        lost = next((step for step in steps if step["name"] in {"submit_refund_request", "request_order_refund"}
                     and step["code"] == "outcome_unknown"), None)
        later = [step for step in steps if lost and step["batch"] > lost["batch"]]
        checks["unknown_write_receipt_visible"] = lost is not None
        checks["read_back_after_lost_receipt"] = any(
            step["name"] == "get_refund_task" and not step["code"] and step["ticket_id"] for step in later)
        checks["no_new_write_after_uncertain_receipt"] = bool(lost) and all(
            step["name"] not in {*REFUND_TOOLS, "observe_voucher_branch", "choose_voucher_step"} for step in later)
    return checks


def run_event(event, user, seed_case):
    import voucher_disputes as vouchers
    from auth import User

    if event == "resolved_then_merchant":
        run_event("review_eligible", user, seed_case)
        run_event("merchant_matches", user, seed_case)
    elif event == "merchant_matches":
        current = vouchers.detail(user, seed_case["id"])
        platform, = [item for item in current["assessment"]["evidence"] if item["source"] == "platform"]
        result = vouchers.ingest_merchant_evidence(user.tenant_id, "held-out-feed", {
            "voucher_id": seed_case["voucher_id"], "source_version": 2, "record_id": "pos-holdout",
            "status": "redeemed", "redeem_ref": platform["redeem_ref"], "redeemed_at": platform["redeemed_at"],
        })
        if not result["case_rechecked"]:
            raise RuntimeError("merchant fixture did not revise the dispute")
    elif event in {"review_resolved", "review_eligible", "review_redeemed"}:
        manager = User("acme-manager", "主管", user.tenant_id, frozenset({"approver"}))
        current = vouchers.detail(manager, seed_case["id"])
        outcome = "redemption_confirmed" if event == "review_redeemed" else "eligible_to_apply"
        note = "合成场景复核结论，确认已核销" if event == "review_redeemed" else "合成场景复核结论，支持申请退款"
        vouchers.review(manager, seed_case["id"], current["revision"], outcome=outcome, note=note)
    else:
        raise ValueError(f"Unknown fixture event {event}")


def tool_log(messages):
    from langchain_core.messages import ToolMessage
    from agent_context import result_code
    from evaluation.plan_recovery_eval import tool_payload

    outcomes = {item.tool_call_id: item for item in messages if isinstance(item, ToolMessage)}
    rows = []
    for batch, item in enumerate(messages):
        for call in getattr(item, "tool_calls", []):
            result = outcomes.get(call["id"])
            payload = tool_payload(result) if result else {}
            observed = {key: payload[key] for key in ("voucher_id", "reason_code", "revision", "case_status",
                "case_outcome", "needs_recheck", "next_step", "required_fields", "customer_next_action",
                "redemption_finding", "order_id", "status", "required_role", "operation", "message") if key in payload}
            rows.append({"batch": batch, "name": call["name"], "args": call["args"],
                         "code": result_code(result), "ticket_id": payload.get("ticket_id"), "observed": observed})
    return rows


def run_case(case, user, model, *, arrival, intake=False):
    import api
    import db
    import agent_runtime
    import mixed_voucher
    import orders
    import refund_service
    import tickets
    import voucher_disputes as vouchers
    from agent_context import ContextBoundary, Limits, current_turn
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.callbacks import BaseCallbackHandler
    from langgraph.checkpoint.memory import InMemorySaver

    case_fault = case.get("fault")
    injected = 0
    active_tool = ContextVar("holdout_active_tool", default=None)
    original_lookup = orders.get_order
    write_name = "request_order_refund" if intake else "prepare_ticket"
    original_write = getattr(refund_service, write_name)

    def lookup(user, order_id):
        nonlocal injected
        if case_fault == "order_read_timeout" and not injected and order_id == "A300" and active_tool.get() == "get_order_info":
            injected += 1
            raise TimeoutError("synthetic first order lookup timeout")
        return original_lookup(user, order_id)

    def write(*args, **kwargs):
        nonlocal injected
        result = original_write(*args, **kwargs)
        ticket = result if intake else result[0]
        if case_fault == "refund_receipt_lost" and not injected and ticket["order_id"] == "A400":
            injected += 1
            raise ConnectionResetError("synthetic response lost after refund ticket commit")
        return result

    class FaultScope(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            token = active_tool.set(request.tool_call["name"])
            try:
                return handler(request)
            finally:
                active_tool.reset(token)

    class Trace(logging.Handler):
        def emit(self, record):
            if event := getattr(record, "agent_event", None):
                events.append(event)

    class Usage(BaseCallbackHandler):
        def on_chat_model_start(self, *args, **kwargs):
            usage["calls"] += 1

        def on_llm_end(self, response, **kwargs):
            data = response.generations[0][0].message.usage_metadata
            if data:
                usage["with_usage"] += 1
                usage["input_tokens"] += data["input_tokens"]
                usage["output_tokens"] += data["output_tokens"]

    graph = create_agent(model=model, checkpointer=InMemorySaver(),
        middleware=[FaultScope(), ContextBoundary(Limits(model_calls=8))],
        tools=[api.get_voucher_branch, api.observe_voucher_branch, api.choose_voucher_step, api.get_refund_task,
               api.update_refund_task, api.get_order_info,
               *([api.withdraw_refund_request, api.request_order_refund] if intake
                 else [api.submit_refund_request, api.withdraw_refund_request])],
        system_prompt=evaluation_prompt(intake))
    seed_case = None
    if voucher := case.get("seed_dispute"):
        seed_case = vouchers.submit(user, voucher, "离线合成争议", store_code="CENTRAL",
                                   visit_at=datetime.fromisoformat(arrival))
        if initial_event := case.get("initial_event"):
            run_event(initial_event, user, seed_case)
    rows, events, usage = [], [], Counter()
    usage_callback = Usage()
    sid = "mixed-holdout-" + case["id"]
    handler = Trace()
    logger = agent_runtime.LOGGER
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    error = None
    provider_status = None
    try:
        with patch.object(orders, "get_order", lookup), patch.object(refund_service, write_name, write):
            for spec in case["turns"]:
                if event := spec.get("event"):
                    run_event(event, user, seed_case)
                message = spec["text"].format(arrival=arrival)
                turn_id = uuid.uuid4().hex
                start = time.perf_counter()
                before = usage.copy()
                state = graph.invoke({"messages": [{"role": "user", "content": message}]}, {"configurable": {
                    "thread_id": sid, "tenant": user.tenant_id, "user_id": user.username, "roles": sorted(user.roles),
                    "turn_id": turn_id, "turn_started": time.monotonic(), "user_message": message},
                    "callbacks": [usage_callback]})
                elapsed = round((time.perf_counter() - start) * 1000, 2)
                messages = current_turn(state["messages"])
                with db.session(orders.ORDERS_DB) as conn:
                    executions = conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]
                # Inspect all tenants, so a cross-tenant write cannot disappear from the grade.
                with db.session(tickets.DB_PATH) as conn:
                    all_tickets = [dict(row) for row in conn.execute("""SELECT id, order_id, status, reason,
                        required_role, tenant_id, requester_id FROM refund_tickets ORDER BY id""")]
                    disputes = conn.execute("SELECT COUNT(*) FROM voucher_disputes").fetchone()[0]
                    pending = [dict(t) for t in conn.execute("""SELECT * FROM refund_tickets
                        WHERE thread_id=? AND tenant_id=? AND requester_id=? AND status='pending_approval'""",
                        (sid, user.tenant_id, user.username))]
                row = {"user": message, "event": event, "answer": messages[-1].text, "elapsed_ms": elapsed,
                       "boundary": messages[-1].response_metadata.get("boundary_reason"),
                       "usage": {field: usage[field] - before[field] for field in
                                 ("calls", "with_usage", "input_tokens", "output_tokens")},
                       "tools": tool_log(messages), "voucher": mixed_voucher.get(user, sid),
                       "tickets": all_tickets, "executions": executions, "disputes": disputes,
                       "runtime": [item for item in events if item["turn_id"] == turn_id]}
                if intake:
                    snapshot = api._conversation_payload(user, sid, [api._action_from_ticket(t, user) for t in pending])
                    row["next_steps"] = snapshot["next_steps"]
                    row["next_step_checks"] = grade_next_steps(spec.get("assertions", {}), row["next_steps"])
                if seed_case is not None:
                    current = vouchers.detail(user, seed_case["id"])
                    row["seeded_case_state"] = {key: current[key] for key in
                                                ("voucher_id", "status", "outcome", "revision")}
                    row["seeded_case_state"]["merchant_record_count"] = sum(
                        item["source"] == "merchant" for item in current["assessment"]["evidence"])
                grade = grade_turn(dict(spec, seeded_disputes=int(seed_case is not None)), row, rows[-1] if rows else None)
                grade["ticket_scope"] = all(t["tenant_id"] == user.tenant_id and t["requester_id"] == user.username
                                            for t in all_tickets)
                row["checks"] = grade
                rows.append(row)
    except Exception as exc:
        provider_status = getattr(exc, "status_code", None)
        error = type(exc).__name__
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    fault_checks = grade_fault(case_fault, rows, injected) if case_fault else {}
    with db.session(orders.ORDERS_DB) as conn:
        final_executions = conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]
    with db.session(tickets.DB_PATH) as conn:
        final_orders = [row[0] for row in conn.execute("SELECT order_id FROM refund_tickets")]
        final_disputes = conn.execute("SELECT COUNT(*) FROM voucher_disputes").fetchone()[0]
    passed = error is None and len(rows) == len(case["turns"]) and all(fault_checks.values()) and all(
        all(row["checks"].values()) for row in rows)
    return {"case": case["id"], "passed": passed, "error": error, "provider_status": provider_status,
            "usage": dict(usage), "fault_checks": fault_checks, "rows": rows,
            "final_writes": {"refund_orders": final_orders, "refund_executions": final_executions,
                             "voucher_disputes": final_disputes}}


def write_report(path, report):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):
        try:
            partial.replace(path)
            return
        except PermissionError:
            # Windows readers can briefly hold the destination without delete sharing.
            # Retry the file operation only, leaving the complete checkpoint on disk.
            if attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def load_dataset(path):
    source = path.read_bytes()
    data = json.loads(source)
    cases = data["cases"]
    if not cases:
        raise ValueError("Dataset must contain at least one case")
    identifiers = set()
    for case in cases:
        case_id = case["id"]
        if not re.fullmatch(r"[a-z0-9_]{1,64}", case_id) or case_id in identifiers or not case["turns"]:
            raise ValueError(f"Invalid, empty or duplicate case: {case_id}")
        identifiers.add(case_id)
        events = [case.get("initial_event"), *(turn.get("event") for turn in case["turns"])]
        if any(event is not None and (event not in EVENTS or not case.get("seed_dispute")) for event in events):
            raise ValueError(f"Unsupported or unseeded case event: {case_id}")
    return source, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true", required=True, help="Run the paid DeepSeek evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--intake", action="store_true", help="Use atomic refund intake and score server action guidance")
    args = parser.parse_args()
    spec, dataset = load_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open("x", encoding="utf-8").close()
    rows = []
    with tempfile.TemporaryDirectory(prefix="mixed-holdout-") as directory:
        os.environ.update(AGENT_DATA_DIR=directory, PLAN_WORKER_ENABLED="0",
                          LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
        import db
        import orders
        import tickets
        import voucher_disputes
        from auth import User
        from langchain_openai import ChatOpenAI
        from evaluation.task_eval import provenance

        model = ChatOpenAI(model=MODEL, base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"),
                           temperature=0, timeout=35, max_retries=0,
                           extra_body={"thinking": {"type": "disabled"}})
        user = User("acme-user", "客户", "acme", frozenset({"customer"}))
        arrival = datetime.now(timezone.utc).replace(hour=4, minute=30, second=0, microsecond=0).isoformat()
        report = dict(dataset=dataset["id"], protocol=dataset["protocol"],
                      dataset_sha256=hashlib.sha256(spec).hexdigest(), model=MODEL, arrival=arrival,
                      prompt_sha256=hashlib.sha256(evaluation_prompt(args.intake).encode()).hexdigest(),
                      provenance=provenance(), planned_cases=len(dataset["cases"]), rows=rows,
                      settings={"temperature": 0, "max_sdk_retries": 0, "model_calls_per_turn": 8,
                                "atomic_intake": args.intake})
        write_report(args.output, report)
        for case in dataset["cases"]:
            root = Path(directory) / case["id"]
            root.mkdir()
            tickets.DB_PATH, orders.ORDERS_DB = root / "tickets.sqlite", root / "orders.sqlite"
            try:
                tickets.init_tickets_table()
                orders.init_order_db()
                voucher_disputes.seed_demo()
                row = run_case(case, user, model, arrival=arrival, intake=args.intake)
            finally:
                db.release_all()
            rows.append(row)
            write_report(args.output, report)
            print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'} ({row['usage'].get('calls', 0)} calls)",
                  flush=True)
            if row["provider_status"] in {401, 402, 429, 500, 502, 503, 504}:
                break
    report.update(complete=len(rows) == len(dataset["cases"]) and all(row["provider_status"] is None for row in rows),
                  passed=sum(row["passed"] for row in rows))
    write_report(args.output, report)
    print(f"REPORT {args.output}")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
