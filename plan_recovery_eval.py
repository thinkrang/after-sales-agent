"""Synthetic paired cases and their independent business postconditions."""
import json
import re

RECOVERY_CASES = [
    {"id": "lost_receipt", "messages": ["请申请 A300 的退款，原因是尺码不合适。请告诉我这笔的处理状态。"]},
    {"id": "edit_while_waiting", "messages": [
        "帮我同时申请两笔退款：A300 尺码不合适，A400 有质量问题。",
        "A400 继续等主管审批。只把 A300 的退款原因改成包装破损，A300 仍然申请退款。",
    ]},
]
RECOVERY_IDS = frozenset(case["id"] for case in RECOVERY_CASES)
STATUS_LINE = re.compile(r"^\s*A300\s*[:：]\s*(待确认|待审批|已退款|未提交|状态未核实)\s*[。.]?\s*$", re.M)
RECOVERY_PROMPT = (
    "这次单笔退款也使用计划工具。收到 outcome_unknown 时，先用 get_refund_plan 核对工单，"
    "不要重提写操作。最终另起一行报告 A300 的状态，格式为 A300：状态，"
    "状态从待确认、待审批、已退款、未提交、状态未核实中按实际情况选一个。"
)


def tool_payload(message):
    if not isinstance(message.content, str):
        return {}
    try:
        data = json.loads(message.content)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def task_states(plan):
    return {task["state"]["order_id"]: {key: task["state"][key] for key in
            ("ticket_id", "status", "revision", "reason", "required_role")}
            for task in plan.get("tasks", [])}


def capture_turn(turn):
    from langchain_core.messages import ToolMessage

    calls = {call["id"]: (batch, call["name"]) for batch, message in enumerate(turn)
             for call in getattr(message, "tool_calls", [])}
    observations = []
    for message in turn:
        if isinstance(message, ToolMessage) and message.tool_call_id in calls:
            batch, name = calls[message.tool_call_id]
            payload = tool_payload(message)
            observations.append({"batch": batch, "name": name, "call_id": message.tool_call_id,
                                 "code": payload.get("code"), "tasks": task_states(payload)})
    return {"observations": observations, "answer": turn[-1].text,
            "boundary_reason": turn[-1].response_metadata.get("boundary_reason")}


def receipt_fault(create_name, ticket_rows, injections):
    from langchain.agents.middleware import AgentMiddleware

    class DropReceipt(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            response = handler(request)
            if request.tool_call["name"] == create_name and not injections:
                rows = ticket_rows()
                if (len(rows) == 1 and rows[0]["order_id"] == "A300"
                        and rows[0]["status"] == "pending_approval"
                        and task_states(tool_payload(response)).get("A300", {}).get("ticket_id") == rows[0]["id"]):
                    injections.append({"call_id": request.tool_call["id"], "ticket_id": rows[0]["id"]})
                    raise ConnectionResetError("synthetic loss of plan tool response after ticket commit")
            return response

    return DropReceipt()


def score(case_id, trace, rows, injections):
    from task_eval_model import reason_matches

    if case_id == "lost_receipt":
        if len(injections) != 1:
            return {"fault_triggered": False}
        fault = injections[0]
        turn = trace[0]
        observations = turn["observations"]
        lost = next((item for item in observations if item["call_id"] == fault["call_id"]
                     and item["code"] == "outcome_unknown"), None)
        later = [item for item in observations if lost and item["batch"] > lost["batch"]]
        verified = any(item["name"] == "get_refund_plan" and not item["code"]
                       and item["tasks"].get("A300", {}).get("ticket_id") == fault["ticket_id"]
                       and item["tasks"]["A300"]["status"] == "pending_approval" for item in later)
        statuses = STATUS_LINE.findall(turn["answer"])
        return {
            "fault_triggered": True,
            "outcome_unknown_seen": lost is not None,
            "read_verified_after_fault": verified,
            "no_repeat_write_after_fault": all(item["name"] == "get_refund_plan" for item in later),
            "single_original_pending_ticket": len(rows) == 1 and rows[0]["id"] == fault["ticket_id"]
            and rows[0]["order_id"] == "A300" and rows[0]["status"] == "pending_approval"
            and reason_matches(rows[0]["reason"], "尺码不合适"),
            "reported_pending": len(statuses) == 1 and statuses[0] in {"待确认", "待审批"},
        }

    before, after = (turn["tasks"] for turn in trace)
    low, high = before.get("A300", {}), before.get("A400", {})
    revised = after.get("A300", {})
    old_ticket = next((row for row in rows if row["id"] == low.get("ticket_id")), {})
    return {
        "initial_two_pending": set(before) == {"A300", "A400"}
        and low.get("status") == high.get("status") == "pending_approval"
        and high.get("required_role") == "approver",
        "only_target_revised": set(after) == {"A300", "A400"} and after["A400"] == high
        and revised.get("revision") == low.get("revision", 0) + 1
        and revised.get("status") == "pending_approval"
        and revised.get("ticket_id") != low.get("ticket_id")
        and reason_matches(revised.get("reason", ""), "包装破损"),
        "old_approval_revoked": old_ticket.get("status") == "rejected" and len(rows) == 3,
        "edit_called": any(item["name"] == "edit_refund_plan_task" and not item["code"]
                           for item in trace[1]["observations"]),
    }
