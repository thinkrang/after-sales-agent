"""A durable approval/execution graph independent of the conversation graph.

Checkpoint resumes are control flow, never authority. Only server-owned ticket
state can authorize the side effect. Replaying the graph reuses the ticket's
downstream idempotency key.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import refund_service
import tickets
from session_guard import conversation_lock


class RefundState(TypedDict, total=False):
    ticket_id: str
    status: str
    refund_id: str | None


def config(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": "refund_" + ticket_id}}


def _ticket(state):
    ticket = tickets.get_ticket(state["ticket_id"])
    if ticket is None:
        raise refund_service.WorkflowConflict("退款工单不存在")
    return ticket


def build(checkpointer):
    def review(state: RefundState):
        ticket = _ticket(state)
        if ticket["status"] == tickets.PENDING:
            interrupt({"approval_id": ticket["id"], "order_id": ticket["order_id"],
                       "amount_cents": ticket["amount_cents"], "task_revision": ticket.get("task_revision")})
            ticket = _ticket(state)
            if ticket["status"] == tickets.PENDING:
                raise refund_service.ApprovalRequired("流程恢复不能替代服务端审批")
        return {"status": ticket["status"], "refund_id": ticket.get("refund_id")}

    def execute(state: RefundState):
        ticket = refund_service.execute_for_ticket(_ticket(state))
        return {"status": ticket["status"], "refund_id": ticket.get("refund_id")}

    graph = StateGraph(RefundState)
    graph.add_node("review", review)
    graph.add_node("execute", execute)
    graph.add_edge(START, "review")
    graph.add_conditional_edges("review", lambda s: "execute" if s["status"] in
                                (tickets.APPROVED, tickets.EXECUTING) else END)
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=checkpointer)


def settle(graph, ticket: dict) -> dict:
    """Start/resume from database truth, including crash-before-checkpoint."""
    with conversation_lock("execution:" + ticket["id"]):
        return _settle(graph, ticket)


def _settle(graph, ticket: dict) -> dict:
    cfg = config(ticket["id"])
    snap = graph.get_state(cfg)
    if not snap.values:
        graph.invoke({"ticket_id": ticket["id"]}, config=cfg)
    elif ticket["status"] != tickets.PENDING:
        if snap.next:
            paused = any(task.interrupts for task in snap.tasks)
            graph.invoke(Command(resume=True) if paused else None, config=cfg)
        elif (snap.values or {}).get("status") != ticket["status"]:
            graph.invoke({"ticket_id": ticket["id"]}, config=cfg)
    return tickets.get_ticket(ticket["id"])
