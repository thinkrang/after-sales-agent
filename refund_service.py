"""Server-side refund workflow.

The LLM may propose ``create_refund_request(order_id, reason)``, but it cannot
create authority by writing those values into a tool call. This module binds an
approval to the authenticated actor, tenant, thread and canonical action
payload, then executes it through an idempotent downstream gateway.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import orders
import tickets
from auth import User


class ApprovalRequired(PermissionError):
    pass


class WorkflowConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovalDecision:
    ticket: dict
    status: str


def _normalize_reason(reason: str) -> str:
    return " ".join((reason or "").strip().split())


def action_payload_hash(
    *,
    user: User,
    thread_id: str,
    order_id: str,
    amount_cents: int,
    reason: str,
) -> str:
    payload = {
        "tenant_id": user.tenant_id,
        "requester_id": user.username,
        "thread_id": thread_id,
        "order_id": order_id.strip(),
        "amount_cents": int(amount_cents),
        "reason": _normalize_reason(reason),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _payload_for(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> tuple[dict, dict, str]:
    order = orders.get_order(user, order_id.strip())
    decision = orders.refund_policy(order)
    payload_hash = action_payload_hash(
        user=user,
        thread_id=thread_id,
        order_id=order_id,
        amount_cents=int(order["amount_cents"]),
        reason=reason,
    )
    return order, decision, payload_hash


def prepare_ticket(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> tuple[dict, dict]:
    """Create the pending approval record before returning an interrupt."""
    order, decision, payload_hash = _payload_for(
        user,
        thread_id=thread_id,
        order_id=order_id,
        reason=reason,
    )
    if not decision["eligible"]:
        raise orders.RefundDenied(decision["reason"])

    ticket = tickets.create_ticket(
        thread_id=thread_id,
        order_id=order["order_id"],
        reason=_normalize_reason(reason),
        operator=user.username,
        tenant_id=user.tenant_id,
        requester_id=user.username,
        payload_hash=payload_hash,
        amount_cents=int(order["amount_cents"]),
    )
    return ticket, order


def find_pending_for_action(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> tuple[dict, dict] | None:
    try:
        order, _, payload_hash = _payload_for(
            user,
            thread_id=thread_id,
            order_id=order_id,
            reason=reason,
        )
    except orders.OrderNotFound:
        return None
    ticket = tickets.find_pending_for_action(
        thread_id,
        tenant_id=user.tenant_id,
        requester_id=user.username,
        order_id=order["order_id"],
        payload_hash=payload_hash,
    )
    return (ticket, order) if ticket else None


def find_action_ticket(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> tuple[dict, dict] | None:
    try:
        order, _, payload_hash = _payload_for(
            user,
            thread_id=thread_id,
            order_id=order_id,
            reason=reason,
        )
    except orders.OrderNotFound:
        return None
    ticket = tickets.find_action_ticket(
        thread_id,
        tenant_id=user.tenant_id,
        requester_id=user.username,
        order_id=order["order_id"],
        payload_hash=payload_hash,
    )
    return (ticket, order) if ticket else None


def decide(
    user: User,
    *,
    approval_id: str,
    thread_id: str,
    decision: str,
) -> ApprovalDecision:
    ticket = tickets.get_ticket(
        approval_id,
        tenant_id=user.tenant_id,
        requester_id=user.username,
    )
    if ticket is None or ticket["thread_id"] != thread_id:
        raise orders.OrderNotFound("审批记录不存在，或不属于当前账号与会话")
    if ticket["status"] != tickets.PENDING:
        raise WorkflowConflict("审批记录已经处理，不能重复提交")

    if decision == "approve":
        ok = tickets.approve_ticket(
            approval_id,
            operator=user.username,
            tenant_id=user.tenant_id,
            requester_id=user.username,
        )
        target_status = tickets.APPROVED
    elif decision == "reject":
        ok = tickets.reject_ticket(
            approval_id,
            reason="用户拒绝本次退款",
            operator=user.username,
            tenant_id=user.tenant_id,
            requester_id=user.username,
        )
        target_status = tickets.REJECTED
    else:
        raise ValueError("decision 必须是 approve 或 reject")

    if not ok:
        raise WorkflowConflict("审批状态已变化，请刷新后重试")
    refreshed = tickets.get_ticket(
        approval_id,
        tenant_id=user.tenant_id,
        requester_id=user.username,
    )
    if refreshed is None:
        raise WorkflowConflict("审批记录在状态迁移后消失")
    return ApprovalDecision(ticket=refreshed, status=target_status)


def execute(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> dict:
    """Execute only an approved, exact action.

    ``claim`` moves approved -> executing before the external call. If the
    process crashes after that transition, a retry finds ``executing`` and uses
    the same ticket ID as the downstream idempotency key. If it crashes after
    the downstream call but before marking executed, the same key returns the
    original refund id and repair can finish safely.
    """
    order, decision, payload_hash = _payload_for(
        user,
        thread_id=thread_id,
        order_id=order_id,
        reason=reason,
    )
    if not decision["eligible"] and decision["code"] != "already_refunded":
        raise orders.RefundDenied(decision["reason"])

    ticket = tickets.acquire_execution(
        thread_id,
        tenant_id=user.tenant_id,
        requester_id=user.username,
        order_id=order["order_id"],
        payload_hash=payload_hash,
    )
    if ticket is None:
        existing = tickets.find_executed(
            order["order_id"],
            tenant_id=user.tenant_id,
            requester_id=user.username,
        )
        if existing:
            return existing
        raise ApprovalRequired("没有找到与该订单、金额和原因绑定的已批准工单")

    if ticket["status"] == tickets.EXECUTED:
        return ticket
    if ticket["status"] not in (tickets.EXECUTING,):
        raise ApprovalRequired("退款工单尚未批准")

    try:
        execution = orders.execute_refund(
            user=user,
            order_id=order["order_id"],
            amount_cents=int(ticket["amount_cents"]),
            idempotency_key=ticket["id"],
        )
    except Exception as exc:
        tickets.mark_failed(ticket["id"], str(exc))
        raise

    # Never mark the ticket failed after the downstream call may have
    # succeeded. Keep it in ``executing`` so reconciliation can retry with the
    # same idempotency key.
    completed = tickets.mark_executed(
        ticket["id"],
        execution["refund_id"],
        execution_id=execution["idempotency_key"],
    )
    current = tickets.get_ticket(ticket["id"])
    if current and current["status"] == tickets.EXECUTED:
        return current
    if not completed:
        raise WorkflowConflict("退款已执行，但工单状态更新失败，可安全重试对账")
    if current is None:
        raise WorkflowConflict("退款已执行，但工单记录不存在")
    return current


def reconcile_executing(
    user: User,
    *,
    thread_id: str,
    order_id: str,
    reason: str,
) -> dict:
    """Retry a previously claimed execution using the same idempotency key."""
    return execute(
        user,
        thread_id=thread_id,
        order_id=order_id,
        reason=reason,
    )


def reconcile_ticket(ticket: dict) -> dict:
    """Recover an approved/executing ticket without relying on the chat stream.

    This is what a periodic reconciliation worker would call after a process
    crash. Authorization was already recorded on the ticket; execution still
    goes through the exact-payload claim and downstream idempotency checks.
    """
    if ticket["status"] not in (tickets.APPROVED, tickets.EXECUTING):
        raise WorkflowConflict(
            f"工单状态 {ticket['status']} 不属于可对账状态"
        )
    user = User(
        username=ticket["requester_id"],
        display_name=ticket["requester_id"],
        tenant_id=ticket["tenant_id"],
        roles=frozenset({"customer"}),
    )
    return execute(
        user,
        thread_id=ticket["thread_id"],
        order_id=ticket["order_id"],
        reason=ticket["reason"] or "",
    )
