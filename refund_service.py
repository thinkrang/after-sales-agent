"""Server-side refund workflow.

The LLM may propose ``create_refund_request(order_id, reason)``, but it cannot
create authority by writing those values into a tool call. This module binds an
approval to the authenticated actor, tenant, thread and canonical action
payload, then executes it through an idempotent downstream gateway.
"""

from __future__ import annotations

import hashlib
import json
import json
from dataclasses import dataclass

import orders
import risk
import tickets
from auth import User


class ApprovalRequired(PermissionError):
    pass


class ApprovalForbidden(PermissionError):
    """当前操作者无权批准这笔退款（风险等级或四眼原则）。"""


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

    # 风险分级在服务端算，不由模型判断——风控结论必须可复现、可审计。
    recent = tickets.count_recent_executed(
        tenant_id=user.tenant_id,
        requester_id=user.username,
        hours=risk.RECENT_WINDOW_HOURS,
    )
    assessment = risk.assess(order, decision, reason=reason, recent_executed=recent)

    ticket = tickets.create_ticket(
        thread_id=thread_id,
        order_id=order["order_id"],
        reason=_normalize_reason(reason),
        operator=user.username,
        tenant_id=user.tenant_id,
        requester_id=user.username,
        payload_hash=payload_hash,
        amount_cents=int(order["amount_cents"]),
        risk_tier=assessment.tier,
        risk_reasons=json.dumps(list(assessment.reasons), ensure_ascii=False),
        required_role=assessment.required_role,
        rule_version=assessment.rule_version,
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


def _risk_reasons(ticket: dict) -> list[str]:
    try:
        value = json.loads(ticket.get("risk_reasons") or "[]")
    except (TypeError, ValueError):
        return []
    return [str(x) for x in value] if isinstance(value, list) else []


def decide(
    user: User,
    *,
    approval_id: str,
    decision: str,
    thread_id: str | None = None,
    message: str | None = None,
) -> ApprovalDecision:
    """提交审批结果。

    两条路径，权限模型不同：

    - **本人路径**（``thread_id`` 必填）：客户在自己的会话里点确认。
      只能处理 ``required_role='customer'`` 的工单。
    - **主管路径**（``thread_id`` 为空）：``approver`` 角色从审批队列处理。
      只能处理 ``required_role='approver'`` 的工单，且**不能是发起人本人**。

    主管路径不校验 thread_id —— 主管不在客户那个会话里，用会话归属鉴权
    会把人挡在门外。这一路径的鉴权边界是「同租户 + 审批角色 + 非本人」。
    """
    ticket = tickets.get_ticket(
        approval_id, tenant_id=user.tenant_id, requester_id=user.username
    )
    is_requester = ticket is not None

    if ticket is None and user.has_role(risk.ROLE_APPROVER):
        # 主管路径：同租户内可见，不限定 requester
        ticket = tickets.get_ticket(approval_id, tenant_id=user.tenant_id)

    if ticket is None:
        raise orders.OrderNotFound("审批记录不存在，或不属于当前账号与会话")

    if is_requester and thread_id and ticket["thread_id"] != thread_id:
        raise orders.OrderNotFound("审批记录不存在，或不属于当前账号与会话")

    needs_manager = (ticket.get("required_role") or risk.ROLE_CUSTOMER) == risk.ROLE_APPROVER

    if needs_manager:
        if not user.has_role(risk.ROLE_APPROVER):
            detail = "；".join(_risk_reasons(ticket)) or "风险等级为高"
            raise ApprovalForbidden(f"该退款需要主管审批（{detail}）")
        if is_requester:
            raise ApprovalForbidden(
                "高风险退款不能由发起人自己批准，需另一位审批人处理（四眼原则）"
            )
    elif not is_requester:
        raise ApprovalForbidden("无权处理他人发起的低风险退款")

    if ticket["status"] != tickets.PENDING:
        raise WorkflowConflict("审批记录已经处理，不能重复提交")

    # 作用域必须和"谁有权动这张单"一致：本人路径限定 requester，
    # 主管路径只限定租户。
    scope: dict = {"tenant_id": user.tenant_id}
    if is_requester:
        scope["requester_id"] = user.username

    if decision == "approve":
        ok = tickets.approve_ticket(approval_id, operator=user.username, **scope)
        target_status = tickets.APPROVED
    elif decision == "reject":
        ok = tickets.reject_ticket(
            approval_id,
            reason=(message or "审批人拒绝本次退款"),
            operator=user.username,
            **scope,
        )
        target_status = tickets.REJECTED
    else:
        raise ValueError("decision 必须是 approve 或 reject")

    if not ok:
        raise WorkflowConflict("审批状态已变化，请刷新后重试")

    refreshed = tickets.get_ticket(approval_id, **scope)
    if refreshed is None:
        raise WorkflowConflict("审批记录在状态迁移后消失")
    return ApprovalDecision(ticket=refreshed, status=target_status)


def execute_for_ticket(ticket: dict) -> dict:
    """按工单执行退款，不依赖发起人的会话上下文。

    主管批准时用这条路径：主管不在客户的会话里，如果直接用主管身份调
    ``execute()``，payload 哈希里的 requester 会变成主管，对不上原工单。
    这里把发起人的身份还原出来，复用同一条幂等执行链路。
    """
    requester = User(
        username=ticket["requester_id"],
        display_name=ticket["requester_id"],
        tenant_id=ticket["tenant_id"],
        roles=frozenset({risk.ROLE_CUSTOMER}),
    )
    return execute(
        requester,
        thread_id=ticket["thread_id"],
        order_id=ticket["order_id"],
        reason=ticket.get("reason") or "",
    )


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
    except (orders.RefundDenied, orders.OrderNotFound) as exc:
        # 业务规则拒绝：金额不符、政策不允许、订单不存在。
        # 这类是确定性失败，重试也不会变，可以安全终结。
        tickets.mark_failed(ticket["id"], str(exc))
        raise
    except Exception as exc:
        # 其它异常（锁超时、连接中断、IO 错误）：可能是瞬时故障，
        # 也可能下游已受理但响应丢了。**不能直接标 failed**——
        # failed 是终态，对账不会再捞它，这笔退款就永远补不上了。
        # 保持 executing，累加尝试次数，交给对账带同一幂等键重试。
        tickets.record_attempt(ticket["id"], str(exc))
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
