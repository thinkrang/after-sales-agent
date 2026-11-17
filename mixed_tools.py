"""Business tools shared by the main Agent and the offline mixed-task replay."""
from __future__ import annotations

import json
import logging
from typing import Literal

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from auth import User
import mixed_voucher
import orders
import plan_store
import refund_service
import refund_tasks
import tool_result
import voucher_disputes


def _user_from_config(config: RunnableConfig) -> User:
    configurable = config.get("configurable") or {}
    return User(
        username=str(configurable.get("user_id") or ""),
        display_name=str(configurable.get("display_name") or ""),
        tenant_id=str(configurable.get("tenant") or ""),
        roles=frozenset(configurable.get("roles") or []),
    )


@tool
def get_order_info(order_id: str, config: RunnableConfig) -> str:
    """查询当前登录用户自己的订单及退款资格，不执行任何写操作。

    返回 JSON 信封：ok=true 时含订单字段；ok=false 时看 code——
    not_found 表示确定没有，应向用户核对订单号而不是重试；
    transient 表示服务暂时不可用，可以重试一次。
    """
    try:
        user = _user_from_config(config)
        order = orders.get_order(user, order_id)
        return tool_result.order_snapshot(order, orders.refund_policy(order))
    except Exception as exc:  # noqa: BLE001
        # 工具异常不能炸掉整轮对话：翻译成失败码交给模型决定下一步。
        logging.warning("订单查询失败 order=%s", order_id, exc_info=True)
        return tool_result.order_lookup_failure(exc)


@tool
def get_voucher_branch(config: RunnableConfig) -> str:
    """读券分支和证据版本。not_started 时可从用户已给的券号开始观察；needs_recheck 时旧结论失效。"""
    cfg = config["configurable"]
    try:
        state = mixed_voucher.get(_user_from_config(config), str(cfg["thread_id"]))
        return json.dumps(mixed_voucher.agent_view(state) if state is not None else {
            "status": "not_started", "hint": "已有券号就调用 observe_voucher_branch，未知到店资料留空",
        }, ensure_ascii=False)
    except voucher_disputes.NotFound as exc:
        return tool_result.fail(tool_result.NOT_FOUND, str(exc))


@tool
def observe_voucher_branch(voucher_id: str, config: RunnableConfig, store_code: str | None = None,
                           visit_at: str | None = None, expected_revision: int | None = None) -> str:
    """观察本人券分支的当前证据或补充明确的到店资料。时间须带时区，缺失时留空。已有分支用其 revision。"""
    cfg = config["configurable"]
    try:
        result = mixed_voucher.observe(_user_from_config(config), str(cfg["thread_id"]), voucher_id,
                                       store_code=store_code, visit_at=visit_at, expected_revision=expected_revision,
                                       request_id=str(cfg.get("turn_id") or ""))
        return json.dumps(mixed_voucher.agent_view(result), ensure_ascii=False)
    except voucher_disputes.NotFound as exc:
        return tool_result.fail(tool_result.NOT_FOUND, str(exc))
    except voucher_disputes.Conflict as exc:
        return tool_result.fail(tool_result.CONFLICT, str(exc))
    except ValueError:
        return tool_result.fail(tool_result.INVALID_ARGS, "请核对券号、门店编号与带时区的到店时间")


@tool
def choose_voucher_step(action: Literal["ask_visit", "guide_dispute", "explain", "wait_review", "report_review"],
                        expected_revision: int, config: RunnableConfig) -> str:
    """根据当前券分支事实选择下一步，服务器校验证据和版本。只记录建议，争议由用户在页面确认提交。"""
    cfg = config["configurable"]
    try:
        result = mixed_voucher.choose(_user_from_config(config), str(cfg["thread_id"]), expected_revision,
                                      action, request_id=str(cfg.get("turn_id") or ""))
        return json.dumps(mixed_voucher.agent_view(result), ensure_ascii=False)
    except mixed_voucher.StepConflict as exc:
        return tool_result.fail(tool_result.CONFLICT, str(exc), **exc.state)
    except voucher_disputes.NotFound as exc:
        return tool_result.fail(tool_result.NOT_FOUND, str(exc))
    except voucher_disputes.Conflict as exc:
        return tool_result.fail(tool_result.CONFLICT, str(exc))


@tool
def get_refund_task(config: RunnableConfig) -> str:
    """读取当前会话的退款任务、版本、缺失信息和审批/执行状态。"""
    return json.dumps(refund_tasks.get(
        _user_from_config(config), str(config["configurable"]["thread_id"])
    ), ensure_ascii=False)


@tool
def update_refund_task(
    expected_revision: int, config: RunnableConfig,
    order_id: str | None = None, reason: str | None = None,
    intent: str | None = None, cancel: bool = False,
) -> str:
    """登记用户明确提供的退款信息；intent 为 consult 或 apply。

    先读取任务版本。换订单会清空未重新提供的原因，并撤销原待审批。
    仅咨询必须用 consult；明确申请才用 apply。取消用 cancel=true。
    插话查询物流等不修改此任务。不确定用户是否改口时先追问。
    """
    if plan_store.get(_user_from_config(config), str(config["configurable"]["thread_id"])):
        return tool_result.fail(tool_result.CONFLICT, "本会话有多笔计划，请用 edit_refund_plan_task 指定任务")
    try:
        task = refund_tasks.update(
            _user_from_config(config), str(config["configurable"]["thread_id"]),
            expected_revision=expected_revision, order_id=order_id,
            reason=reason, intent=intent, cancel=cancel,
            source="chat", turn_id=str(config.get("configurable", {}).get("turn_id") or ""),
        )
        return json.dumps(task, ensure_ascii=False)
    except (refund_tasks.TaskConflict, orders.OrderNotFound) as exc:
        return json.dumps({"error": str(exc), "retry": "重新读取任务，向用户核实"}, ensure_ascii=False)


@tool
def submit_refund_request(
    order_id: str, reason: str, task_revision: int, config: RunnableConfig,
) -> str:
    """提交当前版本的退款申请，返回待审批工单。不会执行退款。

    信息齐全且用户明确要求申请才调用。待审批时仍可继续聊天、补充或撤回。
    """
    user = _user_from_config(config)
    tid = str(config["configurable"]["thread_id"])
    if plan_store.get(user, tid):
        return tool_result.fail(tool_result.CONFLICT, "本会话有多笔计划，请用 edit_refund_plan_task 指定任务")
    try:
        if not refund_tasks.get(user, tid)["revision"]:
            raise refund_tasks.TaskConflict("请先登记退款任务并明确申请意图")
        ticket, _ = refund_service.prepare_ticket(
            user, thread_id=tid, order_id=order_id, reason=reason,
            task_revision=task_revision, approval_protocol="detached",
        )
        # A crash here leaves a durable pending ticket; approval can reconstruct
        # the separate execution graph from database truth.
        # Start the execution checkpoint after the dialogue turn finishes,
        # outside the parent graph's tool context/checkpoint namespace.
        return json.dumps({"approval_id": ticket["id"], "status": ticket["status"],
                           "order_id": ticket["order_id"], "amount_cents": ticket["amount_cents"],
                           "required_role": ticket["required_role"], "task_revision": task_revision}, ensure_ascii=False)
    except (refund_tasks.TaskConflict, orders.OrderNotFound, orders.RefundDenied) as exc:
        return json.dumps({"error": str(exc), "submitted": False}, ensure_ascii=False)
