"""Customer actions projected from authenticated workflow state."""

import mixed_voucher


REPLY_INSTRUCTIONS = (
    "普通订单和券分别说明当前状态与下一步。普通单由本人确认时只需要本人确认，"
    "不要再加主管环节；主管审批的单就告知等待主管；只有 executed 才表示已退款。"
    "券待复核只引导查看进展。复核支持申请退款时引导联系售后，工作台链接只查看结论，"
    "当前尚未退款；不要让用户用普通订单确认卡替这张券申请退款。"
)


def project(refund_task, refund_plan, voucher_branch, pending_actions=()):
    steps = []
    tasks = [refund_task] + [branch["state"] for branch in (refund_plan or {}).get("tasks", [])]
    by_scope = {task["thread_id"]: task for task in tasks if task and task.get("thread_id")}
    for action in pending_actions:
        order_id = (action.get("args") or {}).get("order_id")
        approval_id = action.get("approval_id")
        role = (action.get("risk") or {}).get("required_role")
        if (action.get("name") != "create_refund_request" or action.get("blocked_reason")
                or not order_id or not approval_id or role not in {"customer", "approver"}):
            continue
        task = by_scope.get(action.get("scope_thread_id"))
        if task and (task.get("ticket_id") != approval_id or task["status"] != "pending_approval"):
            continue
        if role == "customer":
            steps.append(dict(kind="refund", code="refund_customer_confirm", entity_id=order_id,
                              text="核对订单和原因，然后由你在退款卡片上确认。", approval_id=approval_id))
        else:
            steps.append(dict(kind="refund", code="refund_await_supervisor", entity_id=order_id,
                              text="申请已提交，等待主管审批。"))

    for task in tasks:
        if not task or not task.get("order_id"):
            continue
        terminal = {
            "approved": ("refund_processing", "申请已获批准，退款处理中。"),
            "executing": ("refund_processing", "退款处理中。"),
            "executed": ("refund_executed", "退款已办理，可查看退款单号。"),
            "cancelled": ("refund_withdrawn", "退款申请已撤回。"),
            "rejected": ("refund_rejected", "本次申请未获批准。"),
            "failed": ("refund_failed", "本次退款办理未完成，请联系售后核实。"),
        }.get(task["status"])
        if terminal:
            code, message = terminal
            steps.append(dict(kind="refund", code=code, entity_id=task["order_id"], text=message))

    if voucher_branch:
        steps.append(_voucher_step(voucher_branch))
    return steps


def _voucher_step(branch):
    step = dict(kind="voucher", entity_id=branch["voucher_id"])
    if branch["needs_recheck"]:
        return dict(step, code="voucher_recheck", text="券的核查证据已更新，请重新查看当前结果。",
                    href=branch["help_url"], link_label="重新查看")
    facts = branch["facts"]
    action = mixed_voucher.route(facts)
    if action == "ask_visit":
        labels = {"store_code": "门店编号", "visit_at": "到店时间"}
        fields = "和".join(labels[field] for field in facts["missing"])
        return dict(step, code="voucher_supply_visit", text=f"请补充{fields}，以便核查核销情况。")
    if action == "guide_dispute":
        return dict(step, code="voucher_submit_dispute", text="请在团购券争议页核对资料并提交争议。",
                    href=branch["help_url"], link_label="核对并提交争议")
    if action == "wait_review":
        return dict(step, code="voucher_wait_review", text="争议已提交，等待人工复核。",
                    href=branch["help_url"], link_label="查看争议进展")
    if action == "report_review":
        if facts.get("case_outcome") == "eligible_to_apply":
            return dict(step, code="voucher_contact_support", text="复核支持申请退款，目前尚未退款。请联系售后申请。",
                        href=branch["help_url"], link_label="查看复核结论")
        return dict(step, code="voucher_view_resolution", text="人工复核已完成，可查看结论。",
                    href=branch["help_url"], link_label="查看复核结论")
    return dict(step, code="voucher_explain", text="券核查已完成，可查看核销情况。",
                href=branch["help_url"], link_label="查看核查结果")
