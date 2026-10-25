"""Refund authorization, approval binding and idempotent execution tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

import db
import orders
import refund_service
import tickets
from auth import User


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    yield
    db.release_all()


@pytest.fixture
def manager():
    return User(
        "acme-manager",
        "ACME 售后主管",
        "acme",
        frozenset({"customer", "approver", "knowledge_admin"}),
    )


@pytest.fixture
def other_user():
    return User(
        "globex-user",
        "GLOBEX 客户",
        "globex",
        frozenset({"customer"}),
    )


def test_order_is_scoped_to_authenticated_user(manager, other_user, workflow):
    assert orders.get_order(manager, "A100")["order_id"] == "A100"
    with pytest.raises(orders.OrderNotFound):
        orders.get_order(other_user, "A100")


def test_direct_execution_without_approval_fails(manager, workflow):
    refund_service.prepare_ticket(
        manager,
        thread_id="thread-1",
        order_id="A100",
        reason="尺码不合适",
    )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-1",
            order_id="A100",
            reason="尺码不合适",
        )
    assert tickets.find_executed(
        "A100", tenant_id="acme", requester_id="acme-manager"
    ) is None


def test_approval_is_bound_to_exact_order_reason_and_thread(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-2",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-2",
        decision="approve",
    )

    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-2",
            order_id="A100",
            reason="商品有质量问题",
        )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="another-thread",
            order_id="A100",
            reason="尺码不合适",
        )


def test_approved_refund_executes_and_retry_returns_same_refund(
    manager, workflow
):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-3",
        decision="approve",
    )

    first = refund_service.execute(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    second = refund_service.execute(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )

    assert first["status"] == tickets.EXECUTED
    assert first["refund_id"] == second["refund_id"]
    assert first["execution_id"] == first["id"]
    matched, _ = refund_service.find_action_ticket(
        manager,
        thread_id="thread-3",
        order_id="A100",
        reason="尺码不合适",
    )
    assert matched["id"] == first["id"]
    assert matched["status"] == tickets.EXECUTED


def test_rejected_refund_cannot_execute(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-4",
        order_id="A200",
        reason="不想要了",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-4",
        decision="reject",
    )
    with pytest.raises(refund_service.ApprovalRequired):
        refund_service.execute(
            manager,
            thread_id="thread-4",
            order_id="A200",
            reason="不想要了",
        )


def test_duplicate_approval_submission_is_rejected(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-5",
        order_id="A200",
        reason="不想要了",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-5",
        decision="approve",
    )
    with pytest.raises(refund_service.WorkflowConflict):
        refund_service.decide(
            manager,
            approval_id=pending["id"],
            thread_id="thread-5",
            decision="approve",
        )


def test_execution_recovers_after_gateway_succeeded_but_ticket_not_updated(
    manager, workflow
):
    pending, order = refund_service.prepare_ticket(
        manager,
        thread_id="thread-6",
        order_id="A200",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-6",
        decision="approve",
    )
    payload_hash = refund_service.action_payload_hash(
        user=manager,
        thread_id="thread-6",
        order_id="A200",
        amount_cents=order["amount_cents"],
        reason="尺码不合适",
    )
    claimed = tickets.acquire_execution(
        "thread-6",
        tenant_id="acme",
        requester_id="acme-manager",
        order_id="A200",
        payload_hash=payload_hash,
    )
    assert claimed["status"] == tickets.EXECUTING

    # Simulate the downstream refund succeeding before the process dies.
    execution = orders.execute_refund(
        user=manager,
        order_id="A200",
        amount_cents=order["amount_cents"],
        idempotency_key=pending["id"],
    )
    assert execution["refund_id"]

    repaired = refund_service.reconcile_executing(
        manager,
        thread_id="thread-6",
        order_id="A200",
        reason="尺码不合适",
    )
    assert repaired["status"] == tickets.EXECUTED
    assert repaired["refund_id"] == execution["refund_id"]


def test_reconciliation_worker_finishes_approved_ticket(manager, workflow):
    pending, _ = refund_service.prepare_ticket(
        manager,
        thread_id="thread-recover-approved",
        order_id="A100",
        reason="尺码不合适",
    )
    refund_service.decide(
        manager,
        approval_id=pending["id"],
        thread_id="thread-recover-approved",
        decision="approve",
    )

    recoverable = tickets.list_recoverable()
    assert [row["id"] for row in recoverable] == [pending["id"]]

    recovered = refund_service.reconcile_ticket(recoverable[0])
    assert recovered["status"] == tickets.EXECUTED
    assert recovered["refund_id"]


def test_concurrent_downstream_retry_creates_one_refund(manager, workflow):
    key = "fixed-idempotency-key"

    def run_once(_):
        return orders.execute_refund(
            user=manager,
            order_id="A100",
            amount_cents=12900,
            idempotency_key=key,
        )["refund_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        refund_ids = list(pool.map(run_once, range(16)))

    assert len(set(refund_ids)) == 1


# ---------------------------------------------------------------- 风险分级与四眼原则


@pytest.fixture
def acme_customer():
    return User("acme-user", "ACME 客户", "acme", frozenset({"customer"}))


@pytest.fixture
def acme_approver():
    """同租户的另一个审批人。四眼原则要求它和发起人不是同一个人。"""
    return User(
        "acme-approver", "ACME 审批人", "acme", frozenset({"customer", "approver"})
    )


def test_high_risk_ticket_records_risk_fields(acme_customer, workflow):
    """高风险单必须把分级、原因、规则版本落在工单上——否则事后无法解释为什么转审批。"""
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-high", order_id="A400", reason="质量问题"
    )
    assert ticket["risk_tier"] == "high"
    assert ticket["required_role"] == "approver"
    assert ticket["rule_version"]
    import json

    reasons = json.loads(ticket["risk_reasons"])
    assert reasons and any("自动审批上限" in r for r in reasons)


def test_low_risk_ticket_stays_with_customer(manager, workflow):
    ticket, _ = refund_service.prepare_ticket(
        manager, thread_id="t-low", order_id="A100", reason="质量问题"
    )
    assert ticket["risk_tier"] == "low"
    assert ticket["required_role"] == "customer"


def test_requester_cannot_approve_own_high_risk(acme_customer, workflow):
    """高风险单，发起人自己点确认 → 拒绝。

    这是四眼原则的核心：风险高到需要审批时，"本人确认"就没有风控意义了。
    """
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-self", order_id="A400", reason="质量问题"
    )
    with pytest.raises(refund_service.ApprovalForbidden):
        refund_service.decide(
            acme_customer,
            approval_id=ticket["id"],
            thread_id="t-self",
            decision="approve",
        )


def test_manager_cannot_approve_own_high_risk(manager, workflow):
    """主管自己发起的高风险单，自己也不能批。

    A500 签收 6 天，距退款截止不足 24 小时 → 边界政策 → high。
    """
    ticket, _ = refund_service.prepare_ticket(
        manager, thread_id="t-mgr", order_id="A500", reason="质量问题"
    )
    assert ticket["required_role"] == "approver"
    with pytest.raises(refund_service.ApprovalForbidden):
        refund_service.decide(
            manager, approval_id=ticket["id"], decision="approve"
        )


def test_another_approver_can_approve_and_execute(
    acme_customer, acme_approver, workflow
):
    """另一个审批人可以批，批完直接执行成功。"""
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-ok", order_id="A400", reason="质量问题"
    )
    outcome = refund_service.decide(
        acme_approver, approval_id=ticket["id"], decision="approve"
    )
    assert outcome.status == tickets.APPROVED

    executed = refund_service.execute_for_ticket(outcome.ticket)
    assert executed["status"] == tickets.EXECUTED
    assert executed["refund_id"]


def test_non_approver_cannot_approve_high_risk(acme_customer, workflow):
    """没有 approver 角色的人（哪怕是同租户客户）不能批高风险单。

    这里返回的是 404 而不是 403：既不是发起人、也没有审批角色的人
    不该知道这张工单存在。和"访问他人会话返回 404"是同一个原则。
    """
    other = User("acme-user2", "另一个客户", "acme", frozenset({"customer"}))
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-role", order_id="A400", reason="质量问题"
    )
    with pytest.raises(orders.OrderNotFound):
        refund_service.decide(other, approval_id=ticket["id"], decision="approve")


def test_approver_cannot_touch_others_low_risk(manager, acme_approver, workflow):
    """低风险单只能本人处理。审批人越权处理别人的低风险单也要拒绝——

    否则"有 approver 角色"就变成可以替任何人退款。
    """
    ticket, _ = refund_service.prepare_ticket(
        manager, thread_id="t-low2", order_id="A100", reason="质量问题"
    )
    with pytest.raises(refund_service.ApprovalForbidden):
        refund_service.decide(
            acme_approver, approval_id=ticket["id"], decision="approve"
        )


def test_manager_reject_high_risk(acme_customer, acme_approver, workflow):
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-rej", order_id="A400", reason="质量问题"
    )
    outcome = refund_service.decide(
        acme_approver,
        approval_id=ticket["id"],
        decision="reject",
        message="金额过大，需线下核实",
    )
    assert outcome.status == tickets.REJECTED


def test_high_risk_approval_is_idempotent_on_retry(
    acme_customer, acme_approver, workflow
):
    """主管重复提交审批 → 409 而不是重复执行。"""
    ticket, _ = refund_service.prepare_ticket(
        acme_customer, thread_id="t-idem", order_id="A400", reason="质量问题"
    )
    refund_service.decide(
        acme_approver, approval_id=ticket["id"], decision="approve"
    )
    with pytest.raises(refund_service.WorkflowConflict):
        refund_service.decide(
            acme_approver, approval_id=ticket["id"], decision="approve"
        )


# ---------------------------------------------------------------- failed 语义


def _approved_ticket(user, thread, order_id="A100"):
    ticket, _ = refund_service.prepare_ticket(
        user, thread_id=thread, order_id=order_id, reason="质量问题"
    )
    refund_service.decide(user, approval_id=ticket["id"], thread_id=thread,
                          decision="approve")
    return ticket


def test_transient_error_keeps_ticket_retryable(manager, workflow, monkeypatch):
    """瞬时故障不能把工单打成 failed —— 那笔退款就永远补不上了。"""
    ticket = _approved_ticket(manager, "tr1")

    def boom(**kwargs):
        raise RuntimeError("数据库锁超时")

    monkeypatch.setattr(orders, "execute_refund", boom)
    with pytest.raises(RuntimeError):
        refund_service.execute(manager, thread_id="tr1", order_id="A100",
                               reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.EXECUTING, "应留在 executing 等对账重试"
    assert current["attempts"] == 1
    assert "数据库锁超时" in (current["last_error"] or "")
    # 仍然能被对账捞出来
    assert any(x["id"] == ticket["id"] for x in tickets.list_recoverable())


def test_retry_limit_escalates_to_failed(manager, workflow, monkeypatch):
    """重试到上限仍未成功，才转 failed。"""
    ticket = _approved_ticket(manager, "tr2")

    def boom(**kwargs):
        raise RuntimeError("持续失败")

    monkeypatch.setattr(orders, "execute_refund", boom)

    for _ in range(tickets.MAX_EXECUTION_ATTEMPTS):
        with pytest.raises(RuntimeError):
            refund_service.execute(manager, thread_id="tr2", order_id="A100",
                                   reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.FAILED
    assert current["attempts"] == tickets.MAX_EXECUTION_ATTEMPTS
    assert "需人工核实" in (current["last_error"] or "")
    # 终态不再被对账捞出
    assert not any(x["id"] == ticket["id"] for x in tickets.list_recoverable())


def test_business_denial_fails_immediately(manager, workflow, monkeypatch):
    """业务规则拒绝是确定性失败，重试也不会变，直接终结。"""
    ticket = _approved_ticket(manager, "tr3")

    def deny(**kwargs):
        raise orders.RefundDenied("审批金额与订单金额不一致")

    monkeypatch.setattr(orders, "execute_refund", deny)
    with pytest.raises(orders.RefundDenied):
        refund_service.execute(manager, thread_id="tr3", order_id="A100",
                               reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.FAILED
    assert current["attempts"] == 0, "业务拒绝不该算一次重试"


def test_reconcile_recovers_after_transient_error(manager, workflow, monkeypatch):
    """瞬时故障之后，对账能把它补完——而且用同一个退款单号。"""
    ticket = _approved_ticket(manager, "tr4")

    original = orders.execute_refund

    def boom(**kwargs):
        raise RuntimeError("临时不可用")

    monkeypatch.setattr(orders, "execute_refund", boom)
    with pytest.raises(RuntimeError):
        refund_service.execute(manager, thread_id="tr4", order_id="A100",
                               reason="质量问题")

    # 只还原这一个补丁。用 monkeypatch.undo() 会把 fixture 里的
    # DB_PATH 补丁一起撤掉，导致后面查的是真实数据库。
    monkeypatch.setattr(orders, "execute_refund", original)
    rows = [x for x in tickets.list_recoverable() if x["id"] == ticket["id"]]
    assert rows, "故障后应仍可被对账捞出"
    result = refund_service.reconcile_ticket(rows[0])
    assert result["status"] == tickets.EXECUTED
    assert result["refund_id"]


# ---------------------------------------------------------------- failed 语义


def _approved_ticket(user, thread, order_id="A100"):
    ticket, _ = refund_service.prepare_ticket(
        user, thread_id=thread, order_id=order_id, reason="质量问题"
    )
    refund_service.decide(user, approval_id=ticket["id"], thread_id=thread,
                          decision="approve")
    return ticket


def test_transient_error_keeps_ticket_retryable(manager, workflow, monkeypatch):
    """瞬时故障不能把工单打成 failed —— 那笔退款就永远补不上了。"""
    ticket = _approved_ticket(manager, "tr1")

    def boom(**kwargs):
        raise RuntimeError("数据库锁超时")

    monkeypatch.setattr(orders, "execute_refund", boom)
    with pytest.raises(RuntimeError):
        refund_service.execute(manager, thread_id="tr1", order_id="A100",
                               reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.EXECUTING, "应留在 executing 等对账重试"
    assert current["attempts"] == 1
    assert "数据库锁超时" in (current["last_error"] or "")
    # 仍然能被对账捞出来
    assert any(x["id"] == ticket["id"] for x in tickets.list_recoverable())


def test_retry_limit_escalates_to_failed(manager, workflow, monkeypatch):
    """重试到上限仍未成功，才转 failed。"""
    ticket = _approved_ticket(manager, "tr2")

    def boom(**kwargs):
        raise RuntimeError("持续失败")

    monkeypatch.setattr(orders, "execute_refund", boom)

    for _ in range(tickets.MAX_EXECUTION_ATTEMPTS):
        with pytest.raises(RuntimeError):
            refund_service.execute(manager, thread_id="tr2", order_id="A100",
                                   reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.FAILED
    assert current["attempts"] == tickets.MAX_EXECUTION_ATTEMPTS
    assert "需人工核实" in (current["last_error"] or "")
    # 终态不再被对账捞出
    assert not any(x["id"] == ticket["id"] for x in tickets.list_recoverable())


def test_business_denial_fails_immediately(manager, workflow, monkeypatch):
    """业务规则拒绝是确定性失败，重试也不会变，直接终结。"""
    ticket = _approved_ticket(manager, "tr3")

    def deny(**kwargs):
        raise orders.RefundDenied("审批金额与订单金额不一致")

    monkeypatch.setattr(orders, "execute_refund", deny)
    with pytest.raises(orders.RefundDenied):
        refund_service.execute(manager, thread_id="tr3", order_id="A100",
                               reason="质量问题")

    current = tickets.get_ticket(ticket["id"])
    assert current["status"] == tickets.FAILED
    assert current["attempts"] == 0, "业务拒绝不该算一次重试"


def test_reconcile_recovers_after_transient_error(manager, workflow, monkeypatch):
    """瞬时故障之后，对账能把它补完——而且用同一个退款单号。"""
    ticket = _approved_ticket(manager, "tr4")

    original = orders.execute_refund

    def boom(**kwargs):
        raise RuntimeError("临时不可用")

    monkeypatch.setattr(orders, "execute_refund", boom)
    with pytest.raises(RuntimeError):
        refund_service.execute(manager, thread_id="tr4", order_id="A100",
                               reason="质量问题")

    # 只还原这一个补丁。用 monkeypatch.undo() 会把 fixture 里的
    # DB_PATH 补丁一起撤掉，导致后面查的是真实数据库。
    monkeypatch.setattr(orders, "execute_refund", original)
    rows = [x for x in tickets.list_recoverable() if x["id"] == ticket["id"]]
    assert rows, "故障后应仍可被对账捞出"
    result = refund_service.reconcile_ticket(rows[0])
    assert result["status"] == tickets.EXECUTED
    assert result["refund_id"]
