"""工具失败码契约：模型靠 code 决策，不靠读一句中文猜下一步。

这些用例只依赖 orders / tool_result，不导入 api -> rag，
所以能在不装 chromadb 的轻量 CI job 里跑。
"""
import json

import pytest

import orders
import tool_result

ORDER = {
    "order_id": "A100",
    "product_name": "测试商品",
    "amount_cents": 19900,
    "status": "delivered",
}


def payload(raw: str) -> dict:
    return json.loads(raw)


def test_success_keeps_existing_fields_and_is_marked_ok():
    data = payload(tool_result.order_snapshot(ORDER, {"eligible": True, "reason": "在退款期内"}))
    assert data["ok"] is True
    assert data["order_id"] == "A100"
    assert data["amount"] == "199.00"
    assert data["refundable"] is True
    assert data["reason"] == "在退款期内"


def test_missing_order_is_terminal_not_retryable():
    data = payload(tool_result.order_lookup_failure(orders.OrderNotFound("订单不存在，或当前账号无权访问")))
    assert data["ok"] is False
    assert data["code"] == tool_result.NOT_FOUND
    # 找不到就是找不到：重试只会撞上循环边界，把"该追问"变成"该停止"。
    assert data["retryable"] is False
    assert "确认订单号" in data["hint"]


def test_transient_failure_is_retryable():
    data = payload(tool_result.order_lookup_failure(TimeoutError("下游超时")))
    assert data["code"] == tool_result.TRANSIENT
    assert data["retryable"] is True
    assert "重试一次" in data["hint"]


def test_failure_payload_does_not_leak_exception_text():
    # 下游异常可能带内部细节；返回给模型的是稳定提示，不是原始报错。
    data = payload(tool_result.order_lookup_failure(RuntimeError("db at 10.0.0.7 refused")))
    assert "10.0.0.7" not in json.dumps(data, ensure_ascii=False)


def test_unknown_code_is_rejected_early():
    with pytest.raises(ValueError):
        tool_result.fail("oops", "未定义的动作")


def test_no_evidence_is_terminal_and_forbids_fabrication():
    data = payload(tool_result.no_evidence("核查结果无效"))
    assert data["ok"] is False
    assert data["code"] == tool_result.NO_EVIDENCE
    # 同一问法再查一次，召回结果不会变，重试只会撞循环边界。
    assert data["retryable"] is False
    assert "不要编造" in data["hint"]
    assert data["reason"] == "核查结果无效"


def test_retrieval_outage_is_not_reported_as_missing_knowledge():
    # 检索挂了和"知识库没写"是两回事：前者可重试，后者必须直接告知无法确认。
    data = payload(tool_result.retrieval_failure())
    assert data["code"] == tool_result.TRANSIENT
    assert data["retryable"] is True
    assert data["code"] != tool_result.NO_EVIDENCE
