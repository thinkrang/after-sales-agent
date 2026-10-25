"""人工审批的 HTTP 端到端测试（需要真实大模型）。

为什么单独标记 llm
------------------
这个文件跑的是完整链路：真实 LLM 决定调工具 → 图中断 → 恢复执行。
每次约 3-5 秒，四条用例要 20 秒以上，而且花 token。

所以标 `@pytest.mark.llm`，默认被 pytest.ini 的 `-m "not llm"` 排除。
需要验证真实链路时显式跑：

    pytest -m llm

SSE 协议本身的正确性由 tests/test_chat_sse.py 用 FakeAgent 覆盖，
那部分毫秒级、零成本，每次提交都会跑。
"""

import json
import uuid

import pytest

import tickets

pytestmark = pytest.mark.llm

# 这句必须把「订单号 + 原因 + 明确要求提交」都说到。
# 只说"帮我退款，不想要了"时模型会先反问原因，根本不会调工具 ——
# 那是合理的谨慎行为，不是 bug。
REFUND_REQUEST = "我要申请退款，订单号 {order}，商品有质量问题，请帮我提交退款申请。"


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = "message"
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            try:
                events.append((name, json.loads("\n".join(data_lines))))
            except json.JSONDecodeError:
                pass
    return events


def answer_of(events: list[tuple[str, dict]]) -> str:
    return "".join(d.get("text", "") for k, d in events if k == "token")


def test_interrupt_then_approve(client):
    """提出退款 → 中断 → 批准 → 工单 executed。"""
    tid = "llm_" + uuid.uuid4().hex[:10]
    order_id = "A100"

    try:
        # 1) 提出退款，应该停在审批点
        r = client.post("/chat/stream", json={
            "thread_id": tid,
            "message": REFUND_REQUEST.format(order=order_id),
        })
        events = parse_sse(r.text)
        kinds = [k for k, _ in events]

        assert "interrupt" in kinds, f"没有触发审批中断，事件序列：{kinds}"
        assert "done" not in kinds

        interrupt = next(d for k, d in events if k == "interrupt")
        assert interrupt["actions"][0]["name"] == "create_refund_request"
        approval_id = interrupt["actions"][0]["approval_id"]

        # 此刻不该有已执行的退款单
        assert tickets.find_executed(
            order_id, tenant_id="acme", requester_id="acme-manager"
        ) is None

        # 工单应该处于待审批
        pending = tickets.find_pending(
            tid, tenant_id="acme", requester_id="acme-manager"
        )
        assert pending is not None
        assert pending["status"] == tickets.PENDING

        # 2) 批准
        r2 = client.post("/chat/resume", json={
            "thread_id": tid,
            "approval_id": approval_id,
            "decision": "approve",
        })
        events2 = parse_sse(r2.text)
        assert answer_of(events2), "批准后应该有回答"

        # 3) 工单应该流转到 executed
        executed = tickets.find_executed(
            order_id, tenant_id="acme", requester_id="acme-manager"
        )
        assert executed is not None
        assert executed["status"] == tickets.EXECUTED
        assert executed["refund_id"]
        assert executed["decision"] == "approve"

        # 4) 重复批准应被 409 挡住
        r3 = client.post("/chat/resume", json={
            "thread_id": tid,
            "approval_id": approval_id,
            "decision": "approve",
        })
        assert r3.status_code == 409

        # 5) 确认没有重复建单
        assert len([
            t
            for t in tickets.list_tickets(
                200, tenant_id="acme", requester_id="acme-manager"
            )
            if t["order_id"] == order_id
        ]) == 1

    finally:
        client.delete(f"/sessions/{tid}")


def test_interrupt_then_reject(client):
    """拒绝路径：工单 rejected，工具不执行。"""
    tid = "llm_" + uuid.uuid4().hex[:10]
    order_id = "A200"

    try:
        r = client.post("/chat/stream", json={
            "thread_id": tid,
            "message": REFUND_REQUEST.format(order=order_id),
        })
        events = parse_sse(r.text)
        if not any(k == "interrupt" for k, _ in events):
            pytest.skip("本次模型没有触发写操作（可能先去查政策了），跳过")
        interrupt = next(d for k, d in events if k == "interrupt")
        approval_id = interrupt["actions"][0]["approval_id"]

        r2 = client.post("/chat/resume", json={
            "thread_id": tid,
            "approval_id": approval_id,
            "decision": "reject",
            "message": "需要用户补充订单截图",
        })
        assert r2.status_code == 200
        assert answer_of(parse_sse(r2.text)), "拒绝后模型应该给出回应"

        # 工具不该执行
        assert tickets.find_executed(
            order_id, tenant_id="acme", requester_id="acme-manager"
        ) is None

        # 工单应该是 rejected
        rejected = [t for t in tickets.list_tickets(
            200, tenant_id="acme", requester_id="acme-manager"
        )
                    if t["order_id"] == order_id and t["status"] == tickets.REJECTED]
        assert rejected, "拒绝后应有一条 rejected 工单"
        assert rejected[0]["decision"] == "reject"

    finally:
        client.delete(f"/sessions/{tid}")
