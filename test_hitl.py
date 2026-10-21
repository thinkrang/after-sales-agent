"""测试人工审批的 HTTP 链路。

覆盖三个场景：
    1. 提出退款 → 收到 interrupt 事件，工具未执行
    2. 批准 → 工具执行一次，返回最终回答
    3. 重复批准 → 返回 409，防止写操作被执行两次
"""

import json
import uuid

from fastapi.testclient import TestClient

from api import _REFUND_STORE, app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成 [(event, data), ...]。"""
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


def main():
    _REFUND_STORE.clear()
    tid = "hitl_" + uuid.uuid4().hex[:10]

    with TestClient(app) as client:
        print("=" * 66)
        print("场景 1：用户提出退款申请")
        print("=" * 66)

        r = client.post("/chat/stream", json={
            "thread_id": tid,
            "message": "我要申请退款，订单号 A123，商品有质量问题。",
        })
        events = parse_sse(r.text)
        kinds = [e[0] for e in events]
        print(f"  收到事件: {kinds}")

        interrupt = next((d for n, d in events if n == "interrupt"), None)
        if not interrupt:
            print("  ✗ 没有收到 interrupt，审批没生效")
            return

        print(f"  待审批动作: {interrupt['actions']}")
        print(f"  工具执行次数: {len(_REFUND_STORE)}（应为 0）")

        print("\n" + "=" * 66)
        print("场景 2：审批通过")
        print("=" * 66)

        r2 = client.post("/chat/resume", json={
            "thread_id": tid,
            "decision": "approve",
        })
        events2 = parse_sse(r2.text)
        print(f"  收到事件: {[e[0] for e in events2]}")

        answer = "".join(
            d.get("text", "") for n, d in events2 if n == "token"
        )
        print(f"  最终回答: {answer[:140]}")
        print(f"  工具执行次数: {len(_REFUND_STORE)}（应为 1）")
        print(f"  退款单: {_REFUND_STORE}")

        print("\n" + "=" * 66)
        print("场景 3：重复提交审批（防重复执行）")
        print("=" * 66)

        r3 = client.post("/chat/resume", json={
            "thread_id": tid,
            "decision": "approve",
        })
        print(f"  HTTP 状态码: {r3.status_code}（应为 409）")
        print(f"  响应: {r3.json()}")
        print(f"  工具执行次数: {len(_REFUND_STORE)}（应仍为 1，没有重复建单）")

        print("\n" + "=" * 66)
        print("场景 4：拒绝路径")
        print("=" * 66)

        tid2 = "hitl_" + uuid.uuid4().hex[:10]
        before = len(_REFUND_STORE)

        # 注意：这句必须把"订单号 + 原因 + 明确要求提交"都说到。
        # 只说"帮我退款，不想要了"时模型会先反问退款原因，
        # 根本不会调工具 —— 那是合理的谨慎行为，不是 bug。
        r4a = client.post("/chat/stream", json={
            "thread_id": tid2,
            "message": "我要申请退款，订单号 B777，商品有质量问题，请帮我提交退款申请。",
        })
        if not any(n == "interrupt" for n, _ in parse_sse(r4a.text)):
            print("  ⚠ 本次模型没有触发写操作（可能先去查政策或反问），跳过拒绝路径测试")
            return

        r4 = client.post("/chat/resume", json={
            "thread_id": tid2,
            "decision": "reject",
            "message": "需要用户补充订单截图",
        })
        events4 = parse_sse(r4.text)
        answer4 = "".join(d.get("text", "") for n, d in events4 if n == "token")
        print(f"  收到事件: {[e[0] for e in events4]}")
        print(f"  最终回答: {answer4[:140]}")
        print(f"  工具执行次数变化: {before} -> {len(_REFUND_STORE)}（应不变）")


if __name__ == "__main__":
    main()
