"""最小实验：验证 HumanInTheLoopMiddleware 的审批-中断-恢复机制。

这个脚本不碰项目代码，只用最小的 agent 把机制跑通，
确认三件事：
    1. 工具调用会被拦住，返回 interrupt
    2. 批准后工具真的执行
    3. 拒绝后工具不执行，模型会收到"被拒绝"的反馈

跑通之后再往 api.py 里接，出问题容易定位。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# 记录工具是否真的被执行过
EXECUTION_LOG = []


@tool
def create_refund_request(order_id: str, reason: str) -> str:
    """为用户提交退款申请。这是一个写操作，会真实创建退款单。"""
    EXECUTION_LOG.append({"order_id": order_id, "reason": reason})
    print(f"    >>> [真实执行] 创建退款单 order={order_id} reason={reason}")
    return f"退款申请已提交，单号 RF{order_id}，预计 3 个工作日内审核。"


def build_agent():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        extra_body={"thinking": {"type": "disabled"}},
        timeout=60,
    )
    return create_agent(
        model=llm,
        tools=[create_refund_request],
        system_prompt=(
            "你是店铺客服助手。用户要求退款时，必须调用 create_refund_request 工具，"
            "需要订单号和退款原因。信息不全就先问清楚。"
        ),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"create_refund_request": True},
            )
        ],
        checkpointer=InMemorySaver(),
    )


def show_interrupt(result):
    """把 interrupt 的内容打印出来，看看长什么样。"""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        print("    （没有 interrupt）")
        return None

    print(f"    收到 {len(interrupts)} 个 interrupt")
    for item in interrupts:
        value = getattr(item, "value", item)
        print(f"    interrupt id: {getattr(item, 'id', '?')}")
        print(f"    内容: {value}")
    return interrupts


def main():
    question = "我要申请退款，订单号 A123，商品有质量问题。"

    # ---------------- 场景 1：批准 ----------------
    print("=" * 66)
    print("场景 1：用户提出退款 → 审批通过")
    print("=" * 66)

    agent = build_agent()
    cfg = {"configurable": {"thread_id": "hitl_approve"}}

    print(f"\n[1] 发起请求：{question}")
    r1 = agent.invoke({"messages": [{"role": "user", "content": question}]}, cfg)
    show_interrupt(r1)
    print(f"    此刻工具执行次数：{len(EXECUTION_LOG)}（应为 0，说明被拦住了）")

    print("\n[2] 提交审批结果：approve")
    r2 = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        cfg,
    )
    print(f"    工具执行次数：{len(EXECUTION_LOG)}（应为 1）")
    last = r2["messages"][-1]
    print(f"    最终回答：{str(last.content)[:120]}")

    # ---------------- 场景 2：拒绝 ----------------
    print("\n" + "=" * 66)
    print("场景 2：用户提出退款 → 审批拒绝")
    print("=" * 66)

    EXECUTION_LOG.clear()
    cfg2 = {"configurable": {"thread_id": "hitl_reject"}}

    print(f"\n[1] 发起请求：{question}")
    r3 = agent.invoke({"messages": [{"role": "user", "content": question}]}, cfg2)
    show_interrupt(r3)

    print("\n[2] 提交审批结果：reject")
    r4 = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "reject", "message": "用户信息不全，先让他补充订单截图"}
                ]
            }
        ),
        cfg2,
    )
    print(f"    工具执行次数：{len(EXECUTION_LOG)}（应为 0，说明被拦下了）")
    last = r4["messages"][-1]
    print(f"    最终回答：{str(last.content)[:160]}")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 66)
    print("结论")
    print("=" * 66)
    print("  批准路径：工具执行 1 次，模型基于执行结果作答")
    print("  拒绝路径：工具执行 0 次，模型收到拒绝理由后改口")


if __name__ == "__main__":
    main()
