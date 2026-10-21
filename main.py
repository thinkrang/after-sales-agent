import os
from threading import current_thread
import re
from uuid import uuid4
from pathlib import  Path

from anyio.lowlevel import checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import  InMemorySaver


load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请先在 .env 文件中填写 DEEPSEEK_API_KEY")


@tool
def search_shop_info(question: str) -> str:
    """查询店铺的退款、退货和发货规则"""
    print(f"[调用工具] 查询店铺规则：{question}")
    if "退货" in question or "退款" in question:
        return "店铺支持签收后7天无理由退货，商品需保持完好。"
    if "发货" in question or "快递" in question:
        return "订单付款后48小时内发货，默认使用普通快递。"
    return "暂无相关店铺规则。"


@tool
def calculate_total(price: float, quantity: int) -> str:
    """根据商品单价和购买数量，计算订单总金额"""
    print(f"[调用工具] 计算金额：{price} x {quantity}")
    total = round(price * quantity, 2)
    return f"订单总金额为 {total} 元。"


llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    extra_body={"thinking": {"type": "disabled"}},
    timeout=60,
)

memory=InMemorySaver()
db_path=Path(__file__).resolve().parent/"chat_memory.sqlite"
with SqliteSaver.from_conn_string(str(db_path)) as memory:
    agent=create_agent(
        model=llm,
        tools=[search_shop_info,calculate_total],
        system_prompt=(
            "你是一个友好的中文店铺客服助手。"
            "请结合历史对话回答用户。"
            "遇到店铺规则查询或订单金额计算时，调用对应工具。"),
        checkpointer=memory,
    )
    current_thread="user_001"
    print("===chatagent多会话版===")
    print("支持命令：新建、切换 会话ID、当前、退出")
    while True:
        try:
            question=input(f"\n[{current_thread}]你:").strip()
            if not question:
                continue
            if question in["退出","exit"]:
                print("再见！")
                break
            if question=="新建" :
                current_thread=f"cha_{uuid4().hex[:8]}"
                print(f"新会话已创建：{current_thread}")
                continue

            # 切换历史会话
            if question.startswith("切换"):
                thread_id = question[len("切换"):].strip()

                if not thread_id:
                    print("[输入错误] 缺少会话ID!")
                    print("请输入正确会话ID，例如：切换 cha_09ebef93")
                    continue
                    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", thread_id):
                        print("[输入错误] 会话ID格式不正确！")
                    print("只能包含字母、数字、下划线和连字符。")
                    print("示例：切换 cha_09ebef93")
                    continue

                current_thread = thread_id
                print(f"已切换至会话：{current_thread}")
                continue
            if question == "当前":
                print(f"[当前会话]{current_thread}")
                continue
            if question=="列表":
                checkpoints=memory.list(None)
                thread_ids=sorted({
                    cp.config["configurable"]["thread_id"]
                    for cp in checkpoints
                })
                if not thread_ids:
                    print("暂无历史会话。")
                    continue
                print("\n===历史会话列表===")
                for i,thread_id in enumerate(thread_ids,1):
                    if thread_id==current_thread:
                        print(f"{i},{thread_id}(当前)")
                    else:
                        print(f"{i},{thread_id}")
                print(f"\n共 {len(thread_ids)} 个历史会话。")
                continue


     # 为当前会话配置 thread_id
            config = {
                "configurable": {
                    "thread_id": current_thread
                }
            }
            try:
                result = agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": question}
                        ]
                },
                config=config
            )
                answer=result["messages"][-1].content
                print(f"\nAgent:{answer}")
            except Exception as e:
                print(f"\n[Agent运行错误] {type(e).__name__}: {e}")
                print("请检查错误信息，可以继续输入。")
                continue

        except (KeyboardInterrupt, EOFError):
            print("\n程序已结束。")
            break
