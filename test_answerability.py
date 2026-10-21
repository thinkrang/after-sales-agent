
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag import debug_retrieval


# 1. 加载 DeepSeek API Key
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有找到 DEEPSEEK_API_KEY")


# 2. 初始化证据核查模型
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=api_key,
    temperature=0,
    extra_body={
        "thinking": {"type": "disabled"}
    },
    timeout=60,
)
def verify_evidence(question:str,results:list[dict])->str:
    """只判断证据是否充足不负责检索"""
    if not results:
        return "NO|未检索文档"
    evidenve="\n\n".join(
        f"【来源：{item["source"]}】\n{item["content"]}"
        for item in results
    )
    system_prompt="""
    你是一个严格的知识库证据核查器。
    你的任务不是回答用户，而是判断：
    提供的知识库文档是否包含足以回答用户问题的明确证据。
    规则：
    1. 只能依据提供的文档。
    2. 不能利用模型自身知识补充答案。
    3. 语义相关不代表能够回答。
    4. 如果用户询问具体号码、金额、日期或其他事实，
    文档必须明确提供这些信息。
    5. 文档没有提到某件事，不代表这件事一定不存在。
    6. 如果证据充分，输出 YES|证据原句。
    7. 如果证据不足，输出 NO|缺少的信息。
    
    只输出上述格式，不要添加其他内容。"""
    response=llm.invoke([
        {"role":"system","content":system_prompt},
        {
            "role":"user",
            "content":(
                f"用户问题：{question}\n\n"
                f"知识库检索内容：\n{evidenve}"
            ),
        },
    ])
    return str(response.content).strip()


# 3. 判断检索证据是否足以回答问题
def check_answerability(question: str)->str:
    """便捷封装：自己检索+判断。保留给测评脚本用。"""
    return verify_evidence(
        question,
        debug_retrieval(question=question,top_k=3)
    )
    # 检索三个候选文档片段
    results = debug_retrieval(
        question=question,
        top_k=3
    )

