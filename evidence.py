"""Evidence sufficiency check with a schema, not a text protocol.

原来让模型输出 ``YES|原句``，再用 ``partition("|")`` 解析。模型多写一句解释、
或者写成"是的|…"，就会被判成"无证据"——一个措辞问题变成了业务结论。

改成 Pydantic 结构化输出后，字段缺失或类型不对会在解析层直接失败，
而不是悄悄降级成"知识库没写"。

不导入检索栈：评测脚本和轻量测试可以直接用它，不必装 chromadb。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

SYSTEM_PROMPT = """
你是一个严格的知识库证据核查器。
你的任务不是回答用户，而是判断：提供的知识库文档是否包含足以回答用户问题的明确证据。

规则：
1. 只能依据提供的文档。
2. 不能利用模型自身知识补充答案。
3. 语义相关不代表能够回答。
4. 如果用户询问具体号码、金额、日期或其他事实，文档必须明确提供这些信息。
5. 文档没有提到某件事，不代表这件事一定不存在。
6. 证据充分时，sufficient=true，并在 quote 中摘录支撑结论的原文。
7. 证据不足时，sufficient=false，并在 missing 中说明缺少哪项关键信息。
""".strip()


class EvidenceVerdict(BaseModel):
    """核查结论。字段本身就是契约，不依赖模型遵守某种文本格式。"""

    sufficient: bool = Field(description="文档是否包含足以回答用户问题的明确证据")
    quote: str = Field(default="", description="支撑结论的证据原句；证据不足时留空")
    missing: str = Field(default="", description="缺少的关键信息；证据充分时留空")


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """惰性创建，import 本模块不需要 API key。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY")
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
        timeout=60,
    )


def evidence_text(results: list[dict]) -> str:
    return "\n\n".join(
        f"【来源：{item['source']}】\n{item['content']}" for item in results
    )


def verify_evidence(question: str, results: list[dict]) -> EvidenceVerdict:
    """返回结构化结论；任何异常都保守地按"证据不足"处理。"""
    if not results:
        return EvidenceVerdict(sufficient=False, missing="未检索到文档")

    try:
        verdict = get_llm().with_structured_output(EvidenceVerdict).invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"用户问题：{question}\n\n知识库检索内容：\n{evidence_text(results)}",
            },
        ])
    except Exception:  # noqa: BLE001
        # 核查器故障不能放行：宁可拒答，也不能让未核查的内容进入回答。
        logging.exception("证据核查调用失败")
        return EvidenceVerdict(sufficient=False, missing="核查器不可用，按证据不足处理")

    if not isinstance(verdict, EvidenceVerdict):
        return EvidenceVerdict(sufficient=False, missing="核查结果无效")

    if verdict.sufficient and not verdict.quote.strip():
        # 声称有证据却不给原句，说明它在凭印象回答。
        return EvidenceVerdict(sufficient=False, missing="核查声称有证据但未给出原文")

    return verdict


def legacy_string(verdict: EvidenceVerdict) -> str:
    """兼容仍按 ``YES|`` / ``NO|`` 解析的评测脚本。"""
    return (f"YES|{verdict.quote}" if verdict.sufficient
            else f"NO|{verdict.missing or '缺少关键信息'}")
