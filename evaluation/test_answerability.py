"""评测脚本的便捷入口：检索 + 证据核查。

核查逻辑已移到 ``evidence.py``（结构化输出，不依赖检索栈）。
这里只保留"自己检索再判断"的封装，以及给旧脚本用的字符串协议。
"""

from evidence import EvidenceVerdict, legacy_string
from evidence import verify_evidence as _verify
from rag import debug_retrieval


def verify_evidence(question: str, results: list[dict]) -> str:
    """兼容旧调用方：返回 ``YES|原句`` / ``NO|缺少信息``。

    新代码请直接用 ``evidence.verify_evidence``——它返回结构化结论，
    不依赖模型遵守某种文本格式。
    """
    return legacy_string(_verify(question, results))


def check_answerability(question: str) -> str:
    """便捷封装：自己检索+判断。保留给测评脚本用。"""
    return verify_evidence(
        question,
        debug_retrieval(question=question, top_k=3),
    )
