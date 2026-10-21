"""检索层测试：策略行为、结果结构、格式化。

这些测试不经过 HTTP，直接调 retrieval / rag 的函数，
跑得快，也更容易定位问题。
"""

import pytest

from rag import debug_retrieval, format_evidence
from retrieval import (
    STRATEGIES,
    bm25_retrieval,
    hybrid_retrieval,
    reranker_ready,
    retrieve,
    vector_retrieval,
)

QUESTION = "退货政策是什么"


def test_vector_retrieval_structure():
    results = vector_retrieval(QUESTION, 5)
    assert results, "向量检索不应为空"
    assert len(results) <= 5

    first = results[0]
    for key in ("rank", "source", "content", "distance"):
        assert key in first

    # rank 必须从 1 开始连续递增
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


def test_bm25_returns_scored_results():
    results = bm25_retrieval("退货政策", 5)
    assert results
    assert all(r["score"] > 0 for r in results), "BM25 不该返回 0 分结果"
    # 分数必须降序
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_on_no_keyword_overlap():
    """完全没有词重叠时应该返回空，而不是硬凑结果。"""
    results = bm25_retrieval("zzzzqqqq", 5)
    assert results == []


def test_hybrid_falls_back_to_vector_when_bm25_weight_zero():
    """当前配置 BM25 权重为 0，hybrid 应当与 vector 结果一致。

    这条测试是在守护一个决策：我们实测 BM25 融合有害，
    把权重设成 0 了。如果哪天有人改回非零权重，这条会失败，
    提醒他先跑一遍评测再改。
    """
    vec = vector_retrieval(QUESTION, 5)
    hyb = hybrid_retrieval(QUESTION, 5)
    assert [r["source"] for r in vec] == [r["source"] for r in hyb]


def test_retrieve_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="未知策略"):
        retrieve(QUESTION, strategy="no_such_strategy")


def test_all_strategies_registered():
    assert set(STRATEGIES) == {"vector", "bm25", "hybrid", "hybrid_rerank"}


@pytest.mark.skipif(not reranker_ready(), reason="reranker 模型未下载")
def test_rerank_returns_scores():
    results = STRATEGIES["hybrid_rerank"](QUESTION, 3)
    assert results
    assert all("rerank_score" in r for r in results)


def test_format_evidence_has_source_header():
    results = vector_retrieval(QUESTION, 3)
    text = format_evidence(results)
    assert "【来源：" in text
    # 每一条结果都要有对应的来源标注
    assert text.count("【来源：") == len(results)


def test_format_evidence_empty():
    assert format_evidence([]) == "没有找到相关文档。"


@pytest.mark.parametrize("bad_top_k", [0, -1, 11])
def test_debug_retrieval_rejects_bad_top_k(bad_top_k):
    with pytest.raises(ValueError):
        debug_retrieval(QUESTION, bad_top_k)


def test_debug_retrieval_rejects_empty_question():
    with pytest.raises(ValueError):
        debug_retrieval("   ", 3)


def test_all_knowledge_files_are_indexed():
    """knowledge/ 里的每个 txt 都应该能在索引里找到。

    这条能抓住一类真实问题：加了文档但忘记重建索引，
    或者 load_documents 只 glob *.txt 导致 md 文件被静默忽略。
    """
    from pathlib import Path

    from rag import get_vector_store

    files = {p.name for p in Path("knowledge").glob("*.txt")}
    indexed = {
        (m or {}).get("source")
        for m in get_vector_store().get(include=["metadatas"])["metadatas"]
    }
    missing = files - indexed
    assert not missing, f"这些文件没进索引：{missing}"
