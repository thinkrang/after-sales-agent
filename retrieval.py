"""检索策略层：向量 / BM25 / RRF 融合 / Reranker。

为什么要单独一个文件
--------------------
rag.py 已经承担了索引构建、文档增删、权限、检索等职责，
再往里塞三种检索策略会变成第二个"什么都干"的模块。
这里只负责"给定问题，返回排序好的结果"，索引的读写仍走 rag.py。

四种策略
--------
1. vector        纯向量语义检索（基线）
2. bm25          纯关键词检索（BM25 + jieba 分词）
3. hybrid        向量 + BM25，用 RRF 融合
4. hybrid_rerank hybrid 召回更多候选，再用 CrossEncoder 精排

为什么用 RRF 而不是加权分数
--------------------------
向量距离（越小越好）和 BM25 分数（越大越好）量纲完全不同，
直接加权需要归一化，而归一化方式会显著影响结果。
RRF（Reciprocal Rank Fusion）只用"排名"不用"分数"：
    score = Σ 1 / (k + rank)
天然免疫量纲问题，且对异常分数不敏感。k 一般取 60。
"""

import re
from functools import lru_cache

import jieba
from rank_bm25 import BM25Okapi

from rag import debug_retrieval, get_vector_store

# RRF 常数，经验值 60
RRF_K = 60

# 混合召回时每一路取多少候选
CANDIDATES_PER_ROUTE = 10

# RRF 融合权重：1.0 : 0.0（纯向量）。
#
# ⚠️ 重要前提：这个模块目前只被 evaluate_rag.py / tune_rrf.py 使用。
# 线上 agent 走的是 rag.debug_retrieval()，那是另一条路径（直接调
# similarity_search_with_score，带租户过滤，top_k=3）。两条路径没有统一，
# 所以**这里的权重改了不会影响 agent 的回答**。
#
# 这个不一致是 2026-10-25 扩语料时发现的，见 README「已知问题」。
# 在统一之前，下面的调参结论只对评测脚本成立，不能当作线上效果。
#
# 历史调参记录（仅供统一路径后参考）：
#   22 chunk：1.0:0.0 Hit@1 92.11% / MRR 0.9561，任何非零 BM25 权重都掉点
#   152 chunk：1.0:0.0 Recall@5 95.52% Hit@1 80.60%
#              0.5:0.5 Recall@5 97.01% Hit@1 79.10%（Recall 涨、Hit@1 跌）
#
# 152 chunk 下混合召回确实补回了漏召（"商品编号001"这类精确实体查询，
# 向量会错配到别的 SKU，BM25 能精确命中），但那条路径 agent 用不上。
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 0.0

# Reranker 模型（中文场景常用）
RERANKER_MODEL = "BAAI/bge-reranker-base"

_TOKEN_RE = re.compile(r"[\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """中文分词 + 过滤纯符号。

    为什么不直接用空格切：中文没有词边界，BM25 需要词粒度的 token。
    """
    tokens = jieba.lcut(text or "")
    return [t.strip().lower() for t in tokens if t.strip() and not _TOKEN_RE.fullmatch(t)]


# ---------------------------------------------------------------- 语料快照


@lru_cache(maxsize=1)
def _corpus():
    """把整个集合读进内存，作为 BM25 的语料。

    返回 (ids, documents, metadatas)。
    注意：这里假设集合在评测/运行期间不变。增删文档后需要调
    refresh_corpus() 清缓存，否则 BM25 会检索到已删除的旧内容。
    """
    store = get_vector_store()
    raw = store.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    return ids, docs, metas


def refresh_corpus() -> None:
    """索引发生变化后调用，清掉 BM25 语料缓存。"""
    _corpus.cache_clear()


@lru_cache(maxsize=1)
def _bm25():
    ids, docs, metas = _corpus()
    tokenized = [tokenize(d) for d in docs]
    return BM25Okapi(tokenized), ids, docs, metas


# ---------------------------------------------------------------- 各策略


def vector_retrieval(question: str, top_k: int) -> list[dict]:
    """纯向量检索，直接复用 rag.debug_retrieval 的口径。"""
    return debug_retrieval(question=question, top_k=top_k)


def bm25_retrieval(question: str, top_k: int) -> list[dict]:
    """纯关键词检索。"""
    bm25, ids, docs, metas = _bm25()
    scores = bm25.get_scores(tokenize(question))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(order, start=1):
        if scores[idx] <= 0:
            # BM25 分数为 0 表示没有任何词命中，没必要往下排
            break
        meta = metas[idx] or {}
        results.append({
            "rank": rank,
            "id": ids[idx],
            "source": meta.get("source", "未知文档"),
            "page": meta.get("page"),
            "score": round(float(scores[idx]), 6),
            "content": docs[idx],
        })
    return results


def _rrf_fuse(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """加权 Reciprocal Rank Fusion：只吃排名，不吃分数。

    weights 让某一路的排名更有话语权。等权（都是 1.0）时，
    如果一路明显更弱，它的坏结果会把另一路的好结果拉下来。
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieval(question: str, top_k: int) -> list[dict]:
    """向量 + BM25，加权 RRF 融合。

    默认权重向量 0.8 / BM25 0.2：向量在本语料上明显更强
    （BM25 对口语化改写失分严重），所以只给它少量话语权，
    让它能在向量失手时起补救作用，而不是主导排序。

    两路各取 CANDIDATES_PER_ROUTE 个候选，融合后再截 top_k。
    """
    n = max(top_k, CANDIDATES_PER_ROUTE)

    vec = vector_retrieval(question, n)
    kw = bm25_retrieval(question, n)

    ids, docs, metas = _corpus()
    id_to_idx = {doc_id: i for i, doc_id in enumerate(ids)}

    # 向量结果没有 id，用内容反查（当前语料每篇 1 个 chunk，内容唯一）
    content_to_id = {d: ids[i] for i, d in enumerate(docs)}

    vec_ids = []
    for item in vec:
        doc_id = content_to_id.get(item["content"])
        if doc_id:
            vec_ids.append(doc_id)

    kw_ids = [item["id"] for item in kw]

    fused = _rrf_fuse([vec_ids, kw_ids], weights=[VECTOR_WEIGHT, BM25_WEIGHT])

    results = []
    for rank, (doc_id, score) in enumerate(fused[:top_k], start=1):
        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue
        meta = metas[idx] or {}
        results.append({
            "rank": rank,
            "id": doc_id,
            "source": meta.get("source", "未知文档"),
            "page": meta.get("page"),
            "rrf_score": round(score, 6),
            "content": docs[idx],
        })
    return results


# ---------------------------------------------------------------- Reranker


@lru_cache(maxsize=1)
def _reranker():
    """懒加载 CrossEncoder。模型缺失时返回 None，调用方降级处理。"""
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL, max_length=512)
    except Exception as e:
        print(f"[Reranker] 不可用（{type(e).__name__}: {str(e)[:80]}）")
        return None


def reranker_ready() -> bool:
    return _reranker() is not None


def hybrid_rerank_retrieval(question: str, top_k: int) -> list[dict]:
    """混合召回 + CrossEncoder 精排。

    如果 reranker 不可用，自动降级为 hybrid，并在结果的 strategy 字段标注，
    避免"以为跑了 reranker 其实没跑"。
    """
    model = _reranker()
    if model is None:
        out = hybrid_retrieval(question, top_k)
        for item in out:
            item["strategy"] = "hybrid(降级)"
        return out

    # 召回阶段多取一些，给精排留出重排空间
    candidates = hybrid_retrieval(question, max(top_k * 3, 10))

    pairs = [(question, item["content"]) for item in candidates]
    scores = model.predict(pairs)

    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(order, start=1):
        item = dict(candidates[idx])
        item["rank"] = rank
        item["rerank_score"] = round(float(scores[idx]), 6)
        item["strategy"] = "hybrid+rerank"
        results.append(item)
    return results


# ---------------------------------------------------------------- 统一入口

STRATEGIES = {
    "vector": vector_retrieval,
    "bm25": bm25_retrieval,
    "hybrid": hybrid_retrieval,
    "hybrid_rerank": hybrid_rerank_retrieval,
}


def retrieve(question: str, top_k: int = 10, strategy: str = "vector") -> list[dict]:
    """统一检索入口。

    注意：线上 agent 不走这里，走 rag.debug_retrieval()（带租户过滤）。
    两条路径尚未统一，见文件顶部权重注释与 README「已知问题」。
    """
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(f"未知策略：{strategy}，可选 {list(STRATEGIES)}")
    return fn(question, top_k)
