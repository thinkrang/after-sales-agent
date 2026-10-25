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

import os
import re
from functools import lru_cache

import jieba
from rank_bm25 import BM25Okapi

from rag import DEFAULT_TENANT, debug_retrieval, get_vector_store, visible_tenants

# RRF 常数，经验值 60
RRF_K = 60

# 混合召回时每一路取多少候选
CANDIDATES_PER_ROUTE = 10

# RRF 融合权重：0.6 : 0.4。
#
# 这个模块是**唯一**的检索入口：evaluate_rag / tune_rrf 和线上 agent 的
# search_shop_info 都走 retrieve()。租户过滤在策略内部完成，
# 所以调这里的权重会真实影响 agent 的回答。
#
# （2026-10-25 之前是两条路径：评测走 STRATEGIES，线上走
#  rag.debug_retrieval。当时按评测调出来的权重对线上毫无影响。
#  已统一，见 README。）
#
# 关键是**用哪个 k 来评**。agent 实际是 top_k=3，所以要看 R@3，
# 而不是评测脚本默认的 R@5。两者结论差很多：
#
#   向量:BM25      R@3      Hit@1     R@5      MRR
#   1.0:0.0     91.04%    80.60%   95.52%   0.8532   ← 纯向量
#   0.9:0.1     95.52%    77.61%   97.01%   0.8557
#   0.7:0.3     97.01%    79.10%   97.01%   0.8731
#   0.6:0.4     97.01%    79.10%   97.01%   0.8756   ← 选它
#   0.5:0.5     97.01%    79.10%   97.01%   0.8731
#   0.4:0.6     92.54%    83.58%   92.54%   0.8781   ← Hit@1 最高但 R@3 崩了
#
# 按 R@5 看只涨 1.5 个点，按 R@3 看涨 6 个点——**top_k 越小，漏召的代价越大**，
# 所以权重对生产的影响比对评测指标的影响更明显。
#
# 0.4:0.6 那行值得注意：Hit@1 反而最高（83.58%），但 R@3 掉到 92.54%。
# 说明 BM25 权重过高时会把个别对的文档顶到第 1，同时把另一些挤出前 3。
# 只看 Hit@1 会选错。
#
# 换语料后请重跑 tune_rrf.py，这个结论不是永久的。
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# 线上默认策略。可用环境变量覆盖，方便灰度对比不同策略的端到端效果。
DEFAULT_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid").strip() or "hybrid"

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


def vector_retrieval(
    question: str, top_k: int, tenant: str | None = None
) -> list[dict]:
    """纯向量检索。

    直接复用 rag.debug_retrieval —— 它是唯一直接访问 Chroma 的地方，
    负责租户过滤（检索前过滤，不是检索后筛）和索引锁。
    这里不重复实现，避免两条路径的过滤条件慢慢走偏。
    """
    return debug_retrieval(question=question, top_k=top_k, tenant=tenant)


def bm25_retrieval(
    question: str, top_k: int, tenant: str | None = None
) -> list[dict]:
    """纯关键词检索。

    租户过滤只能在打分之后做：BM25 的 IDF 是全局统计量，
    按租户重新分词会让同一句话在不同租户下得分不一致，很难解释。
    所以先对全库打分、再按可见租户筛，最后截 top_k。

    副作用要清楚：top_k 是在**筛选后**截断的，不是先截再筛，
    否则可见文档会被无权限文档挤出候选。
    """
    bm25, ids, docs, metas = _bm25()
    scores = bm25.get_scores(tokenize(question))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    allowed = set(visible_tenants(tenant))
    results = []
    for idx in order:
        if scores[idx] <= 0:
            # BM25 分数为 0 表示没有任何词命中，后面的只会更低
            break
        meta = metas[idx] or {}
        if (meta.get("tenant") or DEFAULT_TENANT) not in allowed:
            continue
        results.append({
            "rank": len(results) + 1,
            "id": ids[idx],
            "source": meta.get("source", "未知文档"),
            "tenant": meta.get("tenant") or DEFAULT_TENANT,
            "page": meta.get("page"),
            "score": round(float(scores[idx]), 6),
            "content": docs[idx],
        })
        if len(results) >= top_k:
            break
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


def hybrid_retrieval(
    question: str, top_k: int, tenant: str | None = None
) -> list[dict]:
    """向量 + BM25，加权 RRF 融合。

    两路各取 CANDIDATES_PER_ROUTE 个候选，融合后再截 top_k。

    租户过滤在两路内部各自完成（向量交给 Chroma 的 where，
    BM25 打分后按可见租户筛）。融合阶段只处理可见文档，
    不会出现"无权限文档占了融合名次"的情况。
    """
    n = max(top_k, CANDIDATES_PER_ROUTE)

    vec = vector_retrieval(question, n, tenant=tenant)
    kw = bm25_retrieval(question, n, tenant=tenant)

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


def hybrid_rerank_retrieval(
    question: str, top_k: int, tenant: str | None = None
) -> list[dict]:
    """混合召回 + CrossEncoder 精排。

    如果 reranker 不可用，自动降级为 hybrid，并在结果的 strategy 字段标注，
    避免"以为跑了 reranker 其实没跑"。
    """
    model = _reranker()
    if model is None:
        out = hybrid_retrieval(question, top_k, tenant=tenant)
        for item in out:
            item["strategy"] = "hybrid(降级)"
        return out

    # 召回阶段多取一些，给精排留出重排空间
    candidates = hybrid_retrieval(question, max(top_k * 3, 10), tenant=tenant)

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


def retrieve(
    question: str,
    top_k: int = 10,
    strategy: str | None = None,
    tenant: str | None = None,
) -> list[dict]:
    """统一检索入口 —— 评测脚本和线上 agent 都走这里。

    tenant 是**安全边界**，不是可选优化：它决定能在哪些文档里检索。
    以前线上走 rag.debug_retrieval（带 tenant）、评测走 STRATEGIES（不带），
    两条路径的策略和参数都不一样，评测结论没法代表线上效果。

    现在租户过滤下沉到各策略内部：
    - 向量：交给 Chroma 的 where 条件（检索前过滤）
    - BM25：打分后按可见租户筛（IDF 是全局统计量，不能按租户重算）
    """
    name = strategy or DEFAULT_STRATEGY
    fn = STRATEGIES.get(name)
    if fn is None:
        raise ValueError(f"未知策略：{name}，可选 {list(STRATEGIES)}")
    return fn(question, top_k, tenant=tenant)
