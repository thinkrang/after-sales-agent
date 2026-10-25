"""检索策略对比评测：向量 / BM25 / RRF 融合 / Reranker。

指标口径
--------
- Recall@3   正确文档是否出现在前 3 条 —— **这是线上真实口径**
- Recall@5   正确文档是否出现在前 5 条
- Hit@1      正确文档是否排第 1
- MRR@10     正确文档排名的倒数和均值，衡量"排得够不够前"
- P50 / P95  单次检索延迟分位数（已剔除首次模型加载）

为什么要单独盯 Recall@3
-----------------------
agent 的 search_shop_info 是 top_k=3（3 篇文档全文进证据核查）。
以前这里只报 Recall@5，导致调权重时看的是一个生产用不上的指标：
152 篇语料下 R@5 从 95.52% 涨到 97.01%（+1.5 点），
看着收益很小；但同一组权重下 R@3 是从 91.04% 涨到 97.01%（+6 点）。

**top_k 越小，漏召的代价越大**，所以 k 必须和线上一致。
R@5 保留是为了和旧数据可比，不作为决策依据。

租户口径
--------
TENANT = None 表示以"公共用户"视角评测（只看 public 文档）。
评测集不涉及租户专属文档，这样最干净。线上用户会额外看到自己租户的文档，
但那是权限边界，不是检索质量，混进来反而让指标不可比。
"""

import time

from eval_dataset import retrieval_cases, stats
from retrieval import STRATEGIES, reranker_ready

TOP_K = 10

# 线上真实口径：agent 的 search_shop_info 用 top_k=3
PRODUCTION_TOP_K = 3

# 保留旧口径用于和历史数据对比
RECALL_AT = 5

TENANT = None

# 要对比的策略（按顺序）
# hybrid 现在进默认列表了：语料扩到 152 篇之后，
# BM25 对「商品编号001」这类精确实体查询有真实收益（见 tune_rrf.py）。
COMPARE = ["vector", "bm25", "hybrid", "hybrid_rerank"]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p / 100.0
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (pos - low)


def run_strategy(name: str, cases: list[dict]) -> dict:
    fn = STRATEGIES[name]

    latencies = []
    hits = 0
    hits3 = 0
    hit1 = 0
    rr_sum = 0.0
    detail = []

    for case in cases:
        start = time.perf_counter()
        results = fn(case["question"], TOP_K, tenant=TENANT)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        rank = None
        for item in results:
            if item["source"] in case["expected_sources"]:
                rank = item["rank"]
                break

        if rank is not None and rank <= RECALL_AT:
            hits += 1
        if rank is not None and rank <= PRODUCTION_TOP_K:
            hits3 += 1
        if rank == 1:
            hit1 += 1
        if rank is not None:
            rr_sum += 1.0 / rank

        detail.append({
            "id": case["id"],
            "type": case["type"],
            "question": case["question"],
            "rank": rank,
        })

    total = len(cases)
    return {
        "name": name,
        "total": total,
        "recall3": hits3 / total,
        "recall": hits / total,
        "hit1": hit1 / total,
        "mrr": rr_sum / total,
        "p50": percentile(latencies, 50) * 1000,
        "p95": percentile(latencies, 95) * 1000,
        "detail": detail,
    }


def main():
    cases = retrieval_cases()
    s = stats()

    print("=" * 72)
    print("检索策略对比")
    print("=" * 72)
    print(f"知识库规模：见下方 chunk 数；测试集 {s['total']} 条，"
          f"参与检索 {len(cases)} 条")
    print(f"检索 k={TOP_K}，报告 R@{PRODUCTION_TOP_K}（线上口径）/ "
          f"R@{RECALL_AT} / Hit@1 / MRR@{TOP_K}")
    print(f"租户视角：{TENANT or 'public（只看公共文档）'}")
    print()

    # ---------- 预热 ----------
    print("预热（首次调用含模型加载，不计入统计）...")
    for name in COMPARE:
        if name == "hybrid_rerank" and not reranker_ready():
            print(f"  {name:<14} 跳过：reranker 模型未就绪")
            continue
        t = time.perf_counter()
        STRATEGIES[name]("预热", TOP_K, tenant=TENANT)
        print(f"  {name:<14} {time.perf_counter() - t:.2f}s")
    print()

    # ---------- 逐策略评测 ----------
    reports = []
    for name in COMPARE:
        if name == "hybrid_rerank" and not reranker_ready():
            continue
        r = run_strategy(name, cases)
        reports.append(r)

    # ---------- 对比表 ----------
    print("=" * 72)
    print("对比结果")
    print("=" * 72)
    print(f"{'策略':<16}{'R@3*':>9}{'R@5':>9}{'Hit@1':>9}{'MRR@10':>9}"
          f"{'P50':>9}{'P95':>9}")
    print("-" * 72)
    print("* R@3 是线上口径（agent 用 top_k=3），决策以它为准")

    base = reports[0] if reports else None
    for r in reports:
        line = (f"{r['name']:<16}"
                f"{r['recall3']:>8.2%} "
                f"{r['recall']:>8.2%} "
                f"{r['hit1']:>8.2%} "
                f"{r['mrr']:>8.4f} "
                f"{r['p50']:>8.1f}ms"
                f"{r['p95']:>8.1f}ms")
        print(line)

    if "hybrid_rerank" not in [r["name"] for r in reports]:
        print(f"{'hybrid_rerank':<16}{'— 模型未就绪，需先下载 ' + 'BAAI/bge-reranker-base':<50}")

    # ---------- 提升幅度 ----------
    if len(reports) >= 2:
        # 按 R@3 选：agent 只拿前 3 篇做证据核查，漏出前 3 就等于答不了。
        # 以前按 MRR 选，MRR 衡量的是"整体排得好不好"，
        # 对 top-3 这个硬边界不敏感。
        best = max(reports, key=lambda r: (r["recall3"], r["mrr"]))
        print()
        print(f"按线上口径（R@3）最优策略：{best['name']}")
        for key, label, fmt in [
            ("recall3", "R@3  ", "{:.2%}"),
            ("recall", "R@5  ", "{:.2%}"),
            ("hit1", "Hit@1", "{:.2%}"),
        ]:
            a, b = base[key], best[key]
            print(f"  {label}  {fmt.format(a)} → {fmt.format(b)} "
                  f"({(b - a) * 100:+.2f} 个百分点)")
        print(f"  MRR@10  {base['mrr']:.4f} → {best['mrr']:.4f} "
              f"({best['mrr'] - base['mrr']:+.4f})")

    # ---------- 明细：排名靠后的题 ----------
    for r in reports:
        weak = [d for d in r["detail"] if d["rank"] is None or d["rank"] > 1]
        if not weak:
            continue
        print(f"\n[{r['name']}] 非头名命中 {len(weak)} 条：")
        for d in weak:
            rank_text = f"rank {d['rank']}" if d["rank"] else "未召回"
            print(f"  [{d['id']}] {d['question'][:38]:<40} {rank_text}")

    return reports


if __name__ == "__main__":
    main()
