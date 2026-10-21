"""检索策略对比评测：向量 / BM25 / RRF 融合 / Reranker。

指标口径
--------
- Recall@5   正确文档是否出现在前 5 条（线上只把 top-5 喂给模型）
- Hit@1      正确文档是否排第 1
- MRR@10     正确文档排名的倒数和均值，衡量"排得够不够前"
- P50 / P95  单次检索延迟分位数（已剔除首次模型加载）

注意：当前知识库只有 20 个 chunk，top-5 覆盖 25% 的语料，
Recall@5 天然接近 100%，区分度有限。
真正能看出差异的是 Hit@1 和 MRR@10。
"""

import time

from eval_dataset import retrieval_cases, stats
from retrieval import STRATEGIES, reranker_ready

TOP_K = 10
RECALL_AT = 5

# 要对比的策略（按顺序）
# hybrid 不在默认列表里：tune_rrf.py 实测任何 BM25 权重都会降低 Hit@1，
# 当前语料下它没有收益空间（向量已 100% Recall@5）。
# 想看负结果可以把它加回来对比。
COMPARE = ["vector", "bm25", "hybrid_rerank"]


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
    hit1 = 0
    rr_sum = 0.0
    detail = []

    for case in cases:
        start = time.perf_counter()
        results = fn(case["question"], TOP_K)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        rank = None
        for item in results:
            if item["source"] in case["expected_sources"]:
                rank = item["rank"]
                break

        if rank is not None and rank <= RECALL_AT:
            hits += 1
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
    print(f"检索 k={TOP_K}，报告 Recall@{RECALL_AT} / Hit@1 / MRR@{TOP_K}")
    print()

    # ---------- 预热 ----------
    print("预热（首次调用含模型加载，不计入统计）...")
    for name in COMPARE:
        if name == "hybrid_rerank" and not reranker_ready():
            print(f"  {name:<14} 跳过：reranker 模型未就绪")
            continue
        t = time.perf_counter()
        STRATEGIES[name]("预热", TOP_K)
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
    print(f"{'策略':<16}{'Recall@5':>11}{'Hit@1':>10}{'MRR@10':>10}{'P50':>10}{'P95':>10}")
    print("-" * 72)

    base = reports[0] if reports else None
    for r in reports:
        line = (f"{r['name']:<16}"
                f"{r['recall']:>10.2%} "
                f"{r['hit1']:>9.2%} "
                f"{r['mrr']:>9.4f} "
                f"{r['p50']:>9.1f}ms"
                f"{r['p95']:>9.1f}ms")
        print(line)

    if "hybrid_rerank" not in [r["name"] for r in reports]:
        print(f"{'hybrid_rerank':<16}{'— 模型未就绪，需先下载 ' + 'BAAI/bge-reranker-base':<50}")

    # ---------- 提升幅度 ----------
    if len(reports) >= 2:
        best = max(reports, key=lambda r: r["mrr"])
        print()
        print(f"最优策略：{best['name']}")
        print(f"  Hit@1   {base['hit1']:.2%} → {best['hit1']:.2%} "
              f"({(best['hit1'] - base['hit1']) * 100:+.2f} 个百分点)")
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
