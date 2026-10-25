"""在非 hold-out 题目上扫描 RRF 融合权重。

为什么要分 hold-out
-------------------
如果拿全部 50 条题来调权重，调出来的数字就是"背答案"的结果，
换一批题立刻打回原形。正确做法是：
    - 调参只用非 hold-out 的题（40 条）
    - hold-out 的 10 条只在最后验收时跑一次
这样报出来的指标才反映泛化能力。

注意：本脚本只做权重扫描，不改 retrieval.py 里的默认值。
      确定权重后请手工把 VECTOR_WEIGHT / BM25_WEIGHT 改成最优值。
"""

import retrieval
from eval_dataset import retrieval_cases

TOP_K = 10
RECALL_AT = 5

# (向量权重, BM25 权重)
GRID = [
    (1.0, 0.0),   # 纯向量，作为对照
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),   # 等权
]


def score(wv: float, wb: float, cases: list[dict]) -> dict:
    retrieval.VECTOR_WEIGHT = wv
    retrieval.BM25_WEIGHT = wb

    hits = hit1 = 0
    rr = 0.0

    for case in cases:
        results = retrieval.hybrid_retrieval(case["question"], TOP_K)
        rank = next(
            (r["rank"] for r in results if r["source"] in case["expected_sources"]),
            None,
        )
        if rank is not None and rank <= RECALL_AT:
            hits += 1
        if rank == 1:
            hit1 += 1
        if rank is not None:
            rr += 1.0 / rank

    n = len(cases)
    return {"recall": hits / n, "hit1": hit1 / n, "mrr": rr / n, "n": n}


def main():
    cases = retrieval_cases(include_holdout=False)
    print(f"调参集：非 hold-out {len(cases)} 条\n")
    print(f"{'向量:BM25':<14}{'Recall@5':>11}{'Hit@1':>10}{'MRR@10':>10}")
    print("-" * 46)

    rows = []
    for wv, wb in GRID:
        r = score(wv, wb, cases)
        rows.append((wv, wb, r))
        print(f"{wv:.1f}:{wb:.1f}{'':<8}{r['recall']:>10.2%} "
              f"{r['hit1']:>9.2%} {r['mrr']:>9.4f}")

    # 三个指标不一定会指向同一个配置，所以分别报出来，不替使用者做决定。
    # 之前这里只按 MRR 选"最优"，却把 Hit@1 一起打印出来，
    # 看起来像是"MRR 最优的那个配置 Hit@1 也最好"——实际可能更差。
    by_mrr = max(rows, key=lambda x: x[2]["mrr"])
    by_hit = max(rows, key=lambda x: x[2]["hit1"])
    by_recall = max(rows, key=lambda x: x[2]["recall"])

    def line(tag, pick):
        wv, wb, r = pick
        return (f"  {tag:<10} 向量 {wv:.1f} : BM25 {wb:.1f}   "
                f"Recall@5 {r['recall']:.2%}  Hit@1 {r['hit1']:.2%}  MRR {r['mrr']:.4f}")

    print("\n各指标的最优配置：")
    print(line("按 MRR", by_mrr))
    print(line("按 Hit@1", by_hit))
    print(line("按 Recall@5", by_recall))

    same = by_mrr[:2] == by_hit[:2] == by_recall[:2]
    print()
    if same:
        print("三个指标指向同一配置，可以放心改成这个值。")
    else:
        print("三个指标指向不同配置，需要按下游怎么用检索结果来决定：")
        print("  - 只取 top-1 给用户看 → 看 Hit@1")
        print("  - 把 top-k 全文喂给模型 → 看 Recall@5（漏召回比排序靠后更致命）")
        print("本项目是后者：top_k 篇文档全文进证据核查，漏掉就没法答。")


if __name__ == "__main__":
    main()
