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

    best = None
    for wv, wb in GRID:
        r = score(wv, wb, cases)
        print(f"{wv:.1f}:{wb:.1f}{'':<8}{r['recall']:>10.2%} "
              f"{r['hit1']:>9.2%} {r['mrr']:>9.4f}")
        if best is None or r["mrr"] > best[2]["mrr"]:
            best = (wv, wb, r)

    print(f"\n最优：向量 {best[0]:.1f} : BM25 {best[1]:.1f}"
          f"（Hit@1 {best[2]['hit1']:.2%}，MRR@10 {best[2]['mrr']:.4f}）")
    print("请把 retrieval.py 里的 VECTOR_WEIGHT / BM25_WEIGHT 改成这个值。")


if __name__ == "__main__":
    main()
