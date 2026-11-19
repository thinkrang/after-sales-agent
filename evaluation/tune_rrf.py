"""按线上 top_k=3 扫描 RRF 权重；默认只使用 dev。

python -m evaluation.tune_rrf --output work/rrf-dev.json
python -m evaluation.tune_rrf --holdout --output work/rrf-with-holdout.json

选择规则预先固定为 dev R@3、再 MRR@10，平局取 GRID 中首项。
--holdout 只评估选定权重，不扫描留出集；历史上已看过的题不算未见数据。
不修改生产默认权重。JSON 保留题目 ID、索引/数据集指纹和逐题结果。
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import retrieval
from evaluation.eval_dataset import retrieval_cases
from evaluation.evaluate_rag import run_strategy
from evaluation.task_eval import provenance

GRID = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3),
        (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.0, 1.0)]


def score(wv: float, wb: float, cases: list[dict]) -> dict:
    original = retrieval.VECTOR_WEIGHT, retrieval.BM25_WEIGHT
    try:
        retrieval.VECTOR_WEIGHT, retrieval.BM25_WEIGHT = wv, wb
        result = run_strategy("hybrid", cases)
        return {"vector_weight": wv, "bm25_weight": wb, **result}
    finally:
        retrieval.VECTOR_WEIGHT, retrieval.BM25_WEIGHT = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dev = retrieval_cases(include_holdout=False)
    holdout = [c for c in retrieval_cases() if c["holdout"]]
    retrieval.retrieve("预热", top_k=3, strategy="hybrid")
    rows = [score(wv, wb, dev) for wv, wb in GRID]
    chosen = max(rows, key=lambda r: (r["recall3"], r["mrr"]))
    print(f"dev {len(dev)} 条；holdout {len(holdout)} 条（未参与选择）")
    print("向量:BM25       R@3      R@5     Hit@1    MRR@10")
    for r in rows:
        print(f"{r['vector_weight']:.1f}:{r['bm25_weight']:.1f}"
              f"       {r['recall3']:.2%}   {r['recall']:.2%}"
              f"   {r['hit1']:.2%}   {r['mrr']:.4f}")
    print(f"dev 选择：{chosen['vector_weight']}:{chosen['bm25_weight']}；生产配置未修改")
    ids, docs, metas = retrieval._corpus()
    corpus = sorted(zip(ids, docs, metas), key=lambda row: row[0])
    fingerprint = lambda value: hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "date": datetime.now(timezone.utc).isoformat(), "provenance": provenance(),
        "selection": "dev recall3, then mrr10, then GRID order",
        "production_top_k": 3, "diagnostic_top_k": 10, "tenant": "public",
        "dataset_sha256": fingerprint(retrieval_cases()),
        "corpus_sha256": fingerprint(corpus), "chunks": len(ids),
        "dev_ids": [c["id"] for c in dev], "holdout_ids": [c["id"] for c in holdout],
        "dev": rows, "selected": [chosen["vector_weight"], chosen["bm25_weight"]],
        "holdout": None,
    }
    if args.holdout:
        report["holdout"] = score(*report["selected"], holdout)
        report["holdout_baseline"] = score(1.0, 0.0, holdout)
        print("固定权重 holdout R@3:", f"{report['holdout']['recall3']:.2%}")
        print("固定纯向量 holdout R@3:", f"{report['holdout_baseline']['recall3']:.2%}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
