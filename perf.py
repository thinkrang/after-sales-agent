"""轻量性能追踪。

为什么不用 contextvars 做请求级关联
-----------------------------------
`_stream_agent` 是同步生成器，StreamingResponse 用线程池逐次 next() 迭代它，
每次 next() 可能落在不同线程，contextvars 的写入不会传播回父上下文。
做请求级 trace 需要显式传对象，会把签名搞得很脏。

这里退一步：**只做分阶段全局采样**，统计每个阶段的 P50/P95/次数。
对"耗时瓶颈在哪"这个问题，这个粒度已经足够，而且实现简单、不会出错。

用法
----
    with perf.stage("retrieval"):
        do_something()

    perf.record("llm_ttft", 320.5)     # 手动记一个点
    print(perf.snapshot())             # {"retrieval": {"p50": .., "p95": .., "n": ..}}
"""

import statistics
import threading
import time
from contextlib import contextmanager

# 每个阶段保留最近 N 条样本，避免长跑后内存无限增长
MAX_SAMPLES = 2000

_samples: dict[str, list[float]] = {}
_lock = threading.Lock()


def record(stage: str, ms: float) -> None:
    with _lock:
        bucket = _samples.setdefault(stage, [])
        bucket.append(float(ms))
        if len(bucket) > MAX_SAMPLES:
            del bucket[: len(bucket) - MAX_SAMPLES]


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - start) * 1000)


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


def snapshot() -> dict[str, dict]:
    """返回每个阶段的统计。"""
    with _lock:
        data = {k: list(v) for k, v in _samples.items()}

    out = {}
    for name, values in data.items():
        if not values:
            continue
        out[name] = {
            "n": len(values),
            "p50": round(percentile(values, 50), 2),
            "p95": round(percentile(values, 95), 2),
            "max": round(max(values), 2),
            "mean": round(statistics.fmean(values), 2),
        }
    return out


def reset() -> None:
    with _lock:
        _samples.clear()


def format_table() -> str:
    """把快照渲染成对齐的表格，方便直接 print 或写日志。"""
    snap = snapshot()
    if not snap:
        return "（还没有采样数据）"

    rows = sorted(snap.items(), key=lambda kv: kv[1]["p95"], reverse=True)
    lines = [
        f"{'阶段':<22}{'次数':>6}{'P50':>10}{'P95':>10}{'max':>10}",
        "-" * 58,
    ]
    for name, s in rows:
        lines.append(
            f"{name:<22}{s['n']:>6}{s['p50']:>9.1f}ms{s['p95']:>8.1f}ms{s['max']:>8.1f}ms"
        )
    return "\n".join(lines)
