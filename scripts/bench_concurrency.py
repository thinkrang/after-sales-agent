"""并发压测：量化全局锁对吞吐的影响。

为什么用 stub agent 而不是真实 LLM
----------------------------------
真实 LLM 每次 3 秒左右，且延迟波动大（P95 3206ms）。用它压测有两个问题：
  1. 不可复现——同一份代码跑两次结果可能差 50%
  2. 混淆变量——分不清"慢"是模型慢还是锁排队

所以这里换成一个固定 sleep 的假 agent：延迟恒定，唯一的变量就是锁。
这样测出来的差异，可以确定是并发控制导致的。

用法
----
    python scripts/bench_concurrency.py --n 20 --delay 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


class SlowStubAgent:
    """模拟真实 agent：固定延迟，接口与 LangGraph 编译后的图一致。"""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.concurrent_peak = 0
        self._active = 0

    def invoke(self, payload, config=None, **kwargs):
        # 记录"同时在跑的请求数"，这是判断有没有被串行化的直接证据
        self._active += 1
        self.concurrent_peak = max(self.concurrent_peak, self._active)
        try:
            time.sleep(self.delay)
            return {"messages": [SimpleNamespace(content="好的，已收到。")]}
        finally:
            self._active -= 1

    def stream(self, payload, config=None, **kwargs):  # 压测只打 /chat，这里留个占位
        yield SimpleNamespace(content=""), {"langgraph_node": "model"}


async def fire(
    client: httpx.AsyncClient,
    n: int,
    session_ids: list[str],
) -> list[float]:
    """并发打 n 个请求，返回每个请求的耗时。"""

    async def one(i: int) -> float:
        t0 = time.perf_counter()
        r = await client.post(
            "/chat",
            json={"thread_id": session_ids[i], "message": "你好"},
        )
        r.raise_for_status()
        return time.perf_counter() - t0

    return await asyncio.gather(*(one(i) for i in range(n)))


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[idx]


async def main(n: int, delay: float) -> None:
    from api import app

    async with app.router.lifespan_context(app):
        stub = SlowStubAgent(delay)
        app.state.agent = stub          # 换掉真实 agent，只测并发控制

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=120) as client:
            login = await client.post(
                "/auth/login",
                json={"username": "acme-manager", "password": "demo1234"},
            )
            login.raise_for_status()
            client.headers["Authorization"] = (
                f"Bearer {login.json()['access_token']}"
            )
            session_ids = []
            for _ in range(n + 1):
                created = await client.post("/sessions", json={})
                created.raise_for_status()
                session_ids.append(created.json()["id"])

            # 预热，避开首次请求的线程池/连接建立开销
            await fire(client, 1, session_ids[:1])

            t0 = time.perf_counter()
            latencies = await fire(client, n, session_ids[1:])
            wall = time.perf_counter() - t0

        print()
        print(f"  并发请求数      : {n}")
        print(f"  单请求模拟耗时  : {delay:.2f}s")
        print(f"  总墙钟时间      : {wall:.2f}s")
        print(f"  吞吐            : {n / wall:.2f} req/s  ({n / wall * 60:.0f} req/min)")
        print(f"  P50 延迟        : {pct(latencies, 0.50):.2f}s")
        print(f"  P95 延迟        : {pct(latencies, 0.95):.2f}s")
        print(f"  agent 并发峰值  : {stub.concurrent_peak}")
        print()
        print(f"  理论串行时间    : {n * delay:.2f}s（如果被完全串行化）")
        print(f"  理论并发时间    : {delay:.2f}s（如果完全并行）")
        speedup = (n * delay) / wall
        print(f"  实际并行度      : {speedup:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="并发请求数")
    ap.add_argument("--delay", type=float, default=0.8, help="stub agent 单次耗时（秒）")
    args = ap.parse_args()
    asyncio.run(main(args.n, args.delay))
