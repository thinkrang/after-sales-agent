"""性能报告：跑一批请求，输出各阶段耗时分布并给出瓶颈判断。

回答的是面试问题："你这个系统为什么慢？慢在哪一步？"

用法
----
    python perf_report.py            # 默认 8 次检索 + 3 次完整对话
    python perf_report.py 20 5       # 自定义次数
"""

import sys

import perf
from api import app, search_shop_info
from eval_dataset import retrieval_cases

QUESTIONS = [
    "退货政策是什么",
    "付款后多久发货",
    "休闲卫衣多少钱",
    "100积分能换什么",
    "生日有什么福利",
    "安全培训要多少小时",
    "超7天还能退货吗",
    "怎么查物流",
]


def warm_up():
    """首次调用要加载 embedding 模型，必须排除在统计之外。"""
    print("预热中（加载 embedding 模型）...")
    search_shop_info.invoke({"question": "预热"})
    print("预热完成\n")


def bench_retrieval(n: int):
    print(f"[1/2] 检索 + 证据核查 × {n} 次")
    for q in QUESTIONS[:n]:
        search_shop_info.invoke({"question": q})


def bench_agent(n: int):
    from fastapi.testclient import TestClient

    print(f"\n[2/2] 完整 Agent 对话 × {n} 次（会真实调用大模型）")
    tool_counts = []
    with TestClient(app) as client:
        login = client.post(
            "/auth/login",
            json={"username": "acme-manager", "password": "demo1234"},
        )
        login.raise_for_status()
        client.headers["Authorization"] = (
            f"Bearer {login.json()['access_token']}"
        )
        for i, q in enumerate(QUESTIONS[:n], 1):
            created = client.post("/sessions", json={"title": f"perf_{i}"})
            created.raise_for_status()
            thread_id = created.json()["id"]
            r = client.post(
                "/chat/stream",
                json={"thread_id": thread_id, "message": q},
            )
            if r.status_code != 200:
                print(f"  第 {i} 次失败: HTTP {r.status_code}")
            else:
                # 工具调用次数从 SSE 事件里数，不塞进 perf（perf 只记耗时）
                n_tools = r.text.count("event: tool")
                tool_counts.append(n_tools)
                print(f"  第 {i} 次完成，工具调用 {n_tools} 次")
            client.delete(f"/sessions/{thread_id}")
    return tool_counts


def analyse(snap: dict, tool_counts: list[int] | None = None):
    """给一句人话结论，而不是只丢一堆数字。"""
    print("\n" + "=" * 58)
    print("瓶颈分析")
    print("=" * 58)

    retrieval = snap.get("rag.retrieval", {})
    evidence = snap.get("rag.evidence_check", {})
    ttft = snap.get("agent.ttft", {})
    total = snap.get("agent.stream_total", {})

    if retrieval and evidence:
        r95 = retrieval["p95"]
        e95 = evidence["p95"]
        print(f"  向量检索 P95      {r95:>8.1f} ms")
        print(f"  证据核查 P95      {e95:>8.1f} ms   （含一次 LLM 调用）")
        if e95 > r95 * 10:
            print(f"  → 核查比检索慢 {e95 / max(r95, 0.01):.0f} 倍。"
                  f"慢在模型调用，不在向量库。")
            print("    要压缩这一环，只能换更小的模型、缓存判断结果，或合并进回答的 prompt。")
        else:
            print("  → 两者量级接近，检索不是瓶颈。")

    if ttft:
        print(f"\n  首字延迟 TTFT P50  {ttft['p50']:>8.1f} ms")
        print(f"  首字延迟 TTFT P95  {ttft['p95']:>8.1f} ms")
        # 注意：TTFT 里包含了工具调用（检索 + 证据核查），
        # 因为模型要先拿到工具结果才能开始生成正文。
        if total and ttft["p50"] >= total["p50"] * 0.5:
            print("  → TTFT 占总耗时一半以上：首字前必须先跑完工具调用。")
            print("    这是「先核查再回答」的固有代价，不是流式没生效。")
        else:
            print("  → TTFT 明显小于总耗时，流式输出有效，用户不会干等。")

    if total:
        print(f"\n  整轮耗时 P50      {total['p50']:>8.1f} ms")
        print(f"  整轮耗时 P95      {total['p95']:>8.1f} ms")
        if ttft and total["p50"] > ttft["p50"] * 2:
            print("  → 总耗时主要由 token 生成速度决定，不是首包延迟。")

    tools = snap.get("agent.tool_calls")
    if tool_counts:
        avg = sum(tool_counts) / len(tool_counts)
        print(f"\n  单轮工具调用次数   平均 {avg:.1f} 次（{'/'.join(str(c) for c in tool_counts)}）")
        if max(tool_counts) >= 2:
            print("  → 存在一次对话内多次调用工具的情况，每次都要多一轮模型往返，是优化点。")


def main():
    n_retrieval = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_agent = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print("=" * 58)
    print("性能报告")
    print("=" * 58)
    print(f"知识库：{len(retrieval_cases())} 条测试题对应的索引规模")
    print()

    warm_up()

    perf.reset()          # 清掉预热产生的样本
    bench_retrieval(n_retrieval)
    tool_counts = bench_agent(n_agent)

    print("\n" + "=" * 58)
    print("分阶段耗时（按 P95 降序）")
    print("=" * 58)
    print(perf.format_table())

    analyse(perf.snapshot(), tool_counts)


if __name__ == "__main__":
    main()
