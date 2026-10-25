"""端到端 Agent 评测：工具选择 + 答案正确性 + 任务成功率。

这个脚本回答的是简历上那句话：
「任务执行成功率」到底是多少。

和 evaluate_rag.py 的区别
-------------------------
- evaluate_rag.py  只测检索层（向量库找得准不准）
- 本脚本          测完整链路（Agent 有没有调对工具 + 最终答案对不对）

三个判定维度
------------
1. 工具选择   该调工具的调了、不该调的不调
2. 答案正确   答案里出现了期望的关键事实（数字、关键词）
3. 拒答正确   知识库没答案的题，必须明确说"无法确认"而不是编

三项全过才算这个任务成功。这就是「任务执行成功率」的口径。
"""

import re
import time
import uuid

from fastapi.testclient import TestClient

from api import _extract_text, app, remove_session_row
from eval_dataset import agent_cases, multiturn_cases, stats

# 默认期望调用的工具
DEFAULT_TOOLS = {"search_shop_info"}

# 个别题目的例外：这题既可以先查商品再算，也可以直接算
TOOL_OVERRIDES = {
    "M03": {"calculate_total", "search_shop_info"},
    "M05": {"calculate_total", "search_shop_info"},
}


def normalize(text: str) -> str:
    """去掉所有空白再比较。

    模型输出会经过 Markdown 渲染，"10元" 常常变成 "10 元"、
    "24小时" 变成 "24 小时"。如果直接做子串匹配，
    系统明明答对了也会被判成错 —— 这是评测脚本最常见的假阴性来源。
    """
    return re.sub(r"\s+", "", str(text)).lower()


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    haystack = normalize(answer)
    return any(normalize(k) in haystack for k in keywords)


def extract_tools(messages) -> list[str]:
    """从消息流里抽出所有被调用过的工具名。"""
    names = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                names.append(name)
    return names


def run_one(agent, case) -> dict:
    """跑单个任务，返回结构化结果。

    多轮题（带 history）会先把前几轮逐条回放，让 agent **真实响应**每一轮。
    不用"把历史拼成一条消息"的偷懒做法——那样 agent 从没回应过前几轮，
    测不出 checkpointer 是否真的在累积状态，也就测不出多轮能力。
    """
    tid = "eval_" + uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": tid}}

    # 逐轮回放，并记录每轮新增的消息边界
    all_tools: list[str] = []
    seen = 0
    for turn in case.get("history") or []:
        r = agent.invoke(
            {"messages": [{"role": "user", "content": turn}]}, config=config
        )
        all_tools.extend(extract_tools(r["messages"][seen:]))
        seen = len(r["messages"])

    start = time.perf_counter()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": case["question"]}]},
        config=config,
    )
    elapsed = time.perf_counter() - start

    messages = result["messages"][seen:]
    # 多轮题看整段对话调用过哪些工具：最后一句"那这个多少钱"完全可以
    # 直接用上一轮的结果回答，不该因为没有新工具调用就判错。
    all_tools.extend(extract_tools(messages))
    tools = all_tools
    answer = _extract_text(result["messages"][-1].content)

    # --- 判定 1：工具选择 ---
    if case["expect_refusal"]:
        # 负例题不强制调工具：调了但拒答也算对，不调直接拒答更省
        tool_ok = True
    else:
        expected_tools = TOOL_OVERRIDES.get(case["id"], DEFAULT_TOOLS)
        tool_ok = any(t in expected_tools for t in tools)

    # --- 判定 2：答案正确性 / 拒答正确性（忽略空白差异）---
    kw_hit = keyword_hit(answer, case["must_include"])

    # --- 判定 3：综合 ---
    success = tool_ok and kw_hit

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "thread_id": tid,
        "tools": tools,
        "answer": answer,
        "tool_ok": tool_ok,
        "keyword_hit": kw_hit,
        "success": success,
        "elapsed": elapsed,
    }


def evaluate(include_holdout: bool = True, multiturn: bool = False) -> dict:
    cases = (
        multiturn_cases(include_holdout=include_holdout)
        if multiturn
        else agent_cases(include_holdout=include_holdout)
    )

    print("=" * 68)
    print("端到端 Agent 评测 —— " + ("多轮对话" if multiturn else "单轮问答"))
    print("=" * 68)

    s = stats()
    print(f"数据集：{s['total']} 条，本次跑 {len(cases)} 条"
          f"（{'含' if include_holdout else '不含'} hold-out）")
    print("口径：工具选择正确 且 答案含关键事实 且 负例正确拒答 = 成功")
    print()

    results = []
    latencies = []

    # TestClient 会触发 FastAPI 的 lifespan，从而拿到初始化好的 agent
    with TestClient(app) as _client:
        agent = app.state.agent

        # 预热：第一次调用包含工具/模型的各种懒加载
        print("预热中...")
        t0 = time.perf_counter()
        agent.invoke(
            {"messages": [{"role": "user", "content": "你好"}]},
            config={"configurable": {"thread_id": "eval_warmup"}},
        )
        print(f"预热完成，耗时 {time.perf_counter() - t0:.1f}s\n")

        for i, case in enumerate(cases, 1):
            try:
                r = run_one(agent, case)
            except Exception as e:
                r = {
                    "id": case["id"], "type": case["type"], "question": case["question"],
                    "tools": [], "answer": f"<异常: {e}>", "tool_ok": False,
                    "keyword_hit": False, "success": False, "elapsed": 0.0,
                }

            results.append(r)
            latencies.append(r["elapsed"])

            mark = "✓" if r["success"] else "✗"
            reason = ""
            if not r["success"]:
                if not r["tool_ok"]:
                    reason = "工具选择错"
                elif not r["keyword_hit"]:
                    reason = "答案未命中关键事实"

            print(f"  {mark} [{r['id']}] {r['question'][:30]:<32} "
                  f"tools={','.join(r['tools']) or '-':<20} {r['elapsed']:5.1f}s {reason}")

            # 失败的打印答案片段，方便排查
            if not r["success"]:
                print(f"       答案: {r['answer'][:110].replace(chr(10), ' ')}")

        # 清理评测产生的会话记录。
        # 两处都要删：
        #   1) checkpoints 表（对话记忆）—— 否则下次启动会被 migrate_threads 重新导入
        #   2) sessions 表（侧边栏元信息）—— 否则会一直显示在用户的会话列表里
        cleaned = 0
        for tid in [r.get("thread_id") for r in results] + ["eval_warmup"]:
            if not tid:
                continue
            try:
                app.state.memory.delete_thread(tid)
            except Exception:
                pass
            try:
                remove_session_row(tid)
            except Exception:
                pass
            cleaned += 1
        print(f"\n已清理 {cleaned} 个评测会话（记忆 + 侧边栏）")

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    tool_ok_n = sum(1 for r in results if r["tool_ok"])
    kw_ok_n = sum(1 for r in results if r["keyword_hit"])

    print("\n" + "=" * 68)
    print("汇总")
    print("=" * 68)
    print(f"任务数            : {total}")
    print(f"工具选择正确      : {tool_ok_n}/{total} = {tool_ok_n / total:.1%}")
    print(f"答案命中关键事实  : {kw_ok_n}/{total} = {kw_ok_n / total:.1%}")
    print(f"端到端任务成功率  : {passed}/{total} = {passed / total:.1%}")
    print()
    print(f"平均耗时          : {sum(latencies) / total:.2f}s")
    print(f"最慢任务          : {max(latencies):.2f}s")

    # 按题型拆
    print("\n按题型拆分：")
    types = (["coreference", "slot_fill", "switch"] if multiturn
             else ["easy", "paraphrase", "multihop", "conflict", "temporal", "negative"])
    for t in types:
        sub = [r for r in results if r["type"] == t]
        if not sub:
            continue
        ok = sum(1 for r in sub if r["success"])
        print(f"  {t:<12} {ok}/{len(sub)} = {ok / len(sub):.0%}")

    fails = [r for r in results if not r["success"]]
    if fails:
        print(f"\n失败 {len(fails)} 条：")
        for r in fails:
            print(f"  [{r['id']}] {r['question']}")
            print(f"       tools={r['tools']}  tool_ok={r['tool_ok']}  keyword_ok={r['keyword_hit']}")

    return {
        "total": total,
        "passed": passed,
        "success_rate": passed / total if total else 0.0,
        "tool_accuracy": tool_ok_n / total if total else 0.0,
        "keyword_rate": kw_ok_n / total if total else 0.0,
        "avg_seconds": sum(latencies) / total if total else 0.0,
        "fails": fails,
    }


if __name__ == "__main__":
    single = evaluate(multiturn=False)
    print()
    multi = evaluate(multiturn=True)

    print("\n" + "=" * 68)
    print("总览")
    print("=" * 68)
    print(f"  单轮任务成功率 : {single['passed']}/{single['total']}"
          f" = {single['success_rate']:.0%}")
    print(f"  多轮任务成功率 : {multi['passed']}/{multi['total']}"
          f" = {multi['success_rate']:.0%}")
