"""评测集与多轮评测逻辑的测试。

守三件事：
    1. 多轮题不能污染单轮指标（retrieval / 单轮 agent 成功率）
    2. 多轮题的数据结构完整（history 非空、有判定依据）
    3. run_one 真的按顺序回放了 history，而不是把历史拼成一条消息
"""

from types import SimpleNamespace

import pytest

import eval_dataset as ds


# ---------------------------------------------------------------- 数据集分组


def test_multiturn_cases_exist():
    cases = ds.multiturn_cases()
    assert len(cases) == 12
    types = {c["type"] for c in cases}
    assert types == {"coreference", "slot_fill", "switch"}
    for t in types:
        assert sum(1 for c in cases if c["type"] == t) == 4, f"{t} 应该有 4 条"


def test_multiturn_cases_are_excluded_from_single_turn():
    """agent_cases() 默认不含多轮题——否则「任务成功率」这个数字不可比。"""
    single = ds.agent_cases()
    assert len(single) == 80
    assert all("history" not in c for c in single)

    both = ds.agent_cases(include_multiturn=True)
    assert len(both) == 92


def test_multiturn_cases_have_no_expected_sources():
    """多轮题的 expected_sources 必须为空。

    单轮检索评测用的是 question 单独一句，"那这个多少钱"脱离上下文
    无法检索。混进 retrieval_cases 会直接拉低 Hit@1 / MRR。
    """
    for c in ds.multiturn_cases():
        assert c["expected_sources"] == [], f"{c['id']} 会污染检索指标"


def test_retrieval_cases_exclude_no_source_cases():
    """检索评测只收有 expected_sources 的题。

    多轮题（expected_sources 为空）和 N03/N13 这类无来源负例题都不参与。
    """
    assert len(ds.retrieval_cases()) == 78


def test_every_case_has_judgement_fields():
    for c in ds.DATASET:
        assert c["id"] and c["type"] and c["question"]
        assert "expect_answerable" in c
        assert "holdout" in c
        if not c["expect_refusal"]:
            assert c["must_include"], f"{c['id']} 没有判定依据，永远判不对"


def test_multiturn_history_is_non_empty():
    for c in ds.multiturn_cases():
        assert c["history"], f"{c['id']} 的 history 是空的，等于单轮题"
        assert all(isinstance(t, str) and t.strip() for t in c["history"])


def test_case_ids_are_unique():
    ids = [c["id"] for c in ds.DATASET]
    assert len(ids) == len(set(ids)), "有重复的用例 id"


def test_every_expected_source_is_versioned_in_repo():
    """A fresh clone must be able to reproduce the evaluation corpus.

    Expected sources previously lived under ignored uploads/, so README
    metrics could not be reproduced from GitHub.
    """
    from pathlib import Path

    available = {
        path.name
        for suffix in ("*.txt", "*.md")
        for path in Path("knowledge").glob(suffix)
    }
    expected = {
        source
        for case in ds.DATASET
        for source in case.get("expected_sources", [])
    }
    assert expected <= available, f"评测语料缺失：{sorted(expected - available)}"

# ---------------------------------------------------------------- 回放逻辑


class RecordingAgent:
    """假 agent：记录每次收到的用户消息，并按轮累积消息列表。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._messages: list = []

    def invoke(self, payload, config=None, **kwargs):
        text = payload["messages"][0]["content"]
        self.calls.append(text)
        self._messages = self._messages + [
            SimpleNamespace(content=text, tool_calls=None),
            SimpleNamespace(content="好的", tool_calls=None),
        ]
        return {"messages": list(self._messages)}


def test_run_one_replays_history_in_order():
    """history 要逐条回放，最后才是 question。

    这是多轮评测的核心：如果偷懒把历史拼成一条消息，agent 从没回应过
    前几轮，就测不出 checkpointer 是否真的在累积状态。
    """
    from evaluate_agent import run_one

    case = {
        "id": "X1", "type": "coreference",
        "history": ["第一轮", "第二轮"],
        "question": "第三轮问题",
        "expected_sources": [], "expect_answerable": True,
        "must_include": ["好的"], "expect_refusal": False, "holdout": False,
    }
    agent = RecordingAgent()
    r = run_one(agent, case)

    assert agent.calls == ["第一轮", "第二轮", "第三轮问题"], "回放顺序不对"
    assert r["keyword_hit"] is True, "最后一句的回答应该被判为命中"
    # 注意不断言 success：这个假 agent 不调任何工具，tool_ok 必然 False


def test_run_one_single_turn_makes_one_call():
    from evaluate_agent import run_one

    case = {
        "id": "X2", "type": "easy", "question": "单轮问题",
        "expected_sources": ["a.txt"], "expect_answerable": True,
        "must_include": ["好的"], "expect_refusal": False, "holdout": False,
    }
    agent = RecordingAgent()
    run_one(agent, case)
    assert agent.calls == ["单轮问题"]


def test_run_one_collects_tools_across_turns():
    """多轮题的工具判定看整段对话，不只看最后一句。

    最后一句"那这个多少钱"完全可以直接用上一轮检索到的结果回答，
    不该因为没有新的工具调用就判错。
    """
    from evaluate_agent import run_one

    class ToolAgent(RecordingAgent):
        def invoke(self, payload, config=None, **kwargs):
            text = payload["messages"][0]["content"]
            self.calls.append(text)
            call = {"name": "search_shop_info"}
            self._messages = self._messages + [
                SimpleNamespace(content=text, tool_calls=[call]),
                SimpleNamespace(content="好的", tool_calls=None),
            ]
            return {"messages": list(self._messages)}

    case = {
        "id": "X3", "type": "coreference",
        "history": ["商品003多少钱"],
        "question": "那这个库存呢",
        "expected_sources": [], "expect_answerable": True,
        "must_include": ["好的"], "expect_refusal": False, "holdout": False,
    }
    r = run_one(ToolAgent(), case)
    assert "search_shop_info" in r["tools"]
    assert r["tool_ok"] is True
