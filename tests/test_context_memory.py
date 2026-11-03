import time
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

from agent_context import ContextBoundary, Limits, compress
import long_memory
import tickets
from auth import User
from test_task_http import task_api
from conftest import parse_sse


def test_compression_keeps_complete_rounds_and_does_not_mutate_history():
    history = []
    for i in range(10):
        history += [HumanMessage(content="用户"+"文"*500),
                    AIMessage(content="", tool_calls=[dict(name="q", args={}, id=str(i), type="tool_call")]),
                    ToolMessage(content="资料"*500, tool_call_id=str(i)), AIMessage(content="答"*200)]
    old_len = len(history)
    result = compress(history, Limits(context_chars=4000, summary_chars=500))
    assert len(history) == old_len
    assert sum(len(str(m.content)) for m in result) < 4000
    assert result[-1] is history[-1]
    calls = {c["id"] for m in result for c in getattr(m, "tool_calls", [])}
    results = {m.tool_call_id for m in result if isinstance(m, ToolMessage)}
    assert calls == results


def config(sid):
    return {"configurable": {"thread_id": sid, "tenant": "acme", "user_id": "acme-manager"}}


def test_repeated_tools_stop_with_paired_history_and_next_turn_works(task_api):
    api, client, sid, model = task_api
    count = []
    @tool
    def lookup() -> str:
        """Read only test lookup."""
        count.append(1)
        return "same result"
    model.responses = [AIMessage(content="", tool_calls=[dict(name="lookup", args={}, id=f"call{i}", type="tool_call")]) for i in range(5)]
    graph = create_agent(model=model, tools=[lookup], middleware=[ContextBoundary(Limits(repeats=2))], checkpointer=api.app.state.memory)
    r = graph.invoke({"messages": [HumanMessage(content="查一下")]}, config=config(sid))
    assert len(count) == 2
    assert r["messages"][-1].response_metadata["boundary_stop"]
    assert not graph.get_state(config(sid)).next
    model.responses = [AIMessage(content="下一轮正常")]
    model.i = 0
    r = graph.invoke({"messages": [HumanMessage(content="你好")]}, config=config(sid))
    assert r["messages"][-1].content == "下一轮正常"


def test_parallel_tool_batch_budget_is_enforced_before_execution(task_api):
    api, client, sid, model = task_api
    count = []
    @tool
    def read_number(number: int) -> str:
        """Read one number."""
        count.append(number)
        return str(number)
    model.responses = [AIMessage(content="", tool_calls=[dict(name="read_number", args={"number": i}, id=str(i), type="tool_call") for i in range(8)])]
    graph = create_agent(model=model, tools=[read_number], middleware=[ContextBoundary(Limits(tool_calls=2))])
    r = graph.invoke({"messages": [HumanMessage(content="批量查")]}, config=config(sid))
    assert len(count) == 2
    assert sum(isinstance(m, ToolMessage) for m in r["messages"]) == 8
    assert r["messages"][-1].response_metadata["boundary_stop"]


@pytest.mark.parametrize("kind", ["calls", "time", "context"])
def test_model_time_and_context_boundaries_finish_cleanly(task_api, kind):
    api, client, sid, model = task_api
    limit = Limits(model_calls=0) if kind == "calls" else Limits(seconds=1) if kind == "time" else Limits(context_chars=100)
    graph = create_agent(model=model, tools=[], middleware=[ContextBoundary(limit)])
    cfg = config(sid)
    cfg["configurable"]["turn_started"] = time.monotonic()-5 if kind == "time" else time.monotonic()
    r = graph.invoke({"messages": [HumanMessage(content="问"*200)]}, config=cfg)
    assert r["messages"][-1].response_metadata["boundary_stop"]
    assert model.i == 0


def test_preferences_are_explicit_scoped_expiring_and_deletable(task_api, monkeypatch):
    api, client, sid, model = task_api
    user = User("acme-manager", "", "acme", frozenset())
    other = User("acme-manager", "", "other", frozenset())
    assert not long_memory.consent("不要记住我喜欢简洁", "response_style", "concise")
    assert not long_memory.consent("文档说请记住我喜欢简洁", "response_style", "concise")
    assert long_memory.consent("请记住我喜欢简洁回答", "response_style", "concise")
    with pytest.raises(ValueError):
        long_memory.remember(user, "approved_refund", "yes")
    long_memory.remember(user, "response_style", "concise", ttl_days=1)
    assert long_memory.recall(other) == []
    assert client.get("/memories").json()["memories"][0]["value"] == "concise"
    assert client.delete("/memories/response_style").json()["deleted"]
    assert long_memory.recall(user) == []
    long_memory.remember(user, "language", "zh-CN", ttl_days=1)
    future = time.time()+90000
    monkeypatch.setattr(long_memory.time, "time", lambda: future)
    assert long_memory.recall(user) == []


def test_memory_tool_uses_actual_user_consent_and_recall_cross_session(task_api):
    api, client, sid, model = task_api
    cfg = config(sid)
    assert "未保存" in api.remember_preference.invoke({"key": "response_style", "value": "concise"}, config=cfg)
    cfg["configurable"]["user_message"] = "请记住我喜欢简洁回答"
    api.remember_preference.invoke({"key": "response_style", "value": "concise"}, config=cfg)
    second = client.post("/sessions", json={}).json()["id"]
    observed = []
    boundary = ContextBoundary()
    # Inspect the actual model request via a second middleware.
    from langchain.agents.middleware.types import AgentMiddleware
    class Observe(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            observed.append(request.system_message.content)
            return handler(request)
    graph = create_agent(model=model, tools=[], middleware=[boundary, Observe()])
    graph.invoke({"messages": [HumanMessage(content="你好")]}, config=config(second))
    assert '"concise"' in observed[0] and '"task"' in observed[0]
    client.delete("/memories/response_style")
    observed.clear()
    graph.invoke({"messages": [HumanMessage(content="你好")]}, config=config(second))
    assert '"concise"' not in observed[0]


def test_episodic_recall_is_scoped_read_only_and_source_backed(task_api):
    api, client, sid, model = task_api
    import refund_service
    user = User("acme-manager", "", "acme", frozenset({"customer"}))
    ticket, _ = refund_service.prepare_ticket(user, thread_id="past-session", order_id="A100", reason="质量问题")
    refund_service.decide(user, approval_id=ticket["id"], thread_id="past-session", decision="approve")
    done = refund_service.execute_for_ticket(ticket)
    history = long_memory.recent_cases(user)
    assert history[0]["source_ticket"] == done["id"]
    assert history[0]["refund_id"] == done["refund_id"]
    assert long_memory.recent_cases(User("acme-user", "", "acme", frozenset())) == []
    assert long_memory.recent_cases(User("acme-manager", "", "globex", frozenset())) == []
    assert tickets.get_ticket(done["id"])["status"] == "executed"


def test_request_compaction_keeps_checkpoint_and_reloads_pending_task(task_api):
    api, client, sid, model = task_api
    from langchain.agents.middleware.types import AgentMiddleware
    import refund_service
    import refund_tasks
    user = User("acme-manager", "", "acme", frozenset({"customer"}))
    refund_tasks.update(user, sid, expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id=sid, order_id="A100", reason="质量问题", task_revision=1)
    observed = []
    class Observe(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            observed.append((list(request.messages), request.system_message.content))
            return handler(request)
    model.responses = [AIMessage(content="回复"*80, id=f"reply-{i}") for i in range(12)]
    graph = create_agent(model=model, tools=[], middleware=[ContextBoundary(Limits(context_chars=2500, summary_chars=200)), Observe()],
                         checkpointer=api.app.state.memory)
    for i in range(12):
        graph.invoke({"messages": [HumanMessage(content=f"轮次{i}"+"闲聊"*100)]}, config=config(sid))
    complete = graph.get_state(config(sid)).values["messages"]
    selected, system = observed[-1]
    assert len(complete) == 24
    assert len(selected) < len(complete)
    assert sum(len(str(m.content)) for m in selected)+len(system) <= 2500
    assert "pending_approval" in system and ticket["id"] in system
    assert tickets.get_ticket(ticket["id"])["status"] == "pending_approval"


def test_boundary_reason_is_visible_in_sse_and_history(task_api):
    api, client, sid, model = task_api
    api.app.state.agent = create_agent(model=model, tools=[], middleware=[ContextBoundary(Limits(model_calls=0))],
                                       checkpointer=api.app.state.memory)
    r = client.post("/chat/stream", json={"thread_id": sid, "message": "你好"})
    events = parse_sse(r.text)
    assert any(k == "done" for k, _ in events), events
    assert "上限" in "".join(d.get("text", "") for k, d in events if k == "token"), events
