"""Conversation remains runnable during a real durable refund interrupt."""
import pytest

import db
import orders
import refund_flow
import tickets
from conftest import parse_sse
from test_task_http import task_api, script, start_task


@pytest.fixture
def detached_api(task_api):
    from langchain.agents import create_agent
    api, client, sid, model = task_api
    api.app.state.refund_graph = refund_flow.build(api.app.state.memory)
    api.app.state.agent = create_agent(
        model=model, tools=[api.get_refund_task, api.update_refund_task, api.submit_refund_request],
        checkpointer=api.app.state.memory,
    )
    return api, client, sid, model


def say(client, sid, message):
    r = client.post("/chat/stream", json={"thread_id": sid, "message": message})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert not any(k == "error" for k, _ in events), events
    return events


def submit(env):
    api, client, sid, model = env
    start_task(client, sid)
    script(model, ("submit_refund_request", dict(order_id="A100", reason="质量问题", task_revision=1)))
    events = say(client, sid, "请提交")
    action = next(d["actions"][0] for k, d in events if k == "interrupt")
    assert action["protocol"] == "detached"
    assert not api.app.state.agent.get_state({"configurable": {"thread_id": sid}}).next
    assert api.app.state.refund_graph.get_state(refund_flow.config(action["approval_id"])).next
    return action


def test_side_question_preserves_pending_approval(detached_api):
    api, client, sid, model = detached_api
    old = submit(detached_api)
    script(model, ("get_refund_task", {}))
    events = say(client, sid, "先问一下退款现在什么状态？")
    assert any(k == "tool" for k, _ in events), "新消息必须真实进入对话图"
    new = next(d["actions"][0] for k, d in events if k == "interrupt")
    assert old["approval_id"] == new["approval_id"]
    assert client.get(f"/sessions/{sid}/refund-task").json()["revision"] == 1


def test_natural_turn_changes_order_and_retires_old_flow(detached_api):
    api, client, sid, model = detached_api
    old = submit(detached_api)
    script(model, ("get_refund_task", {}),
           ("update_refund_task", dict(expected_revision=1, order_id="A200")))
    say(client, sid, "改退 A200，原因我待会告诉你")
    task = client.get(f"/sessions/{sid}/refund-task").json()
    assert (task["order_id"], task["reason"], task["next_action"]) == ("A200", "", "ask_reason")
    assert tickets.get_ticket(old["approval_id"])["status"] == "rejected"
    assert not api.app.state.refund_graph.get_state(refund_flow.config(old["approval_id"])).next
    assert client.post("/chat/resume", json={"thread_id": sid, "approval_id": old["approval_id"], "decision": "approve"}).status_code == 409
    script(model, ("update_refund_task", dict(expected_revision=2, reason="尺码不合适")),
           ("submit_refund_request", dict(order_id="A200", reason="尺码不合适", task_revision=3)))
    events = say(client, sid, "新订单尺码不合适，现在提交")
    action = next(d["actions"][0] for k, d in events if k == "interrupt")
    assert action["approval_id"] != old["approval_id"]
    r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": action["approval_id"], "decision": "approve"})
    assert not any(k == "error" for k, _ in parse_sse(r.text)), r.text
    task = client.get(f"/sessions/{sid}/refund-task").json()
    assert task["status"] == "executed" and task["order_id"] == "A200"
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
    assert not api.app.state.refund_graph.get_state(refund_flow.config(action["approval_id"])).next


def test_natural_turn_cancels_pending_without_approval(detached_api):
    api, client, sid, model = detached_api
    old = submit(detached_api)
    script(model, ("update_refund_task", dict(expected_revision=1, cancel=True)))
    events = say(client, sid, "这单先别退了，取消申请")
    assert any(k == "done" for k, _ in events)
    assert client.get(f"/sessions/{sid}/refund-task").json()["status"] == "cancelled"
    assert not api.app.state.refund_graph.get_state(refund_flow.config(old["approval_id"])).next
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 0
    events = client.get(f"/sessions/{sid}/refund-task/history").json()["events"]
    assert events[-1]["snapshot"]["source"] == "chat" and events[-1]["snapshot"]["turn_id"]


def test_raw_workflow_resume_cannot_grant_authority(detached_api):
    from langgraph.types import Command
    import refund_service
    api, client, sid, model = detached_api
    old = submit(detached_api)
    with pytest.raises(refund_service.ApprovalRequired):
        api.app.state.refund_graph.invoke(Command(resume=True), config=refund_flow.config(old["approval_id"]))
    assert tickets.get_ticket(old["approval_id"])["status"] == "pending_approval"


def test_approval_outcome_is_recorded_without_llm_call(detached_api):
    api, client, sid, model = detached_api
    old = submit(detached_api)
    index = model.i
    r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": old["approval_id"], "decision": "approve"})
    events = parse_sse(r.text)
    assert any(k == "done" for k, _ in events), events
    assert model.i == index
    messages = client.get(f"/sessions/{sid}/messages").json()["messages"]
    assert any("退款已执行" in m.get("text", "") for m in messages)


def test_pending_checkpoint_can_be_reconstructed_after_crash(detached_api):
    from langgraph.checkpoint.memory import InMemorySaver
    api, client, sid, model = detached_api
    old = submit(detached_api)
    api.app.state.refund_graph = refund_flow.build(InMemorySaver())
    r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": old["approval_id"], "decision": "approve"})
    assert any(k == "done" for k, _ in parse_sse(r.text)), r.text
    assert tickets.get_ticket(old["approval_id"])["status"] == "executed"


def test_real_evaluation_engine_can_run_natural_cancel(detached_api):
    from types import SimpleNamespace
    from task_eval_model import run_case
    from task_scenarios import cases
    from langchain_core.messages import AIMessage
    api, client, sid, model = detached_api
    def call(name, args, key):
        return AIMessage(content="", tool_calls=[dict(name=name, args=args, id=key, type="tool_call")])
    model.i = 0
    model.responses = [
        call("update_refund_task", dict(expected_revision=0, order_id="A100", reason="质量问题", intent="apply"), "state"),
        call("submit_refund_request", dict(order_id="A100", reason="质量问题", task_revision=1), "submit"),
        AIMessage(content="申请已提交"),
        call("update_refund_task", dict(expected_revision=1, cancel=True), "cancel"),
        AIMessage(content="申请已撤回"),
    ]
    usage = SimpleNamespace(calls=0, input_tokens=0, output_tokens=0, known_usage_calls=0, exhausted=False, last_error=None)
    case = next(c for c in cases() if c["id"] == "cancel")
    row = run_case(api, client, case, 1, usage)
    assert row["passed"], row
    assert len(row["trace"]) == 2 and row["tickets"][0]["status"] == "rejected"


def test_messages_baseline_uses_same_approval_service(detached_api):
    from types import SimpleNamespace
    from task_eval_model import run_case, messages_baseline
    from task_scenarios import cases
    from langchain_core.messages import AIMessage
    api, client, sid, model = detached_api
    api.app.state.llm = model
    api.app.state.agent = messages_baseline(api)
    model.i = 0
    model.responses = [AIMessage(content="", tool_calls=[dict(
        name="submit_refund_request", args=dict(order_id="A100", reason="质量问题"), id="submit", type="tool_call")]),
        AIMessage(content="请审批"), AIMessage(content="好的，已停止申请")]
    usage = SimpleNamespace(calls=0, input_tokens=0, output_tokens=0, known_usage_calls=0, exhausted=False, last_error=None)
    row = run_case(api, client, next(c for c in cases() if c["id"] == "reject"), 1, usage)
    assert row["passed"], row
    assert row["tickets"][0]["approval_protocol"] == "detached"


def test_evaluator_accepts_correction_submitted_before_confirmation(detached_api):
    from types import SimpleNamespace
    from task_eval_model import run_case
    from task_scenarios import cases
    from langchain_core.messages import AIMessage
    api, client, sid, model = detached_api
    def call(name, args, key):
        return AIMessage(content="", tool_calls=[dict(name=name, args=args, id=key, type="tool_call")])
    model.i = 0
    model.responses = [
        call("update_refund_task", dict(expected_revision=0, order_id="A100", reason="质量问题", intent="apply"), "initial"),
        call("submit_refund_request", dict(order_id="A100", reason="质量问题", task_revision=1), "submit-old"),
        AIMessage(content="原申请待审批"),
        call("update_refund_task", dict(expected_revision=1, order_id="A200", reason="尺码不合适"), "correction"),
        call("submit_refund_request", dict(order_id="A200", reason="尺码不合适", task_revision=2), "submit-new"),
        AIMessage(content="新申请待审批"), AIMessage(content="新申请不变，请确认"), AIMessage(content="好的"),
    ]
    usage = SimpleNamespace(calls=0, input_tokens=0, output_tokens=0, known_usage_calls=0, exhausted=False, last_error=None)
    case = dict(next(c for c in cases() if c["id"] == "change_order"), decision="reject")
    row = run_case(api, client, case, 1, usage)
    assert row["passed"], row
    assert len(row["tickets"]) == 2
