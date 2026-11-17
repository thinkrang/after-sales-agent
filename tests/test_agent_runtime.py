"""Exercise the real graph with scripted model replies and I/O faults."""
import json
import logging
import time
from types import SimpleNamespace
import uuid

import httpx
import pytest
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

import agent_runtime
from agent_context import ContextBoundary, Limits
import db
import evidence
import orders
import refund_service
import refund_tasks
import tickets
import voucher_disputes
from conftest import parse_sse
from test_task_http import script, task_api


class ToolResponses:
    """A synthetic test tape. Replay invokes no DB or external tools."""

    def __init__(self, records=None):
        self.records = records if records is not None else []
        self.replaying = records is not None
        self.index = 0

    def run(self, name, args, call):
        if self.replaying:
            expected = self.records[self.index]
            self.index += 1
            assert (name, args) == (expected["name"], expected["args"])
            if expected.get("connection_lost"):
                raise ConnectionResetError("synthetic response lost")
            return expected["response"]
        try:
            response = call()
        except ConnectionResetError:
            self.records.append({"name": name, "args": args, "connection_lost": True})
            raise
        self.records.append({"name": name, "args": args, "response": response})
        return response


def runtime_events(caplog, turn_id):
    return [record.agent_event for record in caplog.records
            if getattr(record, "agent_event", {}).get("turn_id") == turn_id]


def invoke(graph, sid, message, *, scope=True):
    turn_id = uuid.uuid4().hex
    config = {"turn_id": turn_id, "turn_started": time.monotonic()}
    if scope:
        config.update(thread_id=sid, tenant="acme", user_id="acme-user", roles=["customer"])
    state = graph.invoke({"messages": [{"role": "user", "content": message}]}, {"configurable": config})
    return state, turn_id


def test_mixed_case_lost_write_receipt_then_replay(task_api, monkeypatch, caplog):
    api, _, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    voucher_disputes.seed_demo()
    user = api.authenticate("acme-user", "demo1234")
    secret_reason = "synthetic user note that must not enter telemetry"
    refund_tasks.update(user, sid, expected_revision=0, order_id="A300", reason=secret_reason, intent="apply")
    original_lookup = orders.get_order
    reads = []

    def intermittent_lookup(user, number):
        reads.append(number)
        if len(reads) == 1:
            raise TimeoutError("synthetic transient lookup timeout")
        return original_lookup(user, number)

    monkeypatch.setattr(orders, "get_order", intermittent_lookup)
    first = ToolResponses()

    def run(tape):
        @tool
        def get_order_info(order_id: str, config: RunnableConfig) -> str:
            """Look up the customer's order."""
            return tape.run("get_order_info", {"order_id": order_id},
                            lambda: api.get_order_info.invoke({"order_id": order_id}, config=config))

        @tool
        def check_voucher_redemption(voucher_id: str, config: RunnableConfig) -> str:
            """Check a voucher without editing it."""
            return tape.run("check_voucher_redemption", {"voucher_id": voucher_id},
                            lambda: api.check_voucher_redemption.invoke({"voucher_id": voucher_id}, config=config))

        @tool
        def submit_refund_request(order_id: str, reason: str, task_revision: int, config: RunnableConfig) -> str:
            """Submit the current refund request for approval."""
            args = {"order_id": order_id, "reason": reason, "task_revision": task_revision}

            def write_then_disconnect():
                written = json.loads(api.submit_refund_request.invoke(args, config=config))
                assert written["status"] == "pending_approval"
                raise ConnectionResetError("synthetic response lost after commit")

            return tape.run("submit_refund_request", args, write_then_disconnect)

        @tool
        def get_refund_task(config: RunnableConfig) -> str:
            """Look up whether the refund request was submitted."""
            return tape.run("get_refund_task", {}, lambda: api.get_refund_task.invoke({}, config=config))

        args = {"order_id": "A300", "reason": secret_reason, "task_revision": 1}
        script(model, ("get_order_info", {"order_id": "A300"}), ("check_voucher_redemption", {"voucher_id": "V200"}),
               ("submit_refund_request", args), ("submit_refund_request", args), ("get_refund_task", {}))
        model.responses.insert(3, AIMessage(content=""))
        graph = create_agent(model=model, tools=[get_order_info, check_voucher_redemption,
                                                submit_refund_request, get_refund_task],
                             middleware=[ContextBoundary()])
        result, turn = invoke(graph, sid, "查 V200，并提交 A300 的退款申请", scope=not tape.replaying)
        return [json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)], turn

    initial, captured_turn = run(first)
    assert initial[0]["order_id"] == "A300" and initial[1]["reason_code"] == "status_mismatch"
    assert [initial[i]["code"] for i in (2, 3)] == ["outcome_unknown", "outcome_unknown"]
    assert initial[4]["status"] == "pending_approval"
    approval_id = initial[4]["ticket_id"]
    assert [row["name"] for row in first.records] == [
        "get_order_info", "get_order_info", "check_voucher_redemption", "submit_refund_request", "get_refund_task",
    ]
    assert [t["id"] for t in tickets.list_tickets()] == [approval_id]
    refund_service.decide(user, approval_id=approval_id, thread_id=sid, decision="approve")
    ticket = tickets.get_ticket(approval_id)
    refund = refund_service.execute_for_ticket(ticket)
    assert refund_service.execute_for_ticket(ticket)["refund_id"] == refund["refund_id"]
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
    observed = runtime_events(caplog, captured_turn)
    assert any(e["event"] == "tool_attempt" and e["attempt"] == 2 and e["tool"] == "get_order_info" for e in observed)
    retry = next(i for i, event in enumerate(observed) if event["event"] == "model_retry")
    blocked = next(i for i, event in enumerate(observed) if event["event"] == "tool_blocked"
                   and event["reason"] == "outcome_unknown")
    assert retry < blocked
    assert observed[-1]["task_status"] == "pending_approval"
    assert secret_reason not in json.dumps(observed)

    recorded = ToolResponses(json.loads(json.dumps(first.records)))
    replayed, replayed_turn = run(recorded)
    assert replayed == initial and recorded.index == len(recorded.records)
    assert len(tickets.list_tickets()) == 1
    assert any(e["event"] == "turn_stop" and e["reason"] == "response_ready"
               for e in runtime_events(caplog, replayed_turn))


def test_repeated_invalid_input_stops_without_retrying_or_writing(task_api, caplog):
    api, _, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    voucher_disputes.seed_demo()
    bad = ("check_voucher_redemption", {"voucher_id": "V200", "visit_at": "2026-11-17T12:00:00"})
    script(model, ("get_refund_task", {}), bad, bad, bad)
    graph = create_agent(model=model, tools=[api.get_refund_task, api.check_voucher_redemption],
                         middleware=[ContextBoundary()])
    state, turn = invoke(graph, sid, "查退款进度和 V200")
    assert state["messages"][-1].response_metadata["boundary_reason"] == "repeat_limit"
    events = runtime_events(caplog, turn)
    bad_calls = [e for e in events if e["event"] == "tool_attempt" and e["tool"] == "check_voucher_redemption"]
    assert [(e["attempt"], e["code"]) for e in bad_calls] == [(1, "invalid_args"), (1, "invalid_args")]
    assert not tickets.list_tickets()


def test_model_request_gets_remaining_io_timeout(task_api, caplog):
    _, _, sid, _ = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    observed = []

    def simulate_timeout(request):
        observed.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("synthetic model timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(simulate_timeout)) as client:
        model = ChatOpenAI(model="timeout-demo", api_key="no-network", max_retries=0,
                           timeout=60, http_client=client, base_url="http://example.invalid")
        graph = create_agent(model=model, tools=[], middleware=[ContextBoundary(Limits(seconds=2))])
        state, turn = invoke(graph, sid, "查 V200")
    assert len(observed) == 1
    assert all(0 < value <= 2 for value in observed[0].values())
    assert state["messages"][-1].response_metadata["boundary_reason"] == "model_timeout"
    assert runtime_events(caplog, turn)[-1]["reason"] == "model_timeout"


@pytest.mark.parametrize(("limit", "attempts", "stop"), [
    (1, 1, "model_limit"),
    (2, 2, "model_limit"),
    (5, 3, "model_empty_response"),
])
def test_empty_output_retry_counts_against_turn_budget(task_api, caplog, limit, attempts, stop):
    api, _, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    script(model, ("get_refund_task", {}))
    model.responses = [AIMessage(content=""), model.responses[0], AIMessage(content=" ")]
    graph = create_agent(model=model, tools=[api.get_refund_task],
                         middleware=[ContextBoundary(Limits(model_calls=limit))])

    result, turn = invoke(graph, sid, "查退款进度")

    assert result["messages"][-1].response_metadata["boundary_reason"] == stop
    events = runtime_events(caplog, turn)
    assert [event["ordinal"] for event in events if event["event"] == "model_attempt"] == list(range(1, attempts + 1))
    assert sum(event["event"] == "model_retry" for event in events) == (limit > 1)
    assert sum(isinstance(message, ToolMessage) for message in result["messages"]) == (limit > 1)
    assert not tickets.list_tickets()


def test_empty_output_retry_checks_remaining_deadline(task_api, caplog, monkeypatch):
    from langchain.agents.middleware import AgentMiddleware

    _, _, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    start = time.monotonic()
    clock = SimpleNamespace(monotonic=lambda: start)
    monkeypatch.setattr(agent_runtime, "time", clock)
    model.responses = [AIMessage(content="")]

    class ConsumeDeadline(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            response = handler(request)
            clock.monotonic = lambda: start + 10
            return response

    graph = create_agent(model=model, tools=[], middleware=[ContextBoundary(Limits(seconds=2)), ConsumeDeadline()])
    turn = uuid.uuid4().hex
    result = graph.invoke({"messages": [{"role": "user", "content": "查退款进度"}]},
                          {"configurable": {"thread_id": sid, "turn_started": start, "turn_id": turn}})

    assert result["messages"][-1].response_metadata["boundary_reason"] == "deadline_exceeded"
    assert [event["event"] for event in runtime_events(caplog, turn)] == ["model_attempt", "turn_stop"]


def test_empty_model_output_returns_sse_fallback_then_recovers_in_next_turn(task_api, caplog):
    api, client, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    model.responses = [AIMessage(content=""), AIMessage(content=""),
                       AIMessage(content=""), AIMessage(content="目前没有退款申请")]
    api.app.state.agent = create_agent(model=model, tools=[], middleware=[ContextBoundary()],
                                       checkpointer=api.app.state.memory)
    for expected in ("模型暂时未能给出可用回复", "目前没有退款申请"):
        response = client.post("/chat/stream", json={"thread_id": sid, "message": "查退款进度"})
        events = parse_sse(response.text)
        assert [kind for kind, _ in events if kind in {"done", "error"}] == ["done"]
        assert expected in "".join(data["text"] for kind, data in events if kind == "token")

    history = api.app.state.agent.get_state({"configurable": {"thread_id": sid}}).values["messages"]
    assert [message.content for message in history if isinstance(message, AIMessage)] == [
        "模型暂时未能给出可用回复。已完成的操作以工单记录为准，可查询状态后继续。", "目前没有退款申请",
    ]
    assert sum(record.agent_event["event"] == "model_retry" for record in caplog.records
               if hasattr(record, "agent_event")) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_malformed_provider_tool_call_is_never_replayed_in_history(task_api, caplog, mixed):
    api, _, sid, _ = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    requests = []
    good = {"id": "valid", "type": "function", "function": {"name": "get_refund_task", "arguments": "{}"}}
    bad = {"id": "invalid", "type": "function", "function": {"name": "get_refund_task", "arguments": "{bad-json"}}
    batches = [[good, bad]] if mixed else [[bad], [good]]
    responses = [{"message": {"role": "assistant", "content": None, "tool_calls": batch}, "finish_reason": "tool_calls"}
                 for batch in batches]
    responses.append({"message": {"role": "assistant", "content": "我已查询工单"}, "finish_reason": "stop"})

    def send(request):
        requests.append(json.loads(request.content)["messages"])
        choice = responses[len(requests) - 1]
        return httpx.Response(200, json={"id": "synthetic", "model": "sdk-test",
                                        "choices": [{"index": 0, **choice}]})

    with httpx.Client(transport=httpx.MockTransport(send)) as client:
        model = ChatOpenAI(model="sdk-test", api_key="no-network", max_retries=0,
                           http_client=client, base_url="http://example.invalid")
        graph = create_agent(model=model, tools=[api.get_refund_task], middleware=[ContextBoundary()])
        state, turn = invoke(graph, sid, "查退款进度")

    assert len(requests) == (2 if mixed else 3)
    assert state["messages"][-1].content == "我已查询工单"
    assert not any(msg.invalid_tool_calls for msg in state["messages"] if isinstance(msg, AIMessage))
    assert [msg["tool_call_id"] for msg in requests[-1] if msg["role"] == "tool"] == ["valid"]
    assistant_calls = [call["id"] for msg in requests[-1] if msg["role"] == "assistant"
                       for call in msg.get("tool_calls", [])]
    assert assistant_calls == ["valid"]
    if not mixed:
        assert not any(msg["role"] == "assistant" for msg in requests[1])
    first = runtime_events(caplog, turn)[0]
    assert (first["code"], first["finish_reason"], first["invalid_tool_calls"], first["proposed_tools"]) == (
        None if mixed else "model_invalid_tool_call", "tool_calls", 1, int(mixed),
    )


def test_content_filter_does_not_trigger_empty_output_retry(task_api, caplog):
    _, _, sid, model = task_api
    caplog.set_level(logging.INFO, logger=agent_runtime.LOGGER.name)
    model.responses = [AIMessage(content="", response_metadata={"finish_reason": "content_filter"})]
    graph = create_agent(model=model, tools=[], middleware=[ContextBoundary()])

    state, turn = invoke(graph, sid, "查退款进度")

    assert state["messages"][-1].response_metadata["boundary_reason"] == "model_refused"
    assert [event["event"] for event in runtime_events(caplog, turn)] == ["model_attempt", "turn_stop"]


def test_evidence_model_inherits_tool_deadline(monkeypatch):
    observed = []

    def simulate_timeout(request):
        observed.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("synthetic verifier timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(simulate_timeout)) as client:
        model = ChatOpenAI(model="timeout-demo", api_key="no-network", max_retries=0,
                           timeout=60, http_client=client, base_url="http://example.invalid")
        monkeypatch.setattr(evidence, "get_llm", lambda: model)
        token = agent_runtime.tool_deadline.set(time.monotonic() + 2)
        try:
            with pytest.raises(evidence.EvidenceUnavailable):
                evidence.verify_evidence("多久能退", [{"source": "policy", "content": "7 天"}])
        finally:
            agent_runtime.tool_deadline.reset(token)
    assert len(observed) == 1
    assert all(0 < value <= 2 for value in observed[0].values())
