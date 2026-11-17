"""Real ASGI + LangGraph HITL + SQLite; scripted model, no paid API.

Only retrieval/index adapters are replaced here. Task tools, authorization,
approval, refund execution and checkpoint restore all use production code.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

import auth
import db
import orders
import tickets
from conftest import parse_sse


@pytest.fixture
def task_api(tmp_path, monkeypatch):
    # Import an isolated copy: never replace api used by the other test suites.
    from fastapi.testclient import TestClient
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    for name, attrs in {
        "rag": dict(delete_knowledge_document=lambda *a, **k: None,
                    format_evidence=lambda x: "", get_vector_store=lambda: None,
                    list_knowledge_documents=lambda **k: [], search_knowledge=lambda x: ""),
        "retrieval": dict(DEFAULT_STRATEGY="hybrid", retrieve=lambda *a, **k: []),
        # 核查器现在返回结构化结论；测试里不需要真实模型。
        "evidence": dict(cited_evidence=lambda *a: [], verify_evidence=lambda *a: types.SimpleNamespace(
            sufficient=False, quote="", missing="未调用检索")),
        "tasks": dict(get_task_status=lambda x: {}, submit_index_task=lambda *a, **k: {}),
    }.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("task_test_api", Path(__file__).resolve().parents[1] / "api.py")
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, api)
    spec.loader.exec_module(api)
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    monkeypatch.setattr(auth, "USER_DB", tmp_path / "auth.sqlite")
    api.SESSION_DB = tmp_path / "sessions.sqlite"
    api.MEMORY_DB = tmp_path / "memory.sqlite"
    auth.init_auth_db(seed_demo_users=True)
    tickets.init_tickets_table()
    orders.init_order_db()
    api.init_session_db()

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ScriptedModel(responses=[AIMessage(content="已处理")])
    with SqliteSaver.from_conn_string(str(api.MEMORY_DB)) as memory:
        api.app.state.agent = create_agent(
            model=model, tools=[api.get_refund_task, api.update_refund_task, api.create_refund_request],
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"create_refund_request": True})],
            checkpointer=memory,
        )
        api.app.state.memory = memory
        # No context manager: all required services were initialized above;
        # lifespan would load embeddings and construct the real remote model.
        client = TestClient(api.app)
        token = client.post("/auth/login", json={"username": "acme-manager", "password": "demo1234"}).json()["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        sid = client.post("/sessions", json={}).json()["id"]
        yield api, client, sid, model
        client.close()
    db.release_all()


def script(model, *calls):
    from langchain_core.messages import AIMessage
    model.i = 0
    model.responses = [AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": f"call_{i}", "type": "tool_call"
    }]) for i, (name, args) in enumerate(calls)] + [AIMessage(content="处理已完成")]


def test_chat_can_check_voucher_without_creating_a_case(task_api):
    from langchain.agents import create_agent
    import voucher_disputes

    api, client, _, model = task_api
    voucher_disputes.seed_demo()
    auth_response = client.post("/auth/login", json={"username": "acme-user", "password": "demo1234"})
    client.headers["Authorization"] = "Bearer " + auth_response.json()["access_token"]
    sid = client.post("/sessions", json={}).json()["id"]
    page = client.get("/voucher-center")
    assert page.status_code == 200 and "团购券核销争议" in page.text
    api.app.state.agent = create_agent(model=model, tools=[api.check_voucher_redemption],
                                       checkpointer=api.app.state.memory)
    script(model, ("check_voucher_redemption", {"voucher_id": "V200", "store_code": "CENTRAL"}))
    response = client.post("/chat/stream", json={"thread_id": sid, "message": "为什么 V200 显示已经核销？"})
    assert response.status_code == 200
    history = client.get(f"/sessions/{sid}/messages").json()["messages"]
    result = next(m for m in history if m["role"] == "tool_result")
    checked = api.json.loads(result["text"])
    assert checked["reason_code"] == "status_mismatch" and checked["voucher"]["policy_version"] == "voucher-v2"
    assert checked["visit"]["next_step"] == "submit_dispute"
    assert checked["help_url"].startswith("/voucher-center?voucher=V200&store=CENTRAL")
    assert client.get("/voucher-disputes").json() == {"cases": []}
    assert not tickets.list_tickets()


def pending(client, sid, model, order="A100", revision=1):
    script(model, ("create_refund_request", dict(order_id=order, reason="质量问题", task_revision=revision)))
    events = parse_sse(client.post("/chat/stream", json={"thread_id": sid, "message": "请提交"}).text)
    assert not any(k == "error" for k, _ in events), events
    return next(d["actions"][0] for k, d in events if k == "interrupt")


def start_task(client, sid):
    r = client.patch(f"/sessions/{sid}/refund-task", json={
        "expected_revision": 0, "order_id": "A100", "reason": "质量问题", "intent": "apply",
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_real_graph_approve_executes_once_and_closes_tool_calls(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    action = pending(client, sid, model)
    r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": action["approval_id"], "decision": "approve"})
    assert not any(k == "error" for k, _ in parse_sse(r.text)), r.text
    final = client.get(f"/sessions/{sid}/refund-task").json()
    assert final["status"] == "executed" and final["refund_id"]
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
    config = {"configurable": {"thread_id": sid}}
    assert not api.app.state.agent.get_state(config).next
    assert client.post("/chat/resume", json={"thread_id": sid, "approval_id": action["approval_id"], "decision": "approve"}).status_code == 409


def test_edit_pending_rejects_old_approval_and_recovers_real_graph(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    old = pending(client, sid, model)
    assert client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "order_id": "A200"}).json()["missing_fields"] == ["reason"]
    # Old approval cannot execute even before the checkpoint has been repaired.
    assert client.post("/chat/resume", json={"thread_id": sid, "approval_id": old["approval_id"], "decision": "approve"}).status_code == 409
    # First model response closes the rejected tool call; next response handles
    # the new user message and saves a reason against the new revision.
    from langchain_core.messages import AIMessage
    model.i = 0
    model.responses = [
        AIMessage(content="旧申请已作废"),
        AIMessage(content="", tool_calls=[dict(name="update_refund_task", args={"expected_revision": 2, "reason": "尺码不合适"}, id="new-reason", type="tool_call")]),
        AIMessage(content="请确认新申请"),
    ]
    r = client.post("/chat/stream", json={"thread_id": sid, "message": "新订单原因是尺码不合适"})
    events = parse_sse(r.text)
    assert any(k == "done" for k, _ in events) and not any(k == "error" for k, _ in events), events
    task = client.get(f"/sessions/{sid}/refund-task").json()
    assert (task["order_id"], task["reason"], task["revision"]) == ("A200", "尺码不合适", 3)
    assert tickets.get_ticket(old["approval_id"])["status"] == "rejected"
    assert not api.app.state.agent.get_state({"configurable": {"thread_id": sid}}).next


def test_http_task_revision_ownership_and_extra_fields(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    assert client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 0, "cancel": True}).status_code == 409
    assert client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "tenant": "globex"}).status_code == 422
    token = client.post("/auth/login", json={"username": "globex-user", "password": "demo1234"}).json()["access_token"]
    headers = {"Authorization": "Bearer " + token}
    assert client.get(f"/sessions/{sid}/refund-task", headers=headers).status_code == 404
    assert client.patch(f"/sessions/{sid}/refund-task", headers=headers, json={"expected_revision": 1, "cancel": True}).status_code == 404


def test_task_tools_can_fill_slots_with_real_graph(task_api):
    api, client, sid, model = task_api
    script(model, ("get_refund_task", {}), ("update_refund_task", dict(expected_revision=0, intent="apply")))
    client.post("/chat/stream", json={"thread_id": sid, "message": "我要退款"})
    task = client.get(f"/sessions/{sid}/refund-task").json()
    assert task["status"] == "collecting" and task["missing_fields"] == ["order_id", "reason"]
    script(model, ("update_refund_task", dict(expected_revision=1, order_id="A100", reason="质量问题")))
    client.post("/chat/stream", json={"thread_id": sid, "message": "A100，质量问题"})
    assert client.get(f"/sessions/{sid}/refund-task").json()["status"] == "ready"


def test_new_graph_cannot_skip_task_registration(task_api):
    api, client, sid, model = task_api
    action = pending(client, sid, model)
    assert not action.get("approval_id") and action["blocked_reason"]
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets").fetchone()[0] == 0


def test_cancel_survives_failed_graph_repair_and_can_retry(task_api, monkeypatch):
    api, client, sid, model = task_api
    start_task(client, sid)
    old = pending(client, sid, model)
    assert client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "cancel": True}).status_code == 200
    agent = api.app.state.agent

    class BrokenRestore:
        get_state = agent.get_state

        def invoke(self, *args, **kwargs):
            raise TimeoutError("模拟恢复中网络失败")

    monkeypatch.setattr(api.app.state, "agent", BrokenRestore())
    r = client.post("/chat/stream", json={"thread_id": sid, "message": "继续咨询"})
    assert any(k == "error" for k, _ in parse_sse(r.text))
    assert client.get(f"/sessions/{sid}/refund-task").json()["status"] == "cancelled"
    assert tickets.get_ticket(old["approval_id"])["status"] == "rejected"
    monkeypatch.setattr(api.app.state, "agent", agent)
    from langchain_core.messages import AIMessage
    model.i = 0
    model.responses = [AIMessage(content="旧操作已关闭"), AIMessage(content="请问需要咨询什么")]
    r = client.post("/chat/stream", json={"thread_id": sid, "message": "继续咨询"})
    assert any(k == "done" for k, _ in parse_sse(r.text))
    assert not agent.get_state({"configurable": {"thread_id": sid}}).next


def test_synchronous_chat_recovers_changed_task_too(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    old = pending(client, sid, model)
    client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "cancel": True})
    from langchain_core.messages import AIMessage
    model.i = 0
    model.responses = [AIMessage(content="旧操作已关闭"), AIMessage(content="现在可以继续咨询")]
    r = client.post("/chat", json={"thread_id": sid, "message": "继续咨询"})
    assert r.status_code == 200 and r.json()["answer"] == "现在可以继续咨询"
    assert tickets.get_ticket(old["approval_id"])["status"] == "rejected"


def test_old_graph_cannot_approve_new_ticket_for_same_payload(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    old = pending(client, sid, model)
    client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "reason": "尺码不合适"})
    client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 2, "reason": "质量问题"})
    # Model's old graph still carries revision 1. Even a new ticket for the
    # same order/reason must not make that graph's old call approvable.
    user = auth.User("acme-manager", "客户", "acme", frozenset({"customer"}))
    import refund_service
    new, _ = refund_service.prepare_ticket(user, thread_id=sid, order_id="A100", reason="质量问题", task_revision=3)
    assert new["task_revision"] == 3
    r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": new["id"], "decision": "approve"})
    assert r.status_code == 409
    assert tickets.get_ticket(old["approval_id"])["status"] == "rejected"


def test_execution_tool_rejects_stale_task_revision(task_api):
    api, client, sid, model = task_api
    start_task(client, sid)
    client.patch(f"/sessions/{sid}/refund-task", json={"expected_revision": 1, "reason": "尺码不合适"})
    config = {"configurable": {"thread_id": sid, "tenant": "acme", "user_id": "acme-manager", "roles": ["customer"]}}
    response = api.create_refund_request.invoke({"order_id": "A100", "reason": "质量问题", "task_revision": 1}, config=config)
    assert "未执行" in response
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 0


def test_http_queue_filters_before_limit_and_preserves_audit_view(task_api):
    api, client, sid, model = task_api
    target = tickets.create_ticket(thread_id="old", order_id="queue-old", tenant_id="acme",
                                    requester_id="acme-user", required_role="approver")
    for i in range(4):
        tickets.create_ticket(thread_id=f"own{i}", order_id=f"own{i}", tenant_id="acme",
                              requester_id="acme-manager", required_role="approver")
        tickets.create_ticket(thread_id=f"low{i}", order_id=f"low{i}", tenant_id="acme",
                              requester_id="acme-user", required_role="customer")
        tickets.create_ticket(thread_id=f"other{i}", order_id=f"other{i}", tenant_id="globex",
                              requester_id="globex-user", required_role="approver")
    response = client.get("/tickets?actionable_only=true&limit=1")
    assert response.status_code == 200
    assert [t["id"] for t in response.json()["tickets"]] == [target["id"]]
    assert len(client.get("/tickets").json()["tickets"]) == 9
    token = client.post("/auth/login", json={"username": "acme-user", "password": "demo1234"}).json()["access_token"]
    assert client.get("/tickets?actionable_only=true", headers={"Authorization": "Bearer " + token}).status_code == 403


def test_http_pending_filter_precedes_limit(task_api):
    api, client, sid, model = task_api
    target = tickets.create_ticket(thread_id=sid, order_id="queue-target", tenant_id="acme",
                                    requester_id="acme-manager")
    for i in range(4):
        done = tickets.create_ticket(thread_id=f"history{i}", order_id=f"history{i}",
                                     tenant_id="acme", requester_id="acme-manager")
        tickets.reject_ticket(done["id"])
    assert client.get("/tickets?pending_only=true&limit=1").json()["tickets"][0]["id"] == target["id"]


def test_legacy_high_risk_card_can_reject_but_not_self_approve(task_api):
    api, client, sid, model = task_api
    task = client.patch(f"/sessions/{sid}/refund-task", json={
        "expected_revision": 0, "order_id": "A500", "reason": "质量问题", "intent": "apply",
    }).json()
    action = pending(client, sid, model, order="A500", revision=task["revision"])
    payload = dict(thread_id=sid, approval_id=action["approval_id"], decision="approve")
    assert client.post("/chat/resume", json=payload).status_code == 403
    payload["decision"] = "reject"
    response = client.post("/chat/resume", json=payload)
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert not any(k == "error" for k, _ in events), events
    assert tickets.get_ticket(action["approval_id"])["status"] == tickets.REJECTED
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 0
