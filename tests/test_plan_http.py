"""Production chat tools + actual ASGI/graph/checkpoints + two-task approvals."""
import json
import time

import pytest

import db
import orders
import orchestrator
import plan_store
import tickets
from conftest import parse_sse
from plan_eval import fixed_plan_tool
from planning import refund_plan
from test_task_http import task_api, script
from test_detached_tasks import detached_api, say


@pytest.fixture
def plan_api(detached_api):
    from langchain.agents import create_agent
    api, client, sid, model = detached_api
    api.app.state.agent = create_agent(model=model, tools=[api.create_refund_plan, api.get_refund_plan,
        api.edit_refund_plan_task, api.update_refund_task, api.submit_refund_request], checkpointer=api.app.state.memory)
    token = client.post("/auth/login", json={"username":"acme-user", "password":"demo1234"}).json()["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    sid = client.post("/sessions", json={}).json()["id"]
    return api, client, sid, model


def submit(env):
    api, client, sid, model = env
    draft = refund_plan("两笔退款", [dict(key="a", order_id="A300", reason="尺码不合适", intent="apply"),
                                    dict(key="b", order_id="A400", reason="质量问题", intent="apply")])
    script(model, ("create_refund_plan", {"draft":draft.model_dump()}))
    events = say(client, sid, "A300尺码不合适，A400有质量问题，都申请退款")
    event = next(v for k,v in events if k == "interrupt")
    assert len(event["actions"]) == 2
    assert event["refund_plan"]["status"] == "waiting_approval"
    assert all(a["task_id"] and a["scope_thread_id"] != sid for a in event["actions"])
    return {a["args"]["order_id"]: a for a in event["actions"]}


def test_chat_multi_plan_approvals_and_original_conversation_receipts(plan_api):
    api, client, sid, model = plan_api
    actions = submit(plan_api)
    assert client.post("/chat/resume", json={"thread_id":sid,"approval_id":actions["A400"]["approval_id"],"decision":"approve"}).status_code == 403
    response = client.post("/chat/resume", json={"thread_id":sid,"approval_id":actions["A300"]["approval_id"],"decision":"approve"})
    assert response.status_code == 200 and not any(k == "error" for k,v in parse_sse(response.text)), response.text
    snap = client.get(f"/sessions/{sid}/messages").json()
    assert len(snap["pending"]) == 1
    assert snap["pending"][0]["approval_id"] == actions["A400"]["approval_id"]
    # Side question still goes through the model while high-risk approval waits.
    script(model, ("get_refund_plan", {}))
    say(client, sid, "先查一下处理进度")
    token = client.post("/auth/login", json={"username":"acme-manager","password":"demo1234"}).json()["access_token"]
    result = client.post(f"/tickets/{actions['A400']['approval_id']}/decision", json={"decision":"approve"}, headers={"Authorization":"Bearer " + token})
    assert result.status_code == 200 and result.json()["session_synced"], result.text
    snap = client.get(f"/sessions/{sid}/messages").json()
    assert snap["refund_plan"]["status"] == "completed"
    assert not snap["pending"]
    for task in snap["refund_plan"]["tasks"]:
        assert task["state"]["refund_id"] in json.dumps(snap["messages"])
    events = client.get(f"/sessions/{sid}/plan/events").json()["events"]
    assert events and all(e["plan_id"] == snap["refund_plan"]["id"] for e in events)


def test_http_replan_revokes_only_one_and_closes_its_checkpoint(plan_api):
    api, client, sid, model = plan_api
    actions = submit(plan_api)
    high = actions["A400"]
    response = client.patch(f"/sessions/{sid}/plan/tasks/{high['task_id']}", json={"expected_revision":1,"reason":"包装破损"})
    assert response.status_code == 200, response.text
    assert tickets.get_ticket(high["approval_id"])["status"] == "rejected"
    assert not api.app.state.refund_graph.get_state(api.refund_flow.config(high["approval_id"])).next
    assert tickets.get_ticket(actions["A300"]["approval_id"])["status"] == "pending_approval"
    assert client.post("/chat/resume", json={"thread_id":sid,"approval_id":high["approval_id"],"decision":"approve"}).status_code == 409
    assert client.patch(f"/sessions/{sid}/plan/tasks/{high['task_id']}", json={"expected_revision":1,"cancel":True}).status_code == 409
    snap = client.get(f"/sessions/{sid}/plan").json()
    assert len(snap["pending"]) == 2


def test_task_and_plan_endpoints_enforce_session_ownership(plan_api):
    api, client, sid, model = plan_api
    actions = submit(plan_api)
    other_sid = client.post("/sessions",json={}).json()["id"]
    assert client.patch(f"/sessions/{other_sid}/plan/tasks/{actions['A300']['task_id']}",json={"expected_revision":1,"cancel":True}).status_code == 409
    token = client.post("/auth/login",json={"username":"globex-user","password":"demo1234"}).json()["access_token"]
    headers = {"Authorization":"Bearer " + token}
    for suffix in ("plan", "plan/events", "messages"):
        assert client.get(f"/sessions/{sid}/{suffix}",headers=headers).status_code in {403,404}
    assert client.post("/chat/resume",json={"thread_id":other_sid,"approval_id":actions["A300"]["approval_id"],"decision":"approve"}).status_code == 409


def test_legacy_tools_do_not_overwrite_multi_task_plan(plan_api):
    api, client, sid, model = plan_api
    actions = submit(plan_api)
    script(model, ("update_refund_task", {"expected_revision":0,"order_id":"A300","reason":"误修改","intent":"apply"}))
    say(client, sid, "查看状态")
    plan = client.get(f"/sessions/{sid}/plan").json()
    assert all(t["state"]["revision"] == 1 for t in plan["tasks"])
    assert client.patch(f"/sessions/{sid}/refund-task",json={"expected_revision":0,"cancel":True}).status_code == 409


def test_approved_branch_can_resume_when_sibling_still_pending(plan_api):
    api, client, sid, model = plan_api
    actions = submit(plan_api)
    user = api.authenticate("acme-user", "demo1234")
    api.refund_service.decide(user,approval_id=actions["A300"]["approval_id"],decision="approve")
    # Approval persisted; response lost before execution. The sibling is still pending.
    response = client.post("/chat/resume",json={"thread_id":sid,"approval_id":actions["A300"]["approval_id"],"decision":"approve"})
    assert response.status_code == 200 and not any(k == "error" for k,v in parse_sse(response.text)), response.text
    assert tickets.get_ticket(actions["A300"]["approval_id"])["status"] == "executed"
    assert tickets.get_ticket(actions["A400"]["approval_id"])["status"] == "pending_approval"


def test_cannot_delete_session_until_all_live_tickets_are_resolved(plan_api):
    api, client, sid, _ = plan_api
    actions = submit(plan_api)
    low, high = actions["A300"], actions["A400"]
    assert client.delete(f"/sessions/{sid}").status_code == 409
    for action in (low, high):
        result = client.patch(f"/sessions/{sid}/plan/tasks/{action['task_id']}",
                              json={"expected_revision": 1, "cancel": True})
        assert result.status_code == 200, result.text
        assert tickets.get_ticket(action["approval_id"])["status"] == tickets.REJECTED
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.get(f"/sessions/{sid}/messages").status_code == 404
    assert client.delete(f"/sessions/{sid}").status_code == 200


@pytest.mark.parametrize("interruption", ["after_approval", "after_gateway", "after_ticket"])
def test_recovery_projects_one_durable_receipt(plan_api, interruption):
    api, client, sid, _ = plan_api
    actions = submit(plan_api)
    ticket = tickets.get_ticket(actions["A300"]["approval_id"])
    user = api.authenticate("acme-user", "demo1234")
    api.refund_service.decide(user, approval_id=ticket["id"], decision="approve")
    if interruption == "after_gateway":
        claimed = tickets.acquire_execution(
            ticket["thread_id"], tenant_id=ticket["tenant_id"], requester_id=ticket["requester_id"],
            order_id=ticket["order_id"], payload_hash=ticket["payload_hash"])
        assert claimed["status"] == tickets.EXECUTING
        orders.execute_refund(user=user, order_id=ticket["order_id"],
                              amount_cents=ticket["amount_cents"], idempotency_key=ticket["id"])
    elif interruption == "after_ticket":
        assert api.refund_service.execute_for_ticket(ticket)["status"] == tickets.EXECUTED
    if interruption != "after_ticket":
        assert client.delete(f"/sessions/{sid}").status_code == 409

    for _ in range(2):
        orchestrator.recover_once()
        result = tickets.get_ticket(ticket["id"])
        assert result["status"] == tickets.EXECUTED
        history = client.get(f"/sessions/{sid}/messages").json()
        assert sum(result["refund_id"] in m.get("text", "") for m in history["messages"]) == 1
        assert tickets.get_ticket(actions["A400"]["approval_id"])["status"] == tickets.PENDING
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1


def test_close_fences_a_step_claimed_before_the_delete(plan_api):
    api, client, sid, _ = plan_api
    user = api.authenticate("acme-user", "demo1234")
    draft = refund_plan("一笔退款", [dict(key="a", order_id="A300", reason="尺码不合适", intent="apply")])
    plan_store.create(user, sid, draft)
    orchestrator.run(user, sid, max_waves=2)
    claim, = orchestrator._claim(user, sid, now=time.time())
    assert claim["action"] == "request_refund"
    assert client.delete(f"/sessions/{sid}").status_code == 200
    orchestrator._run_step(user, claim)
    assert not tickets.list_tickets(requester_id=user.username, tenant_id=user.tenant_id)


def test_fixed_baseline_uses_same_plan_and_approval_services(plan_api):
    from langchain.agents import create_agent

    api, client, sid, model = plan_api
    fixed = fixed_plan_tool(api.create_refund_plan)
    api.app.state.agent = create_agent(model=model, tools=[fixed], checkpointer=api.app.state.memory)
    script(model, (fixed.name, {"batch": {"goal": "两笔退款", "tasks": [
        {"key": "a", "order_id": "A300", "reason": "尺码不合适", "intent": "apply"},
        {"key": "b", "order_id": "A400", "reason": "质量问题", "intent": "apply"},
    ]}}))
    events = say(client, sid, "请申请两笔退款")
    interrupt = next(value for kind, value in events if kind == "interrupt")
    assert {a["args"]["order_id"] for a in interrupt["actions"]} == {"A300", "A400"}
    assert interrupt["refund_plan"]["status"] == "waiting_approval"
