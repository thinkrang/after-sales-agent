from datetime import datetime, timezone

import customer_guidance
import orders
import refund_flow
import voucher_disputes
from auth import User
from conftest import parse_sse
from test_task_http import script, task_api


def test_changing_voucher_evidence_replaces_previous_instruction():
    branch = {"voucher_id": "V200", "help_url": "/voucher-center?voucher=V200",
              "needs_recheck": False, "facts": {"case_status": "resolved", "case_outcome": "eligible_to_apply"}}
    contact, = customer_guidance.project(None, None, branch)
    assert contact["code"] == "voucher_contact_support"
    assert contact["link_label"] == "查看复核结论"
    stale, = customer_guidance.project(None, None, dict(branch, needs_recheck=True, facts=None))
    assert stale["code"] == "voucher_recheck"
    waiting, = customer_guidance.project(None, None, dict(branch, facts={"case_status": "pending"}))
    assert waiting["code"] == "voucher_wait_review"


def test_stale_approval_cannot_offer_confirmation():
    task = dict(thread_id="sid", order_id="A300", ticket_id="old-approval", status="executed")
    action = dict(name="create_refund_request", scope_thread_id="sid", approval_id="old-approval",
                  args={"order_id": "A300"}, risk={"required_role": "customer"})
    step, = customer_guidance.project(task, None, None, [action])
    assert step["code"] == "refund_executed" and "approval_id" not in step


def test_sse_and_refresh_use_ticket_and_voucher_state(task_api):
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    api, client, _, model = task_api
    login = client.post("/auth/login", json={"username": "acme-user", "password": "demo1234"})
    client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
    sid = client.post("/sessions", json={}).json()["id"]
    voucher_disputes.seed_demo()
    user = User("acme-user", "客户", "acme", frozenset({"customer"}))
    voucher_disputes.submit(user, "V200", "测试", store_code="CENTRAL", visit_at=datetime.now(timezone.utc))
    api.app.state.refund_graph = refund_flow.build(api.app.state.memory)
    api.app.state.agent = create_agent(
        model=model, checkpointer=api.app.state.memory,
        tools=[api.request_order_refund, api.observe_voucher_branch, api.withdraw_refund_request],
    )
    script(model, ("request_order_refund", {"order_id": "A300", "reason": "尺码不合适"}),
           ("observe_voucher_branch", {"voucher_id": "V200"}))
    model.responses[-1] = AIMessage(content="请等待主管审批，券也要重新提交。")
    response = client.post("/chat/stream", json={"thread_id": sid, "message": "申请 A300 退款，尺码不合适。查 V200 争议。"})
    events = parse_sse(response.text)
    assert not any(kind == "error" for kind, _ in events), events
    assert any(kind == "token" and "等待主管审批" in data["text"] for kind, data in events)
    interrupt, = [data for kind, data in events if kind == "interrupt"]
    refund, voucher = interrupt["next_steps"]
    approval_id = interrupt["actions"][0]["approval_id"]
    assert (refund["code"], refund["approval_id"]) == ("refund_customer_confirm", approval_id)
    assert voucher["code"] == "voucher_wait_review"
    assert client.get(f"/sessions/{sid}/messages").json()["next_steps"] == interrupt["next_steps"]
    assert orders.get_order(user, "A300")["status"] == "delivered"

    script(model, ("withdraw_refund_request", {"approval_id": approval_id}))
    response = client.post("/chat/stream", json={"thread_id": sid, "message": "撤回 A300，保留券争议。"})
    done, = [data for kind, data in parse_sse(response.text) if kind == "done"]
    assert [step["code"] for step in done["next_steps"]] == ["refund_withdrawn", "voucher_wait_review"]
    assert client.get(f"/sessions/{sid}/messages").json()["pending"] == []
    assert client.post("/chat/resume", json={"thread_id": sid, "approval_id": approval_id,
                                            "decision": "approve"}).status_code == 409
