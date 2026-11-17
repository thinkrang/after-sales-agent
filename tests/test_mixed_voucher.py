"""A customer chat can advance a voucher and an independent refund."""
from datetime import datetime, timezone
import json

import pytest
from langchain.agents import create_agent

from agent_context import ContextBoundary
from auth import User
import db
import mixed_voucher
import orders
import refund_service
import refund_tasks
import tickets
import voucher_disputes
from test_detached_tasks import detached_api, say
from test_task_http import script, task_api


@pytest.fixture
def mixed_api(detached_api):
    api, client, _, model = detached_api
    voucher_disputes.seed_demo()
    token = client.post("/auth/login", json={"username": "acme-user", "password": "demo1234"}).json()["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    sid = client.post("/sessions", json={}).json()["id"]
    api.app.state.agent = create_agent(model=model, checkpointer=api.app.state.memory, middleware=[ContextBoundary()],
        tools=[api.get_voucher_branch, api.observe_voucher_branch, api.choose_voucher_step,
               api.get_refund_task, api.update_refund_task, api.submit_refund_request])
    return api, client, sid, model


def test_mixed_chat_progresses_refund_and_only_refreshes_voucher(mixed_api):
    api, client, sid, model = mixed_api
    user = api.authenticate("acme-user", "demo1234")
    script(model, ("get_voucher_branch", {}),
           ("observe_voucher_branch", {"voucher_id": "V200", "store_code": "CENTRAL"}),
           ("choose_voucher_step", {"action": "ask_visit", "expected_revision": 1}),
           ("update_refund_task", {"order_id": "A300", "reason": "尺码不合适", "intent": "apply", "expected_revision": 0}),
           ("submit_refund_request", {"order_id": "A300", "reason": "尺码不合适", "task_revision": 1}))
    events = say(client, sid, "V200 在中心店用不了。A300 尺码不合适，请申请退款")
    assert any(kind == "tool" and "not_started" in payload["text"] for kind, payload in events)
    approval = next(payload["actions"][0] for kind, payload in events if kind == "interrupt")
    initial = mixed_voucher.get(user, sid)
    assert initial["facts"]["missing"] == ["visit_at"] and initial["decision"] == "ask_visit"
    assert tickets.get_ticket(approval["approval_id"])["status"] == tickets.PENDING

    arrival = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    script(model, ("get_voucher_branch", {}),
           ("observe_voucher_branch", {"voucher_id": "V200", "visit_at": arrival, "expected_revision": 2}),
           ("choose_voucher_step", {"action": "guide_dispute", "expected_revision": 3}))
    say(client, sid, f"到店时间是 {arrival}")
    branch = mixed_voucher.get(user, sid)
    assert branch["decision"] == "guide_dispute" and branch["visit_at"] == arrival
    assert "store=CENTRAL" in branch["help_url"]
    assert not voucher_disputes.list_cases(user)

    submitted = client.post("/voucher-disputes", json={"voucher_id": "V200", "claim": "到店无法核销",
        "visit_at": arrival, "store_code": "CENTRAL"}).json()
    script(model, ("get_voucher_branch", {}), ("observe_voucher_branch", {"voucher_id": "V200", "expected_revision": 4}),
           ("choose_voucher_step", {"action": "wait_review", "expected_revision": 5}))
    say(client, sid, "争议单提交了，看看状态")
    current = mixed_voucher.get(user, sid)
    assert current["facts"]["case_status"] == "pending_review" and current["decision"] == "wait_review"
    assert current["facts"]["case_id"] == submitted["id"]
    response = client.post("/chat/resume", json={"thread_id": sid, "approval_id": approval["approval_id"],
                                                "decision": "approve"})
    assert response.status_code == 200
    original = tickets.get_ticket(approval["approval_id"])
    assert original["status"] == tickets.EXECUTED

    platform = next(item for item in voucher_disputes.inspect(user, "V200")["evidence"] if item["source"] == "platform")
    outcome = voucher_disputes.ingest_merchant_evidence("acme", "merchant-update", dict(
        voucher_id="V200", source_version=1, record_id="pos-v200", status="redeemed",
        redeem_ref=platform["redeem_ref"], redeemed_at=platform["redeemed_at"]))
    assert outcome["case_rechecked"]
    stale = mixed_voucher.get(user, sid)
    assert stale["needs_recheck"] and stale["facts"] is None and stale["decision"] is None
    with pytest.raises(voucher_disputes.Conflict, match="证据"):
        mixed_voucher.choose(user, sid, current["revision"], "wait_review")
    script(model, ("get_voucher_branch", {}), ("observe_voucher_branch", {"voucher_id": "V200", "expected_revision": 6}),
           ("choose_voucher_step", {"action": "wait_review", "expected_revision": 7}))
    say(client, sid, "商家补传了，看看券现在怎么处理")
    snapshot = client.get(f"/sessions/{sid}/messages").json()["voucher_branch"]
    assert snapshot["upstream"]["merchant"] == 1
    assert snapshot["facts"]["reason_code"] == "matching_redemption"
    reviewer = api.authenticate("acme-manager", "demo1234")
    current_case = voucher_disputes.detail(reviewer, submitted["id"])
    voucher_disputes.review(reviewer, current_case["id"], current_case["revision"],
                            outcome="redemption_confirmed", note="已核对商家补传的凭证")
    script(model, ("get_voucher_branch", {}), ("observe_voucher_branch", {"voucher_id": "V200", "expected_revision": 8}),
           ("choose_voucher_step", {"action": "report_review", "expected_revision": 9}))
    say(client, sid, "现在有复核结论了吗")
    assert mixed_voucher.get(user, sid)["facts"]["case_outcome"] == "redemption_confirmed"
    assert mixed_voucher.get(user, sid)["decision"] == "report_review"
    assert tickets.get_ticket(approval["approval_id"]) == original
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets").fetchone()[0] == 1


def test_mixed_voucher_scope_prerequisites_and_idempotence(mixed_api):
    api, _, sid, _ = mixed_api
    customer = api.authenticate("acme-user", "demo1234")
    other = User("acme-other", "", "acme", frozenset({"customer"}))
    row = mixed_voucher.observe(customer, sid, "V300")
    assert row["facts"]["reason_code"] == "matching_redemption"
    with pytest.raises(voucher_disputes.Conflict):
        mixed_voucher.choose(customer, sid, row["revision"], "guide_dispute")
    with pytest.raises(voucher_disputes.NotFound):
        mixed_voucher.observe(other, sid, "V300")
    assert mixed_voucher.get(other, sid) is None
    done = mixed_voucher.choose(customer, sid, row["revision"], "explain")
    assert mixed_voucher.observe(customer, sid, "V300") == done
    assert mixed_voucher.choose(customer, sid, done["revision"], "explain") == done
    assert not voucher_disputes.list_cases(customer)


def test_model_contract_distinguishes_required_inputs_and_evidence_versions(mixed_api):
    api, _, sid, _ = mixed_api
    user = api.authenticate("acme-user", "demo1234")
    config = {"configurable": {"thread_id": sid, "tenant": user.tenant_id, "user_id": user.username, "roles": ["customer"]}}
    state = json.loads(api.observe_voucher_branch.invoke({"voucher_id": "V300"}, config=config))
    assert state["next_step"] == state["decision"] == "explain" and state["required_fields"] == []
    assert mixed_voucher.get(user, sid)["decision"] == "explain"
    assert "已核销" in state["redemption_finding"] and "upstream" not in state and "facts" not in state
    bad_choice = json.loads(api.choose_voucher_step.invoke(
        {"action": "ask_visit", "expected_revision": state["revision"]}, config=config))
    assert bad_choice["code"] == "conflict" and bad_choice["next_step"] == "explain"
    assert bad_choice["required_fields"] == [] and bad_choice["decision"] == "explain"
    # Existing sessions may have observed the voucher without accepting its step.
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("UPDATE mixed_voucher_branches SET decision=NULL WHERE conversation_id=?", (sid,))
    recovered = mixed_voucher.observe(user, sid, "V300", expected_revision=state["revision"])
    assert recovered["decision"] == "explain" and recovered["revision"] == state["revision"] + 1
    assert mixed_voucher.observe(user, sid, "V300") == recovered

    second_sid = "other-voucher-session"
    case = voucher_disputes.submit(user, "V400", "到店无法使用", store_code="CENTRAL")
    reviewer = api.authenticate("acme-manager", "demo1234")
    voucher_disputes.review(reviewer, case["id"], case["revision"],
                            outcome="eligible_to_apply", note="支持联系售后申请")
    resolved = mixed_voucher.agent_view(mixed_voucher.observe(user, second_sid, "V400"))
    assert resolved["next_step"] == "report_review" and resolved["required_fields"] == []
    assert "联系售后" in resolved["customer_next_action"] and "尚未退款" in resolved["customer_next_action"]
    assert "upstream" not in resolved


def test_withdrawal_tool_reports_success_separately_from_ticket_storage_state(mixed_api):
    api, _, sid, _ = mixed_api
    user = api.authenticate("acme-user", "demo1234")
    draft = refund_tasks.update(user, sid, expected_revision=0, order_id="A300", reason="尺码不合适", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id=sid, order_id="A300", reason="尺码不合适",
                                              task_revision=draft["revision"], approval_protocol="detached")
    config = {"configurable": {"thread_id": sid, "tenant": user.tenant_id, "user_id": user.username, "roles": ["customer"]}}
    result = json.loads(api.withdraw_refund_request.invoke({"approval_id": ticket["id"]}, config=config))
    assert result["ok"] is True and result["status"] == "withdrawn"
    assert tickets.get_ticket(ticket["id"])["status"] == tickets.REJECTED
    assert refund_tasks.get(user, sid)["status"] == "cancelled"
    duplicate = json.loads(api.withdraw_refund_request.invoke({"approval_id": ticket["id"]}, config=config))
    assert duplicate["ok"] is False and duplicate["code"] == "conflict"
