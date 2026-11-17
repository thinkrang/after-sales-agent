"""Authenticated HTTP, rule boundaries and concurrent review using real SQLite."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
import tickets
import voucher_disputes as disputes
from auth import User, create_access_token
from voucher_api import router

CUSTOMER = User("acme-user", "客户", "acme", frozenset({"customer"}))
REVIEWER = User("acme-manager", "主管", "acme", frozenset({"approver"}))
REVIEWER_2 = User("acme-approver", "审批员", "acme", frozenset({"approver"}))
OUTSIDER = User("globex-user", "外租户", "globex", frozenset({"customer", "approver"}))


@pytest.fixture
def voucher_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    tickets.init_tickets_table()
    disputes.seed_demo()
    yield
    db.release_all()


def headers(user):
    return {"Authorization": "Bearer " + create_access_token(user)}


def visit_url(voucher, store, arrival):
    return f"/vouchers/{voucher}/visit-check?" + urlencode({"store_code": store, "visit_at": arrival})


def send_event(client, event, tenant="acme", *, signing_tenant=None, key="acme-demo-key"):
    stamp = str(int(time.time()))
    payload = json.dumps(event).encode()
    signed = f"{signing_tenant or tenant}.{stamp}.".encode() + payload
    signature = hmac.new(key.encode(), signed, hashlib.sha256).hexdigest()
    return client.post(f"/voucher-evidence/{tenant}/merchant", content=payload, headers={
        "X-Voucher-Timestamp": stamp, "X-Voucher-Signature": signature,
    })


def test_visit_guidance_and_booking_belong_to_the_selected_store(voucher_db):
    app = FastAPI()
    app.include_router(router)
    daytime = disputes._now().astimezone(disputes.SERVICE_TZ).replace(hour=13)
    evening = daytime.replace(hour=20, minute=0, second=0)
    with TestClient(app, headers=headers(CUSTOMER)) as client:
        def step(store, when):
            response = client.get(visit_url("V100", store, when))
            assert response.status_code == 200, response.text
            return response.json()["visit"]["next_step"]

        assert step("OTHER", daytime) == "change_store"
        assert step("CENTRAL", evening) == "change_time"
        assert step("CENTRAL", daytime) == "eligible_at_store"
        assert step("PARK", daytime) == "confirm_booking"
        with db.session(tickets.DB_PATH) as conn:
            conn.execute("INSERT INTO voucher_bookings VALUES (?, ?, ?, ?, ?, ?, ?)",
                         ("acme", "V100", "PARK", "BOOK-100", (daytime - timedelta(minutes=10)).isoformat(),
                          (daytime + timedelta(minutes=30)).isoformat(), "confirmed"))
        assert step("PARK", daytime) == "eligible_at_store"
        assert step("PARK", daytime + timedelta(minutes=30)) == "confirm_booking"
        assert client.get(visit_url("V100", "PARK", "2026-01-01T13:00")).status_code == 422
        assert client.get(visit_url("V100", "PARK", daytime), headers=headers(OUTSIDER)).status_code == 404
        assert client.get("/voucher-disputes").json() == {"cases": []}
        response = client.post("/voucher-disputes", json={
            "voucher_id": "V100", "store_code": "PARK", "visit_at": daytime.isoformat(), "claim": "店员无法接待",
        })
        assert response.json()["status"] == "pending_review"
        detail = client.get("/voucher-disputes/" + response.json()["id"]).json()
        assert detail["events"][0]["detail"]["visit"]["booking"]["booking_id"] == "BOOK-100"


def test_late_merchant_event_rechecks_only_affected_case(voucher_db, monkeypatch):
    monkeypatch.setenv("VOUCHER_MERCHANT_KEYS", '{"acme":"acme-demo-key","globex":"globex-demo-key"}')
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        buyer, reviewer = headers(CUSTOMER), headers(REVIEWER)
        when = disputes._now().isoformat()
        at_store = client.get(visit_url("V200", "CENTRAL", when), headers=buyer).json()
        assert at_store["visit"]["next_step"] == "submit_dispute"
        assert client.get("/voucher-disputes", headers=buyer).json() == {"cases": []}
        case = client.post("/voucher-disputes", headers=buyer, json={
            "voucher_id": "V200", "store_code": "CENTRAL", "visit_at": when,
            "claim": "我已到店，但这张券用不了",
        }).json()
        unrelated = disputes.submit(CUSTOMER, "V400", "商家查不到记录")
        same_number_other_tenant = disputes.submit(OUTSIDER, "V200", "核对我的券")
        platform = next(row for row in at_store["evidence"] if row["source"] == "platform")
        event = {"event_id": "late-v2", "voucher_id": "V200", "source_version": 2, "record_id": "merchant-200-corrected",
                 "status": "redeemed", "redeem_ref": platform["redeem_ref"], "redeemed_at": platform["redeemed_at"]}
        assert send_event(client, event, tenant="globex", signing_tenant="acme", key="globex-demo-key").status_code == 401
        assert send_event(client, event, key="wrong").status_code == 401
        assert send_event(client, event).json() == {"status": "applied", "case_rechecked": True}
        assert send_event(client, event).json() == {"status": "duplicate", "case_rechecked": False}
        assert send_event(client, dict(event, record_id="tampered")).status_code == 409
        assert send_event(client, dict(event, event_id="same-version", record_id="different")).status_code == 409
        older = {"event_id": "late-v1", "voucher_id": "V200", "source_version": 1,
                 "record_id": "merchant-200-old", "status": "unused"}
        assert send_event(client, older).json() == {"status": "ignored_old_version", "case_rechecked": False}
        assert disputes.detail(CUSTOMER, unrelated["id"])["revision"] == 1
        assert disputes.detail(OUTSIDER, same_number_other_tenant["id"])["revision"] == 1
        path = "/voucher-disputes/" + case["id"]
        decision = {"expected_revision": 1, "outcome": "redemption_confirmed", "note": "已复核补传流水"}
        assert client.post(path + "/decision", headers=reviewer, json=decision).status_code == 409
        detail = client.get(path, headers=buyer).json()
        assert detail["revision"] == 2 and detail["assessment"]["reason_code"] == "matching_redemption"
        assert detail["events"][0]["detail"]["visit"]["store_code"] == "CENTRAL"
        assert [e["kind"] for e in detail["events"]] == ["case_submitted", "merchant_evidence_rechecked"]
        assert client.post(path + "/decision", headers=reviewer, json=dict(decision, expected_revision=2)).status_code == 200
        assert client.get(path, headers=buyer).json()["outcome"] == "redemption_confirmed"
        assert send_event(client, dict(older, event_id="corrected-v3", source_version=3)).json()["case_rechecked"]
        reopened = client.get(path, headers=buyer).json()
        assert reopened["status"] == "pending_recheck" and reopened["outcome"] is None
        assert reopened["events"][3]["detail"]["previous_resolution"]["outcome"] == "redemption_confirmed"
        assert reopened["events"][2]["detail"]["outcome"] == "redemption_confirmed"
        queue = client.get("/voucher-disputes?queue=true", headers=reviewer).json()["cases"]
        assert {row["id"] for row in queue} == {case["id"], unrelated["id"]}


def test_http_workflow(voucher_db):
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        body = {"voucher_id": "V200", "claim": "没到店，但页面显示已核销。"}
        assert client.get("/vouchers").status_code == 401
        assert client.get("/vouchers/V100/assessment", headers=headers(OUTSIDER)).status_code == 404
        customer, reviewer = headers(CUSTOMER), headers(REVIEWER)
        assessment = client.get("/vouchers/V200/assessment", headers=customer).json()
        assert assessment["recommendation"] == "manual_review"
        assert assessment["reason_code"] == "status_mismatch"
        response = client.post("/voucher-disputes", headers=customer, json=body)
        assert response.status_code == 200, response.text
        case = response.json()
        assert case["status"] == "pending_review" and case["outcome"] is None
        assert client.post("/voucher-disputes", headers=customer, json=body).json()["id"] == case["id"]
        path = "/voucher-disputes/" + case["id"]
        decision = {"expected_revision": 1, "outcome": "eligible_to_apply", "note": "已向商家确认未消费"}
        assert client.get(path, headers=headers(OUTSIDER)).status_code == 404
        assert client.post(path + "/decision", headers=headers(OUTSIDER), json=decision).status_code == 404
        assert client.post(path + "/decision", headers=customer, json=decision).status_code == 403
        assert client.get("/voucher-disputes?queue=true", headers=customer).status_code == 403
        assert client.get("/voucher-disputes?queue=true", headers=reviewer).json()["cases"][0]["id"] == case["id"]
        resolved = client.post(path + "/decision", headers=reviewer, json=decision)
        assert resolved.status_code == 200, resolved.text
        assert client.post(path + "/decision", headers=reviewer, json=decision).status_code == 409
        assert client.get("/voucher-disputes?queue=true", headers=reviewer).json()["cases"] == []
        detail = client.get(path, headers=customer).json()
        assert detail["outcome"] == "eligible_to_apply" and detail["reviewer_id"] == REVIEWER.username
        assert [e["kind"] for e in detail["events"]] == ["case_submitted", "review_resolved"]
        assert detail["events"][0]["detail"]["assessment"]["voucher"]["policy_version"] == "voucher-v2"
        disputes.seed_demo()
        assert client.get(path, headers=customer).json() == detail
        assert client.get("/vouchers/V200/assessment", headers=headers(OUTSIDER)).json()["reason_code"] == "unused_and_valid"
    with db.session(tickets.DB_PATH) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_tickets").fetchone()[0] == 0


@pytest.mark.parametrize(("voucher", "reason", "status"), [
    ("V100", "unused_and_valid", "resolved"),
    ("V300", "matching_redemption", "pending_review"),
    ("V400", "missing_source", "pending_review"),
    ("V500", "expired_voucher", "pending_review"),
])
def test_disputed_or_incomplete_evidence_stays_reviewable(voucher_db, voucher, reason, status):
    case = disputes.submit(CUSTOMER, voucher, "我没有到店消费")
    assert case["assessment"]["reason_code"] == reason
    assert case["status"] == status
    assert case["outcome"] == ("eligible_to_apply" if status == "resolved" else None)


def test_versioned_policy_uses_the_vouchers_rule(voucher_db):
    platform = next(r for r in disputes.inspect(CUSTOMER, "V300")["evidence"] if r["source"] == "platform")
    merchant_time = datetime.fromisoformat(platform["redeemed_at"]) + timedelta(minutes=3)
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("""UPDATE voucher_evidence SET redeemed_at=? WHERE tenant_id='acme'
            AND voucher_id='V300' AND source='merchant'""", (merchant_time.isoformat(),))
    assert disputes.inspect(CUSTOMER, "V300")["reason_code"] == "redemption_time_mismatch"
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("UPDATE governance_vouchers SET policy_version='voucher-v1' WHERE tenant_id='acme' AND voucher_id='V300'")
    assert disputes.inspect(CUSTOMER, "V300")["reason_code"] == "matching_redemption"


def test_old_evidence_cannot_be_approved_and_refresh_is_audited(voucher_db):
    case = disputes.submit(CUSTOMER, "V200", "争议")
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("""UPDATE voucher_evidence SET status='unused', redeem_ref=NULL, redeemed_at=NULL
            WHERE tenant_id='acme' AND voucher_id='V200' AND source='platform'""")
    with pytest.raises(disputes.Conflict, match="证据已变化"):
        disputes.review(REVIEWER, case["id"], 1, outcome="redemption_confirmed", note="看了旧记录")
    refreshed = disputes.review(REVIEWER, case["id"], 1)
    assert refreshed["revision"] == 2 and refreshed["assessment"]["reason_code"] == "unused_and_valid"
    assert disputes.review(REVIEWER, case["id"], 2)["revision"] == 2
    with pytest.raises(disputes.Conflict):
        disputes.review(REVIEWER, case["id"], 1, outcome="eligible_to_apply", note="旧版本")
    disputes.review(REVIEWER, case["id"], 2, outcome="eligible_to_apply", note="双方均显示未核销")
    events = disputes.detail(CUSTOMER, case["id"])["events"]
    assert [e["revision"] for e in events] == [1, 2, 3]
    assert events[0]["detail"]["assessment"]["reason_code"] == "status_mismatch"
    assert events[1]["detail"]["assessment"]["reason_code"] == "unused_and_valid"


def test_review_is_same_tenant_non_owner_and_only_one_reviewer_wins(voucher_db):
    case = disputes.submit(CUSTOMER, "V400", "商家没有给核销凭证")
    customer_approver = User(CUSTOMER.username, "", CUSTOMER.tenant_id, frozenset({"approver"}))
    assert disputes.list_cases(customer_approver, queue=True) == []
    with pytest.raises(disputes.Forbidden):
        disputes.review(customer_approver, case["id"], 1, outcome="eligible_to_apply", note="本人")

    def attempt(actor):
        try:
            disputes.review(actor, case["id"], 1, outcome="insufficient_evidence", note="仍缺商家记录")
            return "resolved"
        except disputes.Conflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, (REVIEWER, REVIEWER_2))) == ["conflict", "resolved"]
    assert [e["kind"] for e in disputes.detail(CUSTOMER, case["id"])["events"]].count("review_resolved") == 1
