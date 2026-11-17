"""Check the public offline replay routes against actual business state."""
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

import db
import orders
from scripts.run_voucher_demo import build_app
import tickets
import voucher_disputes
from test_task_http import task_api


def test_browser_replay_preserves_the_original_refund(task_api, monkeypatch):
    api, _, _, _ = task_api
    voucher_disputes.seed_demo()
    monkeypatch.setenv("VOUCHER_MERCHANT_KEYS", '{"acme":"demo-key"}')
    with TestClient(build_app(8765, "demo-key", mixed=True), base_url="http://127.0.0.1") as client:
        assert "脚本模型回放" in client.get("/").text
        assert client.get("/demo/mixed").json()["next"] == "start"
        assert client.post("/demo/mixed", json={"action": "arrival"}).status_code == 409

        def step(action):
            response = client.post("/demo/mixed", json={"action": action})
            assert response.status_code == 200, response.text
            return response.json()

        initial = step("start")
        assert initial["voucher"]["required_fields"] == ["visit_at"]
        assert initial["refund"]["status"] == tickets.PENDING
        ticket = tickets.get_ticket(initial["refund"]["ticket_id"])
        assert step("start") == initial

        arrived = step("arrival")
        assert arrived["voucher"]["decision"] == "guide_dispute"
        assert "arrival=" in arrived["voucher"]["help_url"]
        assert not arrived["voucher"].get("case_id")

        submitted = step("submit")
        assert submitted["voucher"]["case_status"] == "pending_review"
        assert client.post("/demo/mixed", json={"action": "refresh"}).status_code == 409

        customer = api.authenticate("acme-user", "demo1234")
        platform = next(row for row in voucher_disputes.inspect(customer, "V200")["evidence"]
                        if row["source"] == "platform")
        event = json.dumps({"event_id": "test-merchant-2", "voucher_id": "V200", "source_version": 2,
                            "record_id": "merchant-test", "status": "redeemed", "redeem_ref": platform["redeem_ref"],
                            "redeemed_at": platform["redeemed_at"]}).encode("utf-8")
        stamp = str(int(time.time()))
        signature = hmac.new(b"demo-key", f"acme.{stamp}.".encode() + event, hashlib.sha256).hexdigest()
        response = client.post("/voucher-evidence/acme/merchant", content=event, headers={
            "X-Voucher-Timestamp": stamp, "X-Voucher-Signature": signature, "Content-Type": "application/json"})
        assert response.json()["case_rechecked"] is True
        assert client.get("/demo/mixed").json()["voucher"]["needs_recheck"] is True

        final = step("refresh")
        assert final["voucher"]["reason_code"] == "matching_redemption"
        assert final["next"] is None and final["refund"] == initial["refund"]
        assert any(item["code"] == "conflict" for item in final["trace"][-1]["tools"])
        assert tickets.get_ticket(ticket["id"]) == ticket
        with db.session(orders.ORDERS_DB) as conn:
            assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 0
