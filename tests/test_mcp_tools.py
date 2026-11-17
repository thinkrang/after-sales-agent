"""MCP 工具的鉴权边界与写操作语义（不依赖 MCP SDK）。"""
import json

import pytest

import auth
import db
import mcp_tools
import orders
import tickets


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    monkeypatch.setattr(auth, "USER_DB", tmp_path / "auth.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    auth.init_auth_db(seed_demo_users=True)
    return tmp_path


def token_for(username: str) -> str:
    user = auth.authenticate(username, "demo1234")
    assert user is not None
    return auth.create_access_token(user)


def payload(raw: str) -> dict:
    return json.loads(raw)


def test_missing_or_malformed_token_is_rejected(env):
    with pytest.raises(PermissionError):
        mcp_tools.user_from_headers(None)
    with pytest.raises(PermissionError):
        mcp_tools.user_from_headers({"authorization": "Basic abc"})
    with pytest.raises(PermissionError):
        mcp_tools.user_from_headers({"authorization": "Bearer not-a-token"})


def test_tenant_comes_from_token_not_from_client(env):
    user = mcp_tools.user_from_headers({"Authorization": f"Bearer {token_for('acme-manager')}"})
    assert user.username == "acme-manager"
    assert user.tenant_id == "acme"


def test_order_lookup_is_scoped_to_token_tenant(env):
    globex = mcp_tools.user_from_headers({"authorization": f"Bearer {token_for('globex-user')}"})
    data = payload(mcp_tools.get_order_info(globex, "A100"))
    assert data["ok"] is False
    assert data["code"] == "not_found"
    # 找不到就是找不到：换号重试只会撞上循环边界。
    assert data["retryable"] is False


def test_prepare_refund_creates_pending_ticket_without_executing(env):
    user = mcp_tools.user_from_headers({"authorization": f"Bearer {token_for('acme-manager')}"})
    data = payload(mcp_tools.prepare_refund(user, "A100", "质量问题"))
    assert data["ok"] is True
    assert data["status"] == "pending_approval"
    assert data["approval_id"]

    with db.session(orders.ORDERS_DB) as conn:
        executed = conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0]
    assert executed == 0, "MCP 写操作不能绕过审批直接退款"

    with db.session(tickets.DB_PATH) as conn:
        status = conn.execute("SELECT status FROM refund_tickets WHERE id=?",
                              (data["approval_id"],)).fetchone()[0]
    assert status == "pending_approval"


def test_unknown_order_is_reported_as_not_found(env):
    user = mcp_tools.user_from_headers({"authorization": f"Bearer {token_for('acme-manager')}"})
    data = payload(mcp_tools.prepare_refund(user, "A999", "质量问题"))
    assert data["code"] == "not_found"
    assert data["retryable"] is False


def test_default_thread_is_stable_and_url_safe(env):
    user = mcp_tools.user_from_headers({"authorization": f"Bearer {token_for('acme-manager')}"})
    assert mcp_tools.default_thread(user) == mcp_tools.default_thread(user)
    assert mcp_tools.default_thread(user) == "mcp-acme-manager"


def test_two_mcp_requests_keep_independent_pending_approvals(env):
    user = auth.authenticate("acme-manager", "demo1234")
    first = payload(mcp_tools.prepare_refund(user, "A100", "质量问题"))
    second = payload(mcp_tools.prepare_refund(user, "A200", "尺码不合适"))
    assert first["task_id"] != second["task_id"]
    assert tickets.get_ticket(first["approval_id"])["status"] == "pending_approval"
    assert payload(mcp_tools.prepare_refund(user, "A100", "质量问题"))["approval_id"] == first["approval_id"]
    assert len(payload(mcp_tools.get_refund_task(user))["tasks"]) == 2
    changed = payload(mcp_tools.prepare_refund(user, "A100", "包装破损", first["task_id"]))
    assert changed["code"] == "conflict"
    changed = payload(mcp_tools.prepare_refund(user, "A100", "包装破损", first["task_id"], first["task_revision"]))
    assert changed["ok"] and changed["approval_id"] != first["approval_id"]
    assert tickets.get_ticket(second["approval_id"])["status"] == "pending_approval"
