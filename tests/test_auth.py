"""Authentication and session ownership regression tests."""

import time

import pytest

import auth
import db


def test_password_hash_round_trip():
    encoded = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", encoded)
    assert not auth.verify_password("wrong", encoded)


def test_access_token_round_trip_and_expiry():
    user = auth.User(
        username="alice",
        display_name="Alice",
        tenant_id="acme",
        roles=frozenset({"customer", "approver"}),
    )
    token = auth.create_access_token(user, now=1_000)
    decoded = auth.decode_access_token(token, now=1_001)
    assert decoded == user

    with pytest.raises(ValueError, match="过期"):
        auth.decode_access_token(
            token,
            now=1_000 + auth.TOKEN_TTL_SECONDS + 1,
        )


def test_tampered_token_is_rejected():
    user = auth.User("alice", "Alice", "acme", frozenset({"customer"}))
    token = auth.create_access_token(user)
    header, payload, signature = token.split(".")
    tampered = f"{header}.{payload[:-1]}A.{signature}"
    with pytest.raises(ValueError):
        auth.decode_access_token(tampered, now=int(time.time()))


def test_demo_login_is_tenant_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USER_DB", tmp_path / "auth.sqlite")
    auth.init_auth_db()
    try:
        user = auth.authenticate("acme-manager", "demo1234")
        assert user is not None
        assert user.tenant_id == "acme"
        assert "approver" in user.roles
        assert auth.authenticate("acme-manager", "wrong-password") is None
    finally:
        db.release_all()


def test_request_without_token_is_rejected(client):
    response = client.get(
        "/sessions",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert response.status_code == 401


def test_session_cannot_be_read_by_another_tenant(client, login_as):
    created = client.post("/sessions", json={})
    sid = created.json()["id"]
    try:
        response = client.get(
            f"/sessions/{sid}/messages",
            headers=login_as("globex-user"),
        )
        assert response.status_code == 404
    finally:
        client.delete(f"/sessions/{sid}")


def test_client_supplied_tenant_field_has_no_effect(client, fake_agent):
    """Tenant cannot be supplied by a client at all."""
    sid = client.post("/sessions", json={}).json()["id"]
    fake_agent.script = []
    try:
        response = client.post(
            "/chat/stream",
            json={
                "thread_id": sid,
                "message": "测试",
                "tenant": "globex",
            },
        )
        assert response.status_code == 422
        assert fake_agent.calls == []
    finally:
        client.delete(f"/sessions/{sid}")
