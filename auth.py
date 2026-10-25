"""Authentication and tenant-aware authorization primitives.

The demo keeps users in SQLite so the project remains easy to run, but the
security boundary is no longer controlled by request-body fields:

- the server authenticates a user and issues a signed, expiring Bearer token;
- tenant_id and roles come from that token, never from ``ChatRequest``;
- passwords are stored as salted PBKDF2 hashes.

This is deliberately small rather than pretending to be a complete identity
provider. A production deployment should replace the user table with the
company's IdP, keep the same server-side authorization contract, and rotate
AUTH_SECRET through a secret manager.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import db
from settings import DATA_DIR, ensure_data_dir

USER_DB = DATA_DIR / "auth.sqlite"

TOKEN_TTL_SECONDS = 8 * 60 * 60
PBKDF2_ITERATIONS = 210_000
_DEFAULT_DEMO_SECRET = "dev-only-change-this-auth-secret"
_DEMO_PASSWORD = "demo1234"


@dataclass(frozen=True)
class User:
    username: str
    display_name: str
    tenant_id: str
    roles: frozenset[str]

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def has_role(self, role: str) -> bool:
        return self.is_admin or role in self.roles

    def can_access_tenant(self, tenant_id: str) -> bool:
        return self.is_admin or self.tenant_id == tenant_id


def _auth_secret() -> str:
    secret = os.getenv("AUTH_SECRET")
    if secret:
        return secret
    # A deterministic fallback keeps a fresh clone runnable in demo mode. The
    # app logs a warning at startup; production configuration must override it.
    return _DEFAULT_DEMO_SECRET


def using_demo_secret() -> bool:
    return not bool(os.getenv("AUTH_SECRET"))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str, *, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("ascii"),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), expected)
    except (TypeError, ValueError):
        return False


def _row_to_user(row: sqlite3.Row) -> User:
    roles = frozenset(filter(None, str(row["roles"] or "").split(",")))
    return User(
        username=row["username"],
        display_name=row["display_name"],
        tenant_id=row["tenant_id"],
        roles=roles,
    )


def init_auth_db(*, seed_demo_users: bool = True) -> None:
    ensure_data_dir()
    db.enable_wal(USER_DB)
    db.hold(USER_DB)
    with db.session(USER_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_users (
                username     TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                tenant_id    TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                roles        TEXT NOT NULL,
                active       INTEGER NOT NULL DEFAULT 1,
                created_at   TEXT NOT NULL
            )
            """
        )
        if seed_demo_users:
            _seed_demo_users(conn)


def _seed_demo_users(conn: sqlite3.Connection) -> None:
    demo_users = [
        (
            "acme-manager",
            "ACME 售后主管",
            "acme",
            "customer,approver,knowledge_admin",
        ),
        ("acme-user", "ACME 客户", "acme", "customer"),
        # 独立的审批人：四眼原则要求高风险单不能由发起人自己批，
        # 所以演示里需要一个和 acme-manager 不同的审批账号。
        ("acme-approver", "ACME 审批专员", "acme", "approver"),
        ("globex-user", "GLOBEX 客户", "globex", "customer"),
        ("platform-admin", "平台管理员", "public", "admin"),
    ]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for username, display_name, tenant_id, roles in demo_users:
        existing = conn.execute(
            "SELECT 1 FROM auth_users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO auth_users
                (username, display_name, tenant_id, password_hash, roles,
                 active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (
                username,
                display_name,
                tenant_id,
                hash_password(_DEMO_PASSWORD),
                roles,
                now,
            ),
        )


def authenticate(username: str, password: str) -> User | None:
    with db.session(USER_DB) as conn:
        row = conn.execute(
            """
            SELECT username, display_name, tenant_id, password_hash, roles, active
            FROM auth_users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    if row is None or not row["active"]:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)


def create_access_token(user: User, *, now: int | None = None) -> str:
    now = int(now if now is not None else time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user.username,
        "name": user.display_name,
        "tenant": user.tenant_id,
        "roles": sorted(user.roles),
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),
    }
    header_part = _b64encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    )
    payload_part = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    )
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(
        _auth_secret().encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    return f"{header_part}.{payload_part}.{_b64encode(signature)}"


def decode_access_token(token: str, *, now: int | None = None) -> User:
    try:
        header_part, payload_part, signature_part = token.split(".", 2)
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected = hmac.new(
            _auth_secret().encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        supplied = _b64decode(signature_part)
        if not hmac.compare_digest(expected, supplied):
            raise ValueError("bad signature")

        header = json.loads(_b64decode(header_part))
        payload = json.loads(_b64decode(payload_part))
        if header.get("alg") != "HS256":
            raise ValueError("unsupported algorithm")
        if int(payload.get("exp", 0)) <= int(now if now is not None else time.time()):
            raise ValueError("token expired")
        return User(
            username=str(payload["sub"]),
            display_name=str(payload.get("name") or payload["sub"]),
            tenant_id=str(payload["tenant"]),
            roles=frozenset(str(r) for r in payload.get("roles", [])),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("无效或已过期的访问令牌") from exc


_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer)
    ],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="需要登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(user: User, role: str) -> None:
    if not user.has_role(role):
        raise HTTPException(status_code=403, detail=f"需要 {role} 权限")
