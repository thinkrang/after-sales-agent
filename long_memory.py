"""Explicit, typed preferences; never a source of payment authority."""
import time

import db
import tickets

ALLOWED = {
    "language": {"zh-CN", "en"},
    "response_style": {"concise", "detailed"},
    "contact_channel": {"chat", "email"},
}


def init_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS user_preferences (
        tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, key TEXT NOT NULL,
        value TEXT NOT NULL, source_turn TEXT NOT NULL, updated_at REAL NOT NULL,
        expires_at REAL NOT NULL, PRIMARY KEY(tenant_id,user_id,key))""")


def remember(user, key, value, *, source_turn="ui", ttl_days=90):
    if key not in ALLOWED or value not in ALLOWED[key]:
        raise ValueError("仅支持语言、回答风格和联系渠道偏好")
    if not 1 <= ttl_days <= 365:
        raise ValueError("记忆有效期必须为 1 到 365 天")
    now = time.time()
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("""INSERT INTO user_preferences VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(tenant_id,user_id,key) DO UPDATE SET value=excluded.value,
            source_turn=excluded.source_turn,updated_at=excluded.updated_at,expires_at=excluded.expires_at
        """, (user.tenant_id, user.username, key, value, source_turn, now, now + ttl_days*86400))
        # 清理放在写入路径：这里是写事务，顺手删掉该用户已过期的行，
        # 读取路径（每次模型调用都会走）就不必再抢写锁。
        conn.execute("DELETE FROM user_preferences WHERE tenant_id=? AND user_id=? AND expires_at<=?",
                     (user.tenant_id, user.username, now))
    return {"key": key, "value": value, "expires_at": now + ttl_days*86400}


def recall(user):
    """只读：过期偏好不注入，但不在读路径里删行。

    每次模型调用都会读偏好，如果读时顺手 DELETE，就等于把只读操作
    变成写事务——同一租户的并发对话会互相等锁。
    """
    now = time.time()
    with db.session(tickets.DB_PATH) as conn:
        return [dict(r) for r in conn.execute("""SELECT key,value,updated_at,expires_at
            FROM user_preferences WHERE tenant_id=? AND user_id=? AND expires_at>?
            ORDER BY key""", (user.tenant_id, user.username, now))]


def purge_expired(user=None, now=None) -> int:
    """物理清理过期偏好。user 为空时清理全部租户，供运维脚本调用。"""
    now = time.time() if now is None else now
    with db.session(tickets.DB_PATH) as conn:
        if user is None:
            return conn.execute(
                "DELETE FROM user_preferences WHERE expires_at<=?", (now,)
            ).rowcount
        return conn.execute(
            "DELETE FROM user_preferences WHERE tenant_id=? AND user_id=? AND expires_at<=?",
            (user.tenant_id, user.username, now),
        ).rowcount


def forget(user, key):
    with db.session(tickets.DB_PATH) as conn:
        return bool(conn.execute("DELETE FROM user_preferences WHERE tenant_id=? AND user_id=? AND key=?",
                                 (user.tenant_id, user.username, key)).rowcount)


def consent(message, key, value):
    """Conservative explicit opt-in gate on the actual latest user message."""
    words = {"zh-CN": ("中文",), "en": ("英文", "英语"), "concise": ("简洁", "简短"),
             "detailed": ("详细",), "chat": ("聊天", "在线"), "email": ("邮件",)}
    return (key in ALLOWED and value in ALLOWED[key]
            and message.strip().startswith(("请记住", "记住", "帮我记住", "请帮我记住"))
            and not any(w in message for w in ("不要", "别", "不用", "忘记", "不喜欢", "不是", "不想"))
            and any(w in message for w in words[value]))


def recent_cases(user, limit=3):
    """Read-only episodic recall from original scoped business records."""
    with db.session(tickets.DB_PATH) as conn:
        return [dict(r) for r in conn.execute("""
            SELECT id AS source_ticket,order_id,status,refund_id,amount_cents,updated_at
            FROM refund_tickets WHERE tenant_id=? AND requester_id=?
              AND status IN ('executed','rejected','failed')
            ORDER BY updated_at DESC,rowid DESC LIMIT ?
        """, (user.tenant_id, user.username, min(max(limit, 1), 5)))]
