"""清理脚本的边界：默认只预览、只删过期会话、孤儿线程要显式开启。"""
import sqlite3
from datetime import datetime, timedelta

import pytest

import db
import prune_checkpoints
import tickets

SESSION_FMT = "%Y-%m-%d %H:%M:%S"

SESSIONS_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'legacy', tenant_id TEXT NOT NULL DEFAULT 'public');
"""

MEMORY_SCHEMA = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT, type TEXT,
    checkpoint BLOB, metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id));
CREATE TABLE writes (
    thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, task_path TEXT NOT NULL DEFAULT '',
    idx INTEGER NOT NULL, channel TEXT NOT NULL, type TEXT, value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx));
"""


def stamp(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime(SESSION_FMT)


def build(tmp_path, tenant="acme"):
    sessions = tmp_path / "sessions.sqlite"
    memory = tmp_path / "chat_memory.sqlite"
    with sqlite3.connect(sessions) as conn:
        conn.executescript(SESSIONS_SCHEMA)
        conn.executemany(
            "INSERT INTO sessions (id,title,created_at,updated_at,tenant_id) VALUES (?,?,?,?,?)",
            [("old", "旧会话", stamp(40), stamp(40), tenant),
             ("fresh", "新会话", stamp(1), stamp(1), tenant)],
        )
    with sqlite3.connect(memory) as conn:
        conn.executescript(MEMORY_SCHEMA)
        for tid, n in (("old", 2), ("fresh", 1), ("orphan", 3)):
            for i in range(n):
                conn.execute(
                    "INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?,?)",
                    (tid, f"{tid}-{i}"),
                )
                conn.execute(
                    "INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel) "
                    "VALUES (?,?,?,?,?)",
                    (tid, f"{tid}-{i}", "t", i, "messages"),
                )
    return sessions, memory


def remaining(db, table):
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_dry_run_reports_without_deleting(tmp_path):
    sessions, memory = build(tmp_path)
    report = prune_checkpoints.prune(sessions, memory, days=30)
    assert report["applied"] is False
    assert [t["id"] for t in report["threads"]] == ["old"]
    assert report["checkpoints"] == 2 and report["writes"] == 2
    assert remaining(memory, "checkpoints") == 6
    assert remaining(sessions, "sessions") == 2


def test_apply_removes_only_expired_session(tmp_path):
    sessions, memory = build(tmp_path)
    report = prune_checkpoints.prune(sessions, memory, days=30, apply=True)
    assert report["sessions"] == 1
    with sqlite3.connect(memory) as conn:
        left = {r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert left == {"fresh", "orphan"}
    with sqlite3.connect(sessions) as conn:
        assert [r[0] for r in conn.execute("SELECT id FROM sessions")] == ["fresh"]


def test_orphan_threads_need_explicit_flag(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    sessions, memory = build(first)
    kept = prune_checkpoints.prune(sessions, memory, days=30, apply=True)
    assert kept["orphans"] == []
    assert remaining(memory, "checkpoints") == 4  # fresh 1 + orphan 3

    second = tmp_path / "second"
    second.mkdir()
    sessions, memory = build(second)
    report = prune_checkpoints.prune(sessions, memory, days=30, apply=True, orphans=True)
    assert [o["id"] for o in report["orphans"]] == ["orphan"]
    with sqlite3.connect(memory) as conn:
        left = {r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
    assert left == {"fresh"}


def test_tenant_filter_does_not_touch_other_tenants(tmp_path):
    sessions, memory = build(tmp_path, tenant="globex")
    report = prune_checkpoints.prune(sessions, memory, days=30, tenant="acme", apply=True)
    assert report["threads"] == []
    assert remaining(sessions, "sessions") == 2


def test_invalid_retention_is_rejected(tmp_path):
    sessions, memory = build(tmp_path)
    with pytest.raises(ValueError):
        prune_checkpoints.prune(sessions, memory, days=0, apply=True)


def test_retention_preserves_session_with_pending_ticket(tmp_path, monkeypatch):
    sessions, memory = build(tmp_path)
    path = tmp_path / "tickets.sqlite"
    monkeypatch.setattr(tickets, "DB_PATH", path)
    tickets.init_tickets_table()
    try:
        pending = tickets.create_ticket("old", "A100", tenant_id="acme", requester_id="legacy")
        report = prune_checkpoints.prune(sessions, memory, apply=True)
        assert report["protected"] == ["old"]
        assert remaining(sessions, "sessions") == 2
        assert tickets.reject_ticket(pending["id"])
        assert prune_checkpoints.prune(sessions, memory, apply=True)["sessions"] == 1
    finally:
        db.release_all()
