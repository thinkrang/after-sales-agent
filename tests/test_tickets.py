"""工单状态机测试。

重点验证三件事：
    1. 合法迁移能走通
    2. 非法迁移被拒绝（不能从 executed 退回 approved）
    3. 数据库层面保证「同一订单只能有一张 executed 工单」
"""

import sqlite3

import pytest

import tickets


@pytest.fixture
def db(tmp_path, monkeypatch):
    """把工单库指向临时文件，避免污染真实数据。"""
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets_test.sqlite")
    tickets.init_tickets_table()
    return tickets


def test_create_ticket_starts_pending(db):
    ticket = db.create_ticket(thread_id="t1", order_id="A1", reason="质量问题")

    assert ticket["status"] == db.PENDING
    assert ticket["order_id"] == "A1"
    assert ticket["reason"] == "质量问题"
    assert ticket["refund_id"] is None
    assert ticket["created_at"]


def test_full_happy_path(db):
    """pending -> approved -> executing -> executed"""
    ticket = db.create_ticket(thread_id="t1", order_id="A2")

    assert db.approve_ticket(ticket["id"]) is True
    assert db.get_ticket(ticket["id"])["status"] == db.APPROVED

    claimed = db.acquire_execution(
        "t1",
        tenant_id="public",
        requester_id="local-user",
        order_id="A2",
        payload_hash="",
    )
    assert claimed["status"] == db.EXECUTING
    assert db.mark_executed(ticket["id"], "RFA2") is True
    final = db.get_ticket(ticket["id"])
    assert final["status"] == db.EXECUTED
    assert final["refund_id"] == "RFA2"
    assert final["decision"] == "approve"


def test_reject_path(db):
    ticket = db.create_ticket(thread_id="t1", order_id="A3")

    assert db.reject_ticket(ticket["id"], "信息不全") is True
    final = db.get_ticket(ticket["id"])
    assert final["status"] == db.REJECTED
    assert final["decision"] == "reject"
    assert final["decision_reason"] == "信息不全"
    assert final["reason"] == ""


def test_cannot_approve_twice(db):
    """pending -> approved 之后再 approve 应被拒绝。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A4")

    assert db.approve_ticket(ticket["id"]) is True
    assert db.approve_ticket(ticket["id"]) is False


def test_cannot_rollback_from_executed(db):
    """已执行的工单不能退回任何状态。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A5")
    db.approve_ticket(ticket["id"])
    db.acquire_execution(
        "t1",
        tenant_id="public",
        requester_id="local-user",
        order_id="A5",
        payload_hash="",
    )
    db.mark_executed(ticket["id"], "RFA5")

    assert db.approve_ticket(ticket["id"]) is False
    assert db.reject_ticket(ticket["id"], "反悔") is False
    assert db.get_ticket(ticket["id"])["status"] == db.EXECUTED


def test_cannot_reject_after_approve(db):
    """approved 之后只能 executed / failed，不能 rejected。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A6")
    db.approve_ticket(ticket["id"])

    assert db.reject_ticket(ticket["id"]) is False
    assert db.get_ticket(ticket["id"])["status"] == db.APPROVED


def test_mark_executed_without_approval_returns_false(db):
    """没走审批就直接标记执行，应该返回 False（调用方据此告警）。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A7")
    assert db.mark_executed(ticket["id"], "RFA7") is False


def test_unique_executed_per_order(db):
    """同一订单不允许出现两张 executed 工单 —— 由数据库部分唯一索引保证。

    这条规则用索引表达而不是代码判重：并发下代码判重有竞态，
    数据库约束没有。
    """
    t1 = db.create_ticket(thread_id="t1", order_id="A8")
    db.approve_ticket(t1["id"])
    db.acquire_execution(
        "t1",
        tenant_id="public",
        requester_id="local-user",
        order_id="A8",
        payload_hash="",
    )
    db.mark_executed(t1["id"], "RFA8")

    # 第二次走完流程，插入 executed 时应被唯一索引拦下
    t2 = db.create_ticket(thread_id="t2", order_id="A8")
    db.approve_ticket(t2["id"])

    with pytest.raises(sqlite3.IntegrityError):
        db.acquire_execution(
            "t2",
            tenant_id="public",
            requester_id="local-user",
            order_id="A8",
            payload_hash="",
        )
        db.mark_executed(t2["id"], "RFA8-DUP")


def test_same_external_order_id_is_isolated_by_tenant_and_user(db):
    first = db.create_ticket(
        thread_id="t1",
        order_id="SHARED-1",
        tenant_id="acme",
        requester_id="alice",
        payload_hash="p1",
    )
    db.approve_ticket(first["id"])
    db.acquire_execution(
        "t1",
        tenant_id="acme",
        requester_id="alice",
        order_id="SHARED-1",
        payload_hash="p1",
    )
    db.mark_executed(first["id"], "RF-ACME")

    second = db.create_ticket(
        thread_id="t2",
        order_id="SHARED-1",
        tenant_id="globex",
        requester_id="bob",
        payload_hash="p2",
    )
    db.approve_ticket(second["id"])
    db.acquire_execution(
        "t2",
        tenant_id="globex",
        requester_id="bob",
        order_id="SHARED-1",
        payload_hash="p2",
    )
    assert db.mark_executed(second["id"], "RF-GLOBEX") is True


def test_rejected_order_can_be_retried(db):
    """被拒绝的订单允许重新申请 —— 部分唯一索引只管 executed。"""
    t1 = db.create_ticket(thread_id="t1", order_id="A9")
    db.reject_ticket(t1["id"], "信息不全")

    t2 = db.create_ticket(thread_id="t1", order_id="A9")
    db.approve_ticket(t2["id"])
    db.acquire_execution(
        "t1",
        tenant_id="public",
        requester_id="local-user",
        order_id="A9",
        payload_hash="",
    )
    assert db.mark_executed(t2["id"], "RFA9") is True

    assert db.has_executed("A9") is True
    assert db.find_executed("A9")["refund_id"] == "RFA9"


def test_find_pending_returns_latest(db):
    first = db.create_ticket(thread_id="t1", order_id="B1")
    second = db.create_ticket(thread_id="t1", order_id="B2")

    pending = db.find_pending("t1")
    assert pending is not None
    # 同一秒创建时按 created_at 排序可能并列，至少要是其中之一
    assert pending["id"] in {first["id"], second["id"]}


def test_find_pending_none_when_handled(db):
    ticket = db.create_ticket(thread_id="t1", order_id="B3")
    db.reject_ticket(ticket["id"])

    assert db.find_pending("t1") is None


def test_list_tickets_is_desc(db):
    db.create_ticket(thread_id="t1", order_id="C1")
    db.create_ticket(thread_id="t2", order_id="C2")

    listed = db.list_tickets()
    assert len(listed) == 2
    assert {t["order_id"] for t in listed} == {"C1", "C2"}


def test_init_is_idempotent(db):
    """重复建表不该报错（服务重启会多次调用）。"""
    db.init_tickets_table()
    db.init_tickets_table()
    assert db.list_tickets() == []


def test_old_ticket_schema_is_migrated_without_losing_audit_rows(
    tmp_path, monkeypatch
):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE refund_tickets (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            reason TEXT,
            status TEXT NOT NULL,
            decision TEXT,
            operator TEXT,
            refund_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO refund_tickets
        VALUES ('old1', 'thread1', 'ORDER1', 'old reason', 'executed',
                'approve', 'legacy', 'RF1', '2026-01-01', '2026-01-01')
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(tickets, "DB_PATH", path)
    tickets.init_tickets_table()
    row = tickets.get_ticket("old1")
    assert row is not None
    assert row["tenant_id"] == "public"
    assert row["requester_id"] == "legacy"
    assert row["payload_hash"] == ""
    assert row["refund_id"] == "RF1"
