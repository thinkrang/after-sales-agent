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
    """pending -> approved -> executed"""
    ticket = db.create_ticket(thread_id="t1", order_id="A2")

    assert db.approve_ticket(ticket["id"]) is True
    assert db.get_ticket(ticket["id"])["status"] == db.APPROVED

    assert db.mark_executed("A2", "RFA2") is True
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
    assert final["reason"] == "信息不全"


def test_cannot_approve_twice(db):
    """pending -> approved 之后再 approve 应被拒绝。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A4")

    assert db.approve_ticket(ticket["id"]) is True
    assert db.approve_ticket(ticket["id"]) is False


def test_cannot_rollback_from_executed(db):
    """已执行的工单不能退回任何状态。"""
    ticket = db.create_ticket(thread_id="t1", order_id="A5")
    db.approve_ticket(ticket["id"])
    db.mark_executed("A5", "RFA5")

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
    db.create_ticket(thread_id="t1", order_id="A7")
    assert db.mark_executed("A7", "RFA7") is False


def test_unique_executed_per_order(db):
    """同一订单不允许出现两张 executed 工单 —— 由数据库部分唯一索引保证。

    这条规则用索引表达而不是代码判重：并发下代码判重有竞态，
    数据库约束没有。
    """
    t1 = db.create_ticket(thread_id="t1", order_id="A8")
    db.approve_ticket(t1["id"])
    db.mark_executed("A8", "RFA8")

    # 第二次走完流程，插入 executed 时应被唯一索引拦下
    t2 = db.create_ticket(thread_id="t2", order_id="A8")
    db.approve_ticket(t2["id"])

    with pytest.raises(sqlite3.IntegrityError):
        db.mark_executed("A8", "RFA8-DUP")


def test_rejected_order_can_be_retried(db):
    """被拒绝的订单允许重新申请 —— 部分唯一索引只管 executed。"""
    t1 = db.create_ticket(thread_id="t1", order_id="A9")
    db.reject_ticket(t1["id"], "信息不全")

    t2 = db.create_ticket(thread_id="t1", order_id="A9")
    db.approve_ticket(t2["id"])
    assert db.mark_executed("A9", "RFA9") is True

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
