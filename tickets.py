"""退款工单的状态机与持久化。

为什么需要它
------------
之前退款单是内存字典 `{order_id: refund_id}`，有三个硬伤：
    1. 进程重启就没了
    2. 没有状态 —— 不知道这笔是待审批、已批准还是已执行
    3. 没有审计 —— 谁在什么时候批准了什么，查不到

状态机
------
    pending_approval ──approve──> approved ──执行成功──> executed
            │                        │
            └──────reject──> rejected └──执行失败──> failed

业务规则用**部分唯一索引**表达：同一个订单可以有多次被拒绝的记录，
但只能有一张 executed 的退款单。这比在代码里判重更可靠 ——
并发场景下代码判重有竞态，数据库约束没有。
"""

import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "tickets.sqlite"

# 状态常量，避免各处硬编码字符串
PENDING = "pending_approval"
APPROVED = "approved"
EXECUTED = "executed"
REJECTED = "rejected"
FAILED = "failed"

# 合法状态迁移。任何不在表里的迁移都会被拒绝。
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    PENDING: {APPROVED, REJECTED},
    APPROVED: {EXECUTED, FAILED},
    EXECUTED: set(),
    REJECTED: set(),
    FAILED: set(),
}

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_tickets_table() -> None:
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refund_tickets (
                id          TEXT PRIMARY KEY,
                thread_id   TEXT NOT NULL,
                order_id    TEXT NOT NULL,
                reason      TEXT,
                status      TEXT NOT NULL,
                decision    TEXT,
                operator    TEXT,
                refund_id   TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        # 业务规则：同一订单只能有一张「已执行」的退款单。
        # 用部分索引而不是唯一约束，是因为被拒绝的记录可以有多条。
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_executed_order
            ON refund_tickets(order_id) WHERE status = 'executed'
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticket_thread ON refund_tickets(thread_id)"
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _transition(conn: sqlite3.Connection, ticket_id: str, target: str, **fields) -> bool:
    """执行一次状态迁移，非法迁移直接拒绝。"""
    row = conn.execute(
        "SELECT status FROM refund_tickets WHERE id = ?", (ticket_id,)
    ).fetchone()

    if row is None:
        return False

    current = row["status"]
    if target not in ALLOWED_TRANSITIONS.get(current, set()):
        return False

    # 关键：status 本身也要写进去。
    # 之前漏了这一项，导致函数返回 True 但状态没变 ——
    # 这类 bug 最阴险的地方是「返回值正确」，只有查库才能发现。
    fields["status"] = target
    fields["updated_at"] = _now()

    assignments = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [ticket_id]

    conn.execute(
        f"UPDATE refund_tickets SET {assignments} WHERE id = ?",
        params,
    )
    return True


# ---------------------------------------------------------------- 对外接口


def create_ticket(
    thread_id: str,
    order_id: str,
    reason: str = "",
    operator: str = "local-user",
) -> dict:
    """在写操作被拦截时创建待审批工单。"""
    ticket_id = uuid.uuid4().hex[:16]
    now = _now()

    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO refund_tickets
                (id, thread_id, order_id, reason, status, decision, operator,
                 refund_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)
            """,
            (ticket_id, thread_id, order_id, reason, PENDING, operator, now, now),
        )
        row = conn.execute(
            "SELECT * FROM refund_tickets WHERE id = ?", (ticket_id,)
        ).fetchone()

    return _row_to_dict(row)


def approve_ticket(ticket_id: str, operator: str = "local-user") -> bool:
    with _lock, _connect() as conn:
        return _transition(
            conn, ticket_id, APPROVED, decision="approve", operator=operator
        )


def reject_ticket(ticket_id: str, reason: str = "", operator: str = "local-user") -> bool:
    with _lock, _connect() as conn:
        return _transition(
            conn, ticket_id, REJECTED, decision="reject", operator=operator,
            reason=reason or None,
        )


def mark_executed(order_id: str, refund_id: str) -> bool:
    """工具真正执行成功后调用。

    按 order_id 找最近一张 approved 的工单来标记，
    因为工具函数拿不到 ticket_id（参数由模型生成，不能塞私有字段）。
    """
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM refund_tickets
            WHERE order_id = ? AND status = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (order_id, APPROVED),
        ).fetchone()

        if row is None:
            return False
        return _transition(conn, row["id"], EXECUTED, refund_id=refund_id)


def mark_failed(order_id: str, error: str = "") -> bool:
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM refund_tickets
            WHERE order_id = ? AND status = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (order_id, APPROVED),
        ).fetchone()
        if row is None:
            return False
        return _transition(conn, row["id"], FAILED, reason=error or None)


def find_pending(thread_id: str) -> dict | None:
    """取该会话最近一张待审批工单。"""
    with _lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE thread_id = ? AND status = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (thread_id, PENDING),
        ).fetchone()
    return _row_to_dict(row) if row else None


def has_executed(order_id: str) -> bool:
    """订单是否已有执行成功的退款单 —— 幂等的最终依据。"""
    return find_executed(order_id) is not None


def find_executed(order_id: str) -> dict | None:
    """取该订单执行成功的那张退款单。"""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM refund_tickets WHERE order_id = ? AND status = ? LIMIT 1",
            (order_id, EXECUTED),
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_tickets(limit: int = 50) -> list[dict]:
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM refund_tickets ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_ticket(ticket_id: str) -> dict | None:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM refund_tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None
