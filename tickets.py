"""退款工单的状态机与持久化。

为什么需要它
------------
之前退款单是内存字典 `{order_id: refund_id}`，有三个硬伤：
    1. 进程重启就没了
    2. 没有状态 —— 不知道这笔是待审批、已批准还是已执行
    3. 没有审计 —— 谁在什么时候批准了什么，查不到

状态机
------
    pending_approval ──approve──> approved ──抢占──> executing ──成功──> executed
            │                        │                     │
            └──────reject──> rejected └────失败─────────────> failed

业务规则用**部分唯一索引**表达：同一个订单可以有多次被拒绝的记录，
但同一租户、同一用户只能有一张 executed 的退款单。这比在代码里判重更可靠 ——
并发场景下代码判重有竞态，数据库约束没有。
"""

import sqlite3

import db
import threading
import uuid
from datetime import datetime, timedelta

from settings import DATA_DIR, ensure_data_dir

DB_PATH = DATA_DIR / "tickets.sqlite"

# 状态常量，避免各处硬编码字符串
PENDING = "pending_approval"
APPROVED = "approved"
EXECUTING = "executing"
EXECUTED = "executed"
REJECTED = "rejected"
FAILED = "failed"

# 合法状态迁移。任何不在表里的迁移都会被拒绝。
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    PENDING: {APPROVED, REJECTED},
    APPROVED: {EXECUTING, FAILED},
    EXECUTING: {EXECUTED, FAILED},
    EXECUTED: set(),
    REJECTED: set(),
    FAILED: set(),
}

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 后加的列。老库（CREATE TABLE 时代没有这些字段）需要 ALTER 补上，
# 否则 SELECT * 出来的 dict 缺键，下游 .get() 全是默认值，很难排查。
_ADDED_COLUMNS = {
    "risk_tier": "TEXT NOT NULL DEFAULT 'low'",
    "risk_reasons": "TEXT NOT NULL DEFAULT '[]'",
    "required_role": "TEXT NOT NULL DEFAULT 'customer'",
    "rule_version": "TEXT NOT NULL DEFAULT ''",
}


def _ensure_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(refund_tickets)")
    }
    for name, ddl in _ADDED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE refund_tickets ADD COLUMN {name} {ddl}")


def init_tickets_table() -> None:
    ensure_data_dir()
    # WAL 只需设一次，放初始化里——每次连接都设会在并发下冲突
    db.enable_wal(DB_PATH)
    db.hold(DB_PATH)      # 钉住 -wal/-shm，避免并发首写竞争
    with _lock, db.session(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refund_tickets (
                id          TEXT PRIMARY KEY,
                thread_id   TEXT NOT NULL,
                tenant_id   TEXT NOT NULL DEFAULT 'public',
                requester_id TEXT NOT NULL DEFAULT 'legacy',
                order_id    TEXT NOT NULL,
                amount_cents INTEGER NOT NULL DEFAULT 0,
                reason      TEXT,
                payload_hash TEXT NOT NULL DEFAULT '',
                status      TEXT NOT NULL,
                decision    TEXT,
                decision_reason TEXT,
                operator    TEXT,
                approved_by TEXT,
                approved_at TEXT,
                refund_id   TEXT,
                execution_id TEXT,
                last_error  TEXT,
                risk_tier   TEXT NOT NULL DEFAULT 'low',
                risk_reasons TEXT NOT NULL DEFAULT '[]',
                required_role TEXT NOT NULL DEFAULT 'customer',
                rule_version TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        # Existing local databases predate tenant ownership. Keep the migration
        # idempotent so upgrades do not require deleting audit history.
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(refund_tickets)").fetchall()
        }
        additions = {
            "tenant_id": "TEXT NOT NULL DEFAULT 'public'",
            "requester_id": "TEXT NOT NULL DEFAULT 'legacy'",
            "amount_cents": "INTEGER NOT NULL DEFAULT 0",
            "payload_hash": "TEXT NOT NULL DEFAULT ''",
            "decision_reason": "TEXT",
            "approved_by": "TEXT",
            "approved_at": "TEXT",
            "execution_id": "TEXT",
            "last_error": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE refund_tickets ADD COLUMN {name} {sql_type}"
                )
        # 业务规则：同一订单只能有一张「已执行」的退款单。
        # 用部分索引而不是唯一约束，是因为被拒绝的记录可以有多条。
        # The first project version keyed executed refunds by order_id alone,
        # which incorrectly coupled different tenants that happened to use the
        # same external order number. Recreate the index under owner scope.
        conn.execute("DROP INDEX IF EXISTS idx_ticket_executed_order")
        conn.execute(
            """
            CREATE UNIQUE INDEX idx_ticket_executed_order
            ON refund_tickets(tenant_id, requester_id, order_id)
            WHERE status = 'executed'
            """
        )
        _ensure_columns(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticket_thread ON refund_tickets(thread_id)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_active_action
            ON refund_tickets(
                tenant_id, requester_id, thread_id, order_id, payload_hash
            )
            WHERE status IN ('pending_approval', 'approved', 'executing')
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ticket_owner
            ON refund_tickets(tenant_id, requester_id, created_at DESC)
            """
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


def _transition(
    conn: sqlite3.Connection,
    ticket_id: str,
    target: str,
    *,
    expected: str,
    scope: dict | None = None,
    **fields,
) -> bool:
    """Atomically move a ticket from ``expected`` to ``target``.

    The UPDATE itself checks the source state. A read-then-write transition is
    unsafe across processes even when the caller happens to hold a thread lock.
    """
    row = conn.execute(
        "SELECT status FROM refund_tickets WHERE id = ?", (ticket_id,)
    ).fetchone()

    if row is None:
        return False

    current = row["status"]
    if current != expected or target not in ALLOWED_TRANSITIONS.get(current, set()):
        return False

    fields["status"] = target
    fields["updated_at"] = _now()

    assignments = ", ".join(f"{k} = ?" for k in fields)
    scope = scope or {}
    scope_sql = "".join(f" AND {key} = ?" for key in scope)
    params = list(fields.values()) + [ticket_id, expected] + list(scope.values())

    cursor = conn.execute(
        f"""
        UPDATE refund_tickets
        SET {assignments}
        WHERE id = ? AND status = ?{scope_sql}
        """,
        params,
    )
    return cursor.rowcount == 1


# ---------------------------------------------------------------- 对外接口


def create_ticket(
    thread_id: str,
    order_id: str,
    reason: str = "",
    operator: str = "local-user",
    *,
    tenant_id: str = "public",
    requester_id: str | None = None,
    payload_hash: str = "",
    amount_cents: int = 0,
    risk_tier: str = "low",
    risk_reasons: str = "[]",
    required_role: str = "customer",
    rule_version: str = "",
) -> dict:
    """Create one active ticket for an exact action payload.

    The partial unique index is the concurrency-safe deduplication layer. If a
    caller retries after a crash, it receives the existing active ticket instead
    of creating a second approval.
    """
    ticket_id = uuid.uuid4().hex[:16]
    now = _now()
    requester_id = requester_id or operator

    with _lock, db.session(DB_PATH) as conn:
        existing = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE tenant_id = ? AND requester_id = ? AND thread_id = ?
              AND order_id = ? AND payload_hash = ?
              AND status IN (?, ?, ?)
            ORDER BY created_at DESC LIMIT 1
            """,
            (
                tenant_id,
                requester_id,
                thread_id,
                order_id,
                payload_hash,
                PENDING,
                APPROVED,
                EXECUTING,
            ),
        ).fetchone()
        if existing:
            return _row_to_dict(existing)
        try:
            conn.execute(
                """
                INSERT INTO refund_tickets
                    (id, thread_id, tenant_id, requester_id, order_id,
                     amount_cents, reason, payload_hash, status, decision,
                     operator, refund_id, risk_tier, risk_reasons,
                     required_role, rule_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    thread_id,
                    tenant_id,
                    requester_id,
                    order_id,
                    amount_cents,
                    reason,
                    payload_hash,
                    PENDING,
                    operator,
                    risk_tier,
                    risk_reasons,
                    required_role,
                    rule_version,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            # A different process won the create race. Return its ticket so
            # both callers show the same approval id.
            raced = conn.execute(
                """
                SELECT * FROM refund_tickets
                WHERE tenant_id = ? AND requester_id = ? AND thread_id = ?
                  AND order_id = ? AND payload_hash = ?
                  AND status IN (?, ?, ?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (
                    tenant_id,
                    requester_id,
                    thread_id,
                    order_id,
                    payload_hash,
                    PENDING,
                    APPROVED,
                    EXECUTING,
                ),
            ).fetchone()
            if raced:
                return _row_to_dict(raced)
            raise
        row = conn.execute(
            "SELECT * FROM refund_tickets WHERE id = ?", (ticket_id,)
        ).fetchone()

    return _row_to_dict(row)


def approve_ticket(
    ticket_id: str,
    operator: str = "local-user",
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> bool:
    with _lock, db.session(DB_PATH) as conn:
        scope = {}
        if tenant_id is not None:
            scope["tenant_id"] = tenant_id
        if requester_id is not None:
            scope["requester_id"] = requester_id
        return _transition(
            conn,
            ticket_id,
            APPROVED,
            expected=PENDING,
            scope=scope,
            decision="approve",
            operator=operator,
            approved_by=operator,
            approved_at=_now(),
        )


def reject_ticket(
    ticket_id: str,
    reason: str = "",
    operator: str = "local-user",
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> bool:
    with _lock, db.session(DB_PATH) as conn:
        scope = {}
        if tenant_id is not None:
            scope["tenant_id"] = tenant_id
        if requester_id is not None:
            scope["requester_id"] = requester_id
        return _transition(
            conn,
            ticket_id,
            REJECTED,
            expected=PENDING,
            scope=scope,
            decision="reject",
            operator=operator,
            approved_by=operator,
            approved_at=_now(),
            decision_reason=reason or None,
        )


def mark_executed(
    ticket_id: str,
    refund_id: str,
    *,
    execution_id: str | None = None,
) -> bool:
    """工具真正执行成功后调用。

    ``ticket_id`` comes from the server-side approval record, never from model
    arguments or the user request.
    """
    with _lock, db.session(DB_PATH) as conn:
        return _transition(
            conn,
            ticket_id,
            EXECUTED,
            expected=EXECUTING,
            refund_id=refund_id,
            execution_id=execution_id,
            last_error=None,
        )


def mark_failed(ticket_id: str, error: str = "") -> bool:
    with _lock, db.session(DB_PATH) as conn:
        cursor = conn.execute(
            """
            UPDATE refund_tickets
            SET status = ?, last_error = ?, updated_at = ?
            WHERE id = ? AND status IN (?, ?)
            """,
            (
                FAILED,
                error or "退款执行失败",
                _now(),
                ticket_id,
                APPROVED,
                EXECUTING,
            ),
        )
        return cursor.rowcount == 1


def find_pending(
    thread_id: str,
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> dict | None:
    """取该会话最近一张待审批工单。"""
    clauses = ["thread_id = ?", "status = ?"]
    params: list[str] = [thread_id, PENDING]
    if tenant_id is not None:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if requester_id is not None:
        clauses.append("requester_id = ?")
        params.append(requester_id)
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT * FROM refund_tickets
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC LIMIT 1
            """,
            params,
        ).fetchone()
    return _row_to_dict(row) if row else None


def has_executed(
    order_id: str,
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> bool:
    """订单是否已有执行成功的退款单 —— 幂等的最终依据。"""
    return (
        find_executed(
            order_id, tenant_id=tenant_id, requester_id=requester_id
        )
        is not None
    )


def find_executed(
    order_id: str,
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> dict | None:
    """取该订单执行成功的那张退款单。"""
    clauses = ["order_id = ?", "status = ?"]
    params: list[str] = [order_id, EXECUTED]
    if tenant_id is not None:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if requester_id is not None:
        clauses.append("requester_id = ?")
        params.append(requester_id)
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT * FROM refund_tickets
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC LIMIT 1
            """,
            params,
        ).fetchone()
    return _row_to_dict(row) if row else None


def count_recent_executed(
    *,
    tenant_id: str,
    requester_id: str,
    hours: int = 24,
) -> int:
    """该用户近 N 小时内执行成功的退款笔数。

    用于风险规则 R3（异常频次）。按 updated_at 算——那是工单进入 executed
    的时间，比 created_at 更贴近"实际退款发生的时间"。
    """
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM refund_tickets
            WHERE tenant_id = ? AND requester_id = ?
              AND status = ? AND updated_at >= ?
            """,
            (tenant_id, requester_id, EXECUTED, since),
        ).fetchone()
    return int(row["n"]) if row else 0


def list_tickets(
    limit: int = 50,
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> list[dict]:
    clauses = []
    params: list[str | int] = []
    if tenant_id is not None:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if requester_id is not None:
        clauses.append("requester_id = ?")
        params.append(requester_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    with _lock, db.session(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM refund_tickets
            {where}
            ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_ticket(
    ticket_id: str,
    *,
    tenant_id: str | None = None,
    requester_id: str | None = None,
) -> dict | None:
    clauses = ["id = ?"]
    params: list[str] = [ticket_id]
    if tenant_id is not None:
        clauses.append("tenant_id = ?")
        params.append(tenant_id)
    if requester_id is not None:
        clauses.append("requester_id = ?")
        params.append(requester_id)
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            f"SELECT * FROM refund_tickets WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
    return _row_to_dict(row) if row else None


def find_pending_for_action(
    thread_id: str,
    *,
    tenant_id: str,
    requester_id: str,
    order_id: str,
    payload_hash: str,
) -> dict | None:
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE thread_id = ? AND tenant_id = ? AND requester_id = ?
              AND order_id = ? AND payload_hash = ?
              AND status IN (?, ?, ?)
            ORDER BY created_at DESC LIMIT 1
            """,
            (
                thread_id,
                tenant_id,
                requester_id,
                order_id,
                payload_hash,
                PENDING,
                APPROVED,
                EXECUTING,
            ),
        ).fetchone()
    return _row_to_dict(row) if row else None


def find_action_ticket(
    thread_id: str,
    *,
    tenant_id: str,
    requester_id: str,
    order_id: str,
    payload_hash: str,
) -> dict | None:
    """Find the newest ticket for an exact action, including terminal states."""
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE thread_id = ? AND tenant_id = ? AND requester_id = ?
              AND order_id = ? AND payload_hash = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (
                thread_id,
                tenant_id,
                requester_id,
                order_id,
                payload_hash,
            ),
        ).fetchone()
    return _row_to_dict(row) if row else None


def acquire_execution(
    thread_id: str,
    *,
    tenant_id: str,
    requester_id: str,
    order_id: str,
    payload_hash: str,
) -> dict | None:
    """Claim an approved action, or return an in-flight/completed retry.

    Only one process can move ``approved`` to ``executing`` because the
    transition is a conditional UPDATE checked by rowcount.
    """
    with _lock, db.session(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE thread_id = ? AND tenant_id = ? AND requester_id = ?
              AND order_id = ? AND payload_hash = ?
              AND status IN (?, ?, ?)
            ORDER BY created_at DESC LIMIT 1
            """,
            (
                thread_id,
                tenant_id,
                requester_id,
                order_id,
                payload_hash,
                APPROVED,
                EXECUTING,
                EXECUTED,
            ),
        ).fetchone()
        if row is None:
            return None
        ticket = _row_to_dict(row)
        if ticket["status"] == APPROVED:
            changed = _transition(
                conn,
                ticket["id"],
                EXECUTING,
                expected=APPROVED,
                scope={
                    "tenant_id": tenant_id,
                    "requester_id": requester_id,
                },
            )
            if not changed:
                return None
            refreshed = conn.execute(
                "SELECT * FROM refund_tickets WHERE id = ?", (ticket["id"],)
            ).fetchone()
            return _row_to_dict(refreshed)
        return ticket


def list_recoverable(limit: int = 100) -> list[dict]:
    """Approved/executing tickets that can be safely reconciled."""
    with _lock, db.session(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT * FROM refund_tickets
            WHERE status IN (?, ?)
            ORDER BY updated_at ASC
            LIMIT ?
            """,
            (APPROVED, EXECUTING, limit),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]
