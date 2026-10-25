"""Small order repository and idempotent refund gateway for the demo workflow.

In a real deployment the methods that cross a transaction boundary would call
the order/payment service and retain the same idempotency-key contract. Keeping
the boundary explicit makes the Agent unable to invent an order or refund an
order that does not belong to the authenticated user.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta

import db
from auth import User
from settings import DATA_DIR, ensure_data_dir

ORDERS_DB = DATA_DIR / "orders.sqlite"


class OrderNotFound(ValueError):
    pass


class RefundDenied(ValueError):
    pass


def _now() -> datetime:
    return datetime.now()


def _fmt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def init_order_db(*, seed_demo: bool = True) -> None:
    ensure_data_dir()
    db.enable_wal(ORDERS_DB)
    db.hold(ORDERS_DB)
    with db.session(ORDERS_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id        TEXT PRIMARY KEY,
                tenant_id       TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                product_name    TEXT NOT NULL,
                amount_cents    INTEGER NOT NULL,
                status          TEXT NOT NULL,
                delivered_at    TEXT,
                refundable_until TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refund_executions (
                idempotency_key TEXT PRIMARY KEY,
                order_id        TEXT NOT NULL,
                tenant_id       TEXT NOT NULL,
                user_id         TEXT NOT NULL,
                amount_cents    INTEGER NOT NULL,
                refund_id       TEXT NOT NULL UNIQUE,
                created_at      TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_orders_owner
            ON orders(tenant_id, user_id, created_at DESC)
            """
        )
        if seed_demo:
            _seed_demo_orders(conn)


def _seed_demo_orders(conn: sqlite3.Connection) -> None:
    now = _now()
    samples = [
        ("A100", "acme", "acme-manager", "休闲卫衣", 12900, "delivered", 1),
        ("A200", "acme", "acme-manager", "运动长裤", 8900, "delivered", 2),
        ("A300", "acme", "acme-user", "纯棉短袖T恤", 5900, "delivered", 1),
        ("G100", "globex", "globex-user", "企业定制卫衣", 10900, "delivered", 1),
        # 风险分级演示数据：
        # A400 归 acme-user，金额 1280 元 → 触发 R1 大额，需 acme-manager 审批
        # A500 归 acme-manager，签收 6 天（距退款截止不足 24 小时）→ 触发 R2 边界政策，
        #      发起人自己批准会被四眼原则拒绝
        ("A400", "acme", "acme-user", "人体工学按摩椅", 128000, "delivered", 1),
        # A600 同样是 acme-user 的高价值订单，专供"完整会话闭环"验收用：
        # A400 会被审批流程退掉，闭环测试需要一笔没被碰过的单子。
        ("A600", "acme", "acme-user", "智能手表", 168000, "delivered", 1),
        ("A500", "acme", "acme-manager", "换季清仓套装", 29900, "delivered", 6),
        ("A900", "acme", "acme-manager", "已取消的订单", 9900, "cancelled", None),
    ]
    for (
        order_id,
        tenant_id,
        user_id,
        product_name,
        amount_cents,
        status,
        delivered_days_ago,
    ) in samples:
        delivered_at = (
            now - timedelta(days=delivered_days_ago)
            if delivered_days_ago is not None
            else None
        )
        refundable_until = (
            delivered_at + timedelta(days=7) if delivered_at is not None else None
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO orders
                (order_id, tenant_id, user_id, product_name, amount_cents,
                 status, delivered_at, refundable_until, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                tenant_id,
                user_id,
                product_name,
                amount_cents,
                status,
                _fmt(delivered_at) if delivered_at else None,
                _fmt(refundable_until) if refundable_until else None,
                _fmt(now),
                _fmt(now),
            ),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def get_order(user: User, order_id: str) -> dict:
    with db.session(ORDERS_DB) as conn:
        row = conn.execute(
            """
            SELECT * FROM orders
            WHERE order_id = ? AND tenant_id = ? AND user_id = ?
            """,
            (order_id, user.tenant_id, user.username),
        ).fetchone()
    if row is None:
        raise OrderNotFound("订单不存在，或当前账号无权访问")
    return _row_to_dict(row)


def list_orders(user: User) -> list[dict]:
    with db.session(ORDERS_DB) as conn:
        if user.is_admin:
            rows = conn.execute(
                "SELECT * FROM orders ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM orders
                WHERE tenant_id = ? AND user_id = ?
                ORDER BY created_at DESC
                """,
                (user.tenant_id, user.username),
            ).fetchall()
    return [_row_to_dict(row) for row in rows]


def refund_policy(order: dict) -> dict:
    """Return a deterministic eligibility decision for the current order."""
    if order["status"] == "refunded":
        return {
            "eligible": False,
            "reason": "该订单已经退款",
            "code": "already_refunded",
        }
    if order["status"] != "delivered":
        return {
            "eligible": False,
            "reason": f"订单状态为 {order['status']}，当前不能退款",
            "code": "invalid_order_status",
        }
    refundable_until = order.get("refundable_until")
    if refundable_until and datetime.strptime(
        refundable_until, "%Y-%m-%d %H:%M:%S"
    ) < _now():
        return {
            "eligible": False,
            "reason": "已超过七天无理由退款期限，请转人工售后",
            "code": "refund_window_expired",
        }
    return {
        "eligible": True,
        "reason": "订单属于当前用户，状态已签收且在退款期限内",
        "code": "eligible",
    }


def execute_refund(
    *,
    user: User,
    order_id: str,
    amount_cents: int,
    idempotency_key: str,
) -> dict:
    """Execute a refund at most once for a given idempotency key.

    The final idempotency guarantee is the primary key on
    ``refund_executions.idempotency_key``. A retry returns the original result,
    which covers a process crash after the downstream call but before the
    ticket state is updated.
    """
    conn = db.connect(ORDERS_DB)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM refund_executions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing:
            conn.commit()
            return _row_to_dict(existing)

        order = conn.execute(
            """
            SELECT * FROM orders
            WHERE order_id = ? AND tenant_id = ? AND user_id = ?
            """,
            (order_id, user.tenant_id, user.username),
        ).fetchone()
        if order is None:
            raise OrderNotFound("订单不存在，或当前账号无权访问")

        order_dict = _row_to_dict(order)
        decision = refund_policy(order_dict)
        if not decision["eligible"]:
            raise RefundDenied(decision["reason"])
        if int(order_dict["amount_cents"]) != int(amount_cents):
            raise RefundDenied("审批金额与订单金额不一致")

        refund_id = "RF" + uuid.uuid4().hex[:16].upper()
        now = _fmt(_now())
        conn.execute(
            """
            INSERT INTO refund_executions
                (idempotency_key, order_id, tenant_id, user_id,
                 amount_cents, refund_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idempotency_key,
                order_id,
                user.tenant_id,
                user.username,
                amount_cents,
                refund_id,
                now,
            ),
        )
        conn.execute(
            "UPDATE orders SET status = 'refunded', updated_at = ? WHERE order_id = ?",
            (now, order_id),
        )
        conn.commit()
        return {
            "idempotency_key": idempotency_key,
            "order_id": order_id,
            "tenant_id": user.tenant_id,
            "user_id": user.username,
            "amount_cents": amount_cents,
            "refund_id": refund_id,
            "created_at": now,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def find_refund(idempotency_key: str) -> dict | None:
    with db.session(ORDERS_DB) as conn:
        row = conn.execute(
            "SELECT * FROM refund_executions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
    return _row_to_dict(row) if row else None
