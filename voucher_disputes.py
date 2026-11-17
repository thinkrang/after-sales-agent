"""Voucher redemption evidence and dispute workflow for the demo marketplace."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import db
import tickets
from auth import User
from voucher_policy import ACTIVE_POLICY, evaluate_redemption, initial_dispute_result, parse_aware_datetime


class NotFound(ValueError):
    pass


class Forbidden(PermissionError):
    pass


class Conflict(ValueError):
    pass


REVIEW_OUTCOMES = {"eligible_to_apply", "redemption_confirmed", "insufficient_evidence"}
SERVICE_TZ = timezone(timedelta(hours=8))


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def init_schema(conn: sqlite3.Connection):
    conn.execute("""CREATE TABLE IF NOT EXISTS governance_vouchers (
        tenant_id TEXT NOT NULL, voucher_id TEXT NOT NULL, owner_id TEXT NOT NULL,
        title TEXT NOT NULL, expires_at TEXT NOT NULL, policy_version TEXT NOT NULL,
        PRIMARY KEY(tenant_id, voucher_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_evidence (
        tenant_id TEXT NOT NULL, voucher_id TEXT NOT NULL, source TEXT NOT NULL,
        record_id TEXT NOT NULL, status TEXT NOT NULL, redeem_ref TEXT, redeemed_at TEXT,
        PRIMARY KEY(tenant_id, voucher_id, source))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_disputes (
        id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, voucher_id TEXT NOT NULL,
        claim TEXT NOT NULL, revision INTEGER NOT NULL, status TEXT NOT NULL, outcome TEXT,
        assessment TEXT NOT NULL, reviewer_id TEXT, resolution_note TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE(tenant_id, voucher_id))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_voucher_review
        ON voucher_disputes(tenant_id, status, created_at, id)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_dispute_events (
        id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, revision INTEGER NOT NULL,
        actor_id TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_voucher_events
        ON voucher_dispute_events(case_id, id)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_visit_stores (
        tenant_id TEXT NOT NULL, voucher_id TEXT NOT NULL, store_code TEXT NOT NULL,
        store_name TEXT NOT NULL, starts_minute INTEGER NOT NULL, ends_minute INTEGER NOT NULL,
        booking_required INTEGER NOT NULL,
        PRIMARY KEY(tenant_id, voucher_id, store_code))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_bookings (
        tenant_id TEXT NOT NULL, voucher_id TEXT NOT NULL, store_code TEXT NOT NULL,
        booking_id TEXT NOT NULL, starts_at TEXT NOT NULL, ends_at TEXT NOT NULL, status TEXT NOT NULL,
        PRIMARY KEY(tenant_id, voucher_id, booking_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS voucher_merchant_events (
        tenant_id TEXT NOT NULL, event_id TEXT NOT NULL, voucher_id TEXT NOT NULL,
        source_version INTEGER NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
        received_at TEXT NOT NULL, PRIMARY KEY(tenant_id, event_id))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_voucher_merchant_version
        ON voucher_merchant_events(tenant_id, voucher_id, source_version DESC)
        WHERE status='applied'""")


def seed_demo():
    """Insert sample coupons once. Subsequent launches leave their evidence intact."""
    now = _now()
    future = (now + timedelta(days=30)).isoformat()
    past = (now - timedelta(days=1)).isoformat()
    redeemed = (now - timedelta(hours=1)).isoformat()
    samples = [
        ("acme", "V100", "acme-user", "双人咖啡券", future, "voucher-v1", "unused", "unused"),
        ("acme", "V200", "acme-user", "午餐套餐券", future, ACTIVE_POLICY, "redeemed", "unused"),
        ("acme", "V300", "acme-user", "周末体验券", future, ACTIVE_POLICY, "redeemed", "redeemed"),
        ("acme", "V400", "acme-user", "单人观影券", future, ACTIVE_POLICY, "unused", None),
        ("acme", "V500", "acme-user", "下午茶券", past, ACTIVE_POLICY, "unused", "unused"),
        ("globex", "V200", "globex-user", "烘焙体验券", future, ACTIVE_POLICY, "unused", "unused"),
    ]
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for tenant, voucher, owner, title, expiry, version, platform, merchant in samples:
            created = conn.execute("INSERT OR IGNORE INTO governance_vouchers VALUES (?, ?, ?, ?, ?, ?)",
                                   (tenant, voucher, owner, title, expiry, version)).rowcount
            if not created:
                continue
            for source, status in (("platform", platform), ("merchant", merchant)):
                if status is None:
                    continue
                ref = f"REDEEM-{voucher}" if status == "redeemed" else None
                timestamp = redeemed if ref else None
                conn.execute("INSERT INTO voucher_evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (tenant, voucher, source, f"{source}-{voucher}", status, ref, timestamp))
        # Seed visit terms independently so existing demo databases can be upgraded.
        conn.executemany("INSERT OR IGNORE INTO voucher_visit_stores VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("acme", "V100", "CENTRAL", "中心店", 600, 1200, 0),
            ("acme", "V100", "PARK", "公园店", 660, 1260, 1),
            ("acme", "V200", "CENTRAL", "中心店", 0, 1440, 0),
            ("acme", "V300", "PARK", "公园店", 600, 1260, 1),
            ("acme", "V400", "CENTRAL", "中心店", 0, 1440, 0),
            ("acme", "V500", "CENTRAL", "中心店", 600, 1200, 0),
            ("globex", "V200", "BAKERY", "烘焙门店", 600, 1200, 0),
        ])


def _voucher(conn, user: User, voucher_id: str):
    row = conn.execute("""SELECT voucher_id, title, expires_at, policy_version FROM governance_vouchers
        WHERE tenant_id=? AND owner_id=? AND voucher_id=?""",
        (user.tenant_id, user.username, voucher_id.strip().upper())).fetchone()
    if not row:
        raise NotFound("券不存在或不属于当前账号")
    return dict(row)


def list_vouchers(user: User) -> list[dict]:
    with db.session(tickets.DB_PATH) as conn:
        return [dict(row) for row in conn.execute("""SELECT voucher_id, title, expires_at FROM governance_vouchers
            WHERE tenant_id=? AND owner_id=? ORDER BY voucher_id LIMIT 100""",
            (user.tenant_id, user.username))]


def _assessment(conn, tenant, voucher, now):
    evidence = [dict(row) for row in conn.execute("""SELECT source, record_id, status, redeem_ref, redeemed_at
        FROM voucher_evidence WHERE tenant_id=? AND voucher_id=? ORDER BY source""",
        (tenant, voucher["voucher_id"]))]
    recommendation, reason_code = evaluate_redemption(voucher, evidence, now)
    return {"voucher": voucher, "evidence": evidence, "recommendation": recommendation, "reason_code": reason_code}


def inspect(user: User, voucher_id: str) -> dict:
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN")
        return _assessment(conn, user.tenant_id, _voucher(conn, user, voucher_id), _now())


def _hours(minute):
    hour, rest = divmod(minute, 60)
    return f"{hour:02d}:{rest:02d}"


def _visit(conn, tenant, assessment, store_code, visit_at):
    voucher = assessment["voucher"]
    rows = conn.execute("""SELECT store_code, store_name, starts_minute, ends_minute, booking_required
        FROM voucher_visit_stores WHERE tenant_id=? AND voucher_id=? ORDER BY store_code""",
        (tenant, voucher["voucher_id"])).fetchall()
    stores = [{"code": row["store_code"], "name": row["store_name"],
               "window": f'{_hours(row["starts_minute"])}–{_hours(row["ends_minute"])}',
               "booking_required": bool(row["booking_required"])} for row in rows]
    result = {"store_code": store_code, "visit_at": visit_at.isoformat(), "stores": stores, "booking": None}
    store = next((row for row in rows if row["store_code"] == store_code), None)
    if store and store["booking_required"]:
        bookings = conn.execute("""SELECT booking_id, starts_at, ends_at FROM voucher_bookings
            WHERE tenant_id=? AND voucher_id=? AND store_code=? AND status='confirmed'
            ORDER BY booking_id""", (tenant, voucher["voucher_id"], store_code))
        result["booking"] = next((dict(row) for row in bookings
                                  if parse_aware_datetime(row["starts_at"]) <= visit_at <
                                  parse_aware_datetime(row["ends_at"])), None)
    reason = assessment["reason_code"]
    if reason == "matching_redemption":
        return dict(result, next_step="review_redemption")
    if reason == "expired_voucher":
        return dict(result, next_step="check_expired_voucher")
    if reason != "unused_and_valid":
        return dict(result, next_step="submit_dispute")
    if visit_at >= parse_aware_datetime(voucher["expires_at"]):
        return dict(result, next_step="check_expired_voucher")
    if not store_code:
        return dict(result, next_step="choose_store")
    if not rows:
        return dict(result, next_step="verify_store_terms")
    if store is None:
        return dict(result, next_step="change_store")
    local_time = visit_at.astimezone(SERVICE_TZ)
    minute = local_time.hour * 60 + local_time.minute
    if not store["starts_minute"] <= minute < store["ends_minute"]:
        return dict(result, next_step="change_time")
    if store["booking_required"] and result["booking"] is None:
        return dict(result, next_step="confirm_booking")
    return dict(result, next_step="eligible_at_store")


def inspect_visit(user: User, voucher_id: str, store_code: str | None, visit_at: datetime | None) -> dict:
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN")
        now = _now()
        assessment = _assessment(conn, user.tenant_id, _voucher(conn, user, voucher_id), now)
        return dict(assessment, visit=_visit(conn, user.tenant_id, assessment, store_code, visit_at or now))


def _item(row):
    item = dict(row)
    item["assessment"] = json.loads(item["assessment"])
    return item


def _event(conn, item, actor_id, kind, detail, now):
    conn.execute("""INSERT INTO voucher_dispute_events(case_id, revision, actor_id, kind, detail, created_at)
        VALUES (?, ?, ?, ?, ?, ?)""", (item["id"], item["revision"], actor_id, kind,
                                     json.dumps(detail, ensure_ascii=False), now.isoformat()))


def submit(user: User, voucher_id: str, claim: str, *,
           store_code: str | None = None, visit_at: datetime | None = None) -> dict:
    claim = claim.strip()
    if not 1 <= len(claim) <= 500:
        raise ValueError("请填写 1 到 500 字的争议说明")
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        voucher = _voucher(conn, user, voucher_id)
        existing = conn.execute("SELECT * FROM voucher_disputes WHERE tenant_id=? AND voucher_id=?",
                                (user.tenant_id, voucher["voucher_id"])).fetchone()
        if existing:
            if existing["owner_id"] != user.username:
                raise Conflict("该券已有历史争议，请联系人工核查")
            return _item(existing)
        now = _now()
        assessment = _assessment(conn, user.tenant_id, voucher, now)
        visit = _visit(conn, user.tenant_id, assessment, store_code, visit_at or now) if store_code else None
        status, outcome = initial_dispute_result(assessment["recommendation"], on_site=visit is not None)
        case_id = uuid.uuid4().hex
        conn.execute("""INSERT INTO voucher_disputes
            (id, tenant_id, owner_id, voucher_id, claim, revision, status, outcome, assessment, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)""",
            (case_id, user.tenant_id, user.username, voucher["voucher_id"], claim, status,
             outcome, json.dumps(assessment, ensure_ascii=False), now.isoformat(), now.isoformat()))
        item = _item(conn.execute("SELECT * FROM voucher_disputes WHERE id=?", (case_id,)).fetchone())
        event = {"assessment": assessment, "status": status, "outcome": outcome}
        if visit:
            event["visit"] = visit
        _event(conn, item, user.username, "case_submitted", event, now)
        return item


def _get(conn, user: User, case_id: str):
    row = conn.execute("SELECT * FROM voucher_disputes WHERE id=? AND tenant_id=?",
                       (case_id, user.tenant_id)).fetchone()
    if not row or (row["owner_id"] != user.username and not user.has_role("approver")):
        raise NotFound("争议单不存在")
    return _item(row)


def detail(user: User, case_id: str) -> dict:
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN")
        item = _get(conn, user, case_id)
        item["events"] = [dict(row) for row in conn.execute("""SELECT revision, actor_id, kind, detail, created_at
            FROM voucher_dispute_events WHERE case_id=? ORDER BY id""", (case_id,))]
        for event in item["events"]:
            event["detail"] = json.loads(event["detail"])
        return item


def list_cases(user: User, *, queue: bool = False) -> list[dict]:
    if queue and not user.has_role("approver"):
        raise Forbidden("需要审批员权限")
    scope = "owner_id!=? AND status IN ('pending_review','pending_recheck')" if queue else "owner_id=?"
    order = "created_at ASC, id ASC" if queue else "created_at DESC, id DESC"
    with db.session(tickets.DB_PATH) as conn:
        return [_item(row) for row in conn.execute(f"""SELECT * FROM voucher_disputes
            WHERE tenant_id=? AND {scope} ORDER BY {order} LIMIT 100""",
            (user.tenant_id, user.username))]


def review(user: User, case_id: str, expected_revision: int, *,
           outcome: str | None = None, note: str = "") -> dict:
    """Refresh pending evidence, or record a human decision using that version."""
    if not user.has_role("approver"):
        raise Forbidden("需要审批员权限")
    note = note.strip()
    if outcome is not None and (outcome not in REVIEW_OUTCOMES or not 1 <= len(note) <= 500):
        raise ValueError("请选择有效结论，并填写 1 到 500 字的复核依据")
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        item = _get(conn, user, case_id)
        if item["owner_id"] == user.username:
            raise Forbidden("发起人不能复核自己的争议单")
        if item["status"] not in {"pending_review", "pending_recheck"} or item["revision"] != expected_revision:
            raise Conflict("争议单已更新，请重新打开")
        now = _now()
        owner = User(item["owner_id"], "", user.tenant_id, frozenset())
        voucher = _voucher(conn, owner, item["voucher_id"])
        assessment = _assessment(conn, user.tenant_id, voucher, now)
        if outcome is None:
            if item["assessment"] == assessment:
                return item
            item["assessment"] = assessment
            kind, event = "evidence_refreshed", {"assessment": assessment}
        else:
            if item["assessment"] != assessment:
                raise Conflict("券状态或证据已变化，请先重新核查再提交结论")
            item.update(status="resolved", outcome=outcome, reviewer_id=user.username, resolution_note=note)
            kind, event = "review_resolved", {"outcome": outcome, "note": note}
        _save_review(conn, item, user.username, kind, event, now)
        return item


def _save_review(conn, item, actor_id, kind, event, now):
    item["revision"] += 1
    item["updated_at"] = now.isoformat()
    conn.execute("""UPDATE voucher_disputes SET revision=?, status=?, outcome=?, assessment=?,
        reviewer_id=?, resolution_note=?, updated_at=? WHERE id=?""",
        (item["revision"], item["status"], item["outcome"], json.dumps(item["assessment"], ensure_ascii=False),
         item["reviewer_id"], item["resolution_note"], item["updated_at"], item["id"]))
    _event(conn, item, actor_id, kind, event, now)


def ingest_merchant_evidence(tenant: str, event_id: str, payload: dict) -> dict:
    """Atomically accept one source version and recheck just its affected case."""
    voucher_id, source_version = payload["voucher_id"], payload["source_version"]
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        prior = conn.execute("""SELECT payload FROM voucher_merchant_events
            WHERE tenant_id=? AND event_id=?""", (tenant, event_id)).fetchone()
        if prior:
            if prior["payload"] != normalized:
                raise Conflict("同一事件编号的内容发生变化")
            return {"status": "duplicate", "case_rechecked": False}
        voucher = conn.execute("""SELECT voucher_id, title, expires_at, policy_version FROM governance_vouchers
            WHERE tenant_id=? AND voucher_id=?""", (tenant, voucher_id)).fetchone()
        if not voucher:
            raise NotFound("券不存在")
        latest = conn.execute("""SELECT source_version, payload FROM voucher_merchant_events
            WHERE tenant_id=? AND voucher_id=? AND status='applied'
            ORDER BY source_version DESC LIMIT 1""", (tenant, voucher_id)).fetchone()
        if latest and source_version == latest["source_version"] and normalized != latest["payload"]:
            raise Conflict("同一来源版本的内容发生变化")
        skipped = latest and source_version <= latest["source_version"]
        status = "ignored_old_version" if skipped else "applied"
        now = _now()
        conn.execute("INSERT INTO voucher_merchant_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (tenant, event_id, voucher_id, source_version, normalized, status, now.isoformat()))
        if skipped:
            return {"status": status, "case_rechecked": False}
        conn.execute("""INSERT INTO voucher_evidence VALUES (?, ?, 'merchant', ?, ?, ?, ?)
            ON CONFLICT(tenant_id, voucher_id, source) DO UPDATE SET
                record_id=excluded.record_id, status=excluded.status,
                redeem_ref=excluded.redeem_ref, redeemed_at=excluded.redeemed_at""",
            (tenant, voucher_id, payload["record_id"], payload["status"], payload["redeem_ref"], payload["redeemed_at"]))
        row = conn.execute("SELECT * FROM voucher_disputes WHERE tenant_id=? AND voucher_id=?",
                           (tenant, voucher_id)).fetchone()
        if not row:
            return {"status": status, "case_rechecked": False}
        item = _item(row)
        assessment = _assessment(conn, tenant, dict(voucher), now)
        if item["assessment"] == assessment:
            return {"status": status, "case_rechecked": False}
        item["assessment"] = assessment
        event = {"assessment": assessment, "event_id": event_id, "source_version": source_version}
        if item["status"] == "resolved":
            event["previous_resolution"] = {key: item[key] for key in ("outcome", "reviewer_id", "resolution_note")}
            item.update(status="pending_recheck", outcome=None, reviewer_id=None, resolution_note=None)
        _save_review(conn, item, "merchant", "merchant_evidence_rechecked", event, now)
        return {"status": status, "case_rechecked": True}
