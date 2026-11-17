"""The voucher branch of a mixed request. Refunds use their existing workflow."""
import json
import re
from datetime import timezone
from urllib.parse import urlencode

import db
import tickets
import voucher_disputes as vouchers
from voucher_policy import parse_aware_datetime


INSTRUCTIONS = (
    "同一请求同时查券和处理普通订单退款时，使用 observe_voucher_branch 建立独立券分支；"
    "普通订单仍走原退款流程，可以先推进到相应确认或审批，不等待券。"
    "退款任务快照为空表示这笔申请尚未登记；从本轮用户原话提取已给的订单编号和原因。"
    "明确给出的订单编号（如字母加数字）先用 get_order_info 校验，查不到或指代不明才追问。"
    "标明是券的编号只给券工具，不当作普通订单查询。"
    "按退款的 required_role 说明下一步：customer 必须由本人在确认卡片操作，approver 等主管，只有 executed 才表示已退款。"
    "不要使用普通退款工具处理券本身。继续券分支时先 get_voucher_branch，证据过期或用户补充时再 observe，"
    "带当前 revision，不要重复提交已知的到店资料。"
    "券分支尚未建立且用户已给券号时，先用券号观察事实。门店或时间不明可以留空，观察后再决定是否需要追问。"
    "读取工具返回的 decision；已记录就直接使用，待选择时按 next_step 和 revision 调用 choose_voucher_step。"
    "仅追问 required_fields。"
    "券的核查结果与人工复核结论分别说明。guide_dispute 和查看进展用返回链接，由用户确认提交。"
)


class StepConflict(vouchers.Conflict):
    def __init__(self, state):
        super().__init__("该步骤不符合当前证据；按返回的 next_step 继续")
        self.state = state


def init_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS mixed_voucher_branches (
        tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
        voucher_id TEXT NOT NULL, store_code TEXT, visit_at TEXT, revision INTEGER NOT NULL,
        upstream TEXT NOT NULL, facts TEXT NOT NULL, decision TEXT,
        PRIMARY KEY(tenant_id, owner_id, conversation_id))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS mixed_voucher_events (
        id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, owner_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
        revision INTEGER NOT NULL, request_id TEXT NOT NULL, action TEXT NOT NULL, upstream TEXT NOT NULL)""")


def _scope(user, sid):
    if not user.tenant_id or not user.username or not sid:
        raise vouchers.Conflict("缺少已认证的会话")
    return user.tenant_id, user.username, sid


def _read(conn, scope):
    row = conn.execute("""SELECT voucher_id, store_code, visit_at, revision, upstream, facts, decision
        FROM mixed_voucher_branches WHERE tenant_id=? AND owner_id=? AND conversation_id=?""", scope).fetchone()
    if not row:
        return None
    return dict(row, upstream=json.loads(row["upstream"]), facts=json.loads(row["facts"]))


def _view(row, *, stale=False):
    query = {"voucher": row["voucher_id"]}
    if row["store_code"] and row["visit_at"]:
        query.update(store=row["store_code"], arrival=row["visit_at"])
    return dict(row, facts=None if stale else row["facts"], decision=None if stale else row["decision"],
                needs_recheck=stale, help_url="/voucher-center?" + urlencode(query))


def get(user, sid):
    scope = _scope(user, sid)
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN")
        row = _read(conn, scope)
        if row:
            _, _, version = vouchers.workflow_source(conn, user, row["voucher_id"])
            return _view(row, stale=version != row["upstream"])
        return None


def observe(user, sid, voucher_id, *, store_code=None, visit_at=None, expected_revision=None, request_id=""):
    scope = _scope(user, sid)
    voucher_id = voucher_id.strip().upper()
    store_code = store_code.strip().upper() if store_code else None
    if not 1 <= len(voucher_id) <= 64 or (store_code and not re.fullmatch(r"[A-Z0-9_-]{1,32}", store_code)):
        raise ValueError("请核对券号或门店编号")
    if visit_at is not None:
        visit_at = parse_aware_datetime(visit_at).astimezone(timezone.utc).isoformat()
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if tickets.conversation_closed(conn, *scope):
            raise vouchers.Conflict("会话已关闭")
        old = _read(conn, scope)
        if old and old["voucher_id"] != voucher_id:
            raise vouchers.Conflict("当前券分支已绑定其他券")
        if old:
            store_code = store_code or old["store_code"]
            visit_at = visit_at or old["visit_at"]
        revision = old["revision"] if old else 0
        changed = old and (store_code, visit_at) != (old["store_code"], old["visit_at"])
        if (expected_revision is not None and expected_revision != revision) or (changed and expected_revision is None):
            raise vouchers.Conflict("券分支版本已变化，请先读取当前状态")
        voucher, case, version = vouchers.workflow_source(conn, user, voucher_id)
        if old and not changed and version == old["upstream"]:
            if old["decision"] is None and route(old["facts"]) == "explain":
                _save_decision(conn, scope, old, "explain", request_id)
            return _view(old)
        if case:
            facts = {"reason_code": case["assessment"]["reason_code"], "case_status": case["status"],
                     "case_id": case["id"], "case_outcome": case["outcome"]}
        else:
            facts = vouchers.observe_for_plan(conn, user, voucher, store_code,
                                             parse_aware_datetime(visit_at) if visit_at else None)
            facts["missing"] = [field for field, value in (("store_code", store_code), ("visit_at", visit_at)) if not value]
        # A known explanation needs no extra model mutation. All other steps
        # retain the normal proposal/validation flow.
        decision = "explain" if route(facts) == "explain" else None
        row = dict(voucher_id=voucher_id, store_code=store_code, visit_at=visit_at, revision=revision + 1,
                   upstream=version, facts=facts, decision=decision)
        conn.execute("""INSERT INTO mixed_voucher_branches
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, owner_id, conversation_id) DO UPDATE SET
            store_code=excluded.store_code, visit_at=excluded.visit_at, revision=excluded.revision,
            upstream=excluded.upstream, facts=excluded.facts, decision=excluded.decision""",
            (*scope, voucher_id, store_code, visit_at, row["revision"], json.dumps(version), json.dumps(facts), decision))
        _event(conn, scope, row, request_id, "observe")
        if decision:
            _event(conn, scope, row, request_id, decision)
        return _view(row)


def route(facts):
    """Allow a step only when the current voucher facts satisfy its prerequisites."""
    if facts.get("case_status"):
        return "report_review" if facts["case_status"] == "resolved" else "wait_review"
    if facts["reason_code"] in {"matching_redemption", "expired_voucher"}:
        return "explain"
    if facts["missing"]:
        return "ask_visit"
    return "guide_dispute" if facts["visit_step"] == "submit_dispute" else "explain"


def agent_view(state):
    """Expose business facts and required inputs without internal storage counters."""
    view = {key: state[key] for key in ("voucher_id", "revision", "decision", "needs_recheck", "help_url",
                                       "store_code", "visit_at")}
    if state["needs_recheck"]:
        return dict(view, next_step="observe", required_fields=[])
    facts = state["facts"]
    step = route(facts)
    reason = facts["reason_code"]
    view.update(next_step=step, required_fields=facts["missing"] if step == "ask_visit" else [],
                reason_code=reason)
    if facts.get("visit_step"):
        view["visit_step"] = facts["visit_step"]
    if reason == "matching_redemption":
        view["redemption_finding"] = "平台与商家的核销记录吻合，当前显示已核销，无需补充到店资料。"
    elif reason == "expired_voucher":
        view["redemption_finding"] = "券已过期。"
    if facts.get("case_status"):
        view.update(case_status=facts["case_status"], case_id=facts["case_id"], case_outcome=facts["case_outcome"])
        if facts["case_outcome"] == "eligible_to_apply":
            view["customer_next_action"] = "请联系售后申请退款。链接用于查看复核结论；目前尚未退款。"
    return view


def choose(user, sid, expected_revision, action, *, request_id=""):
    scope = _scope(user, sid)
    with db.session(tickets.DB_PATH) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _read(conn, scope)
        if tickets.conversation_closed(conn, *scope) or not row or row["revision"] != expected_revision:
            raise vouchers.Conflict("券分支版本已变化，请先读取当前状态")
        _, _, version = vouchers.workflow_source(conn, user, row["voucher_id"])
        if version != row["upstream"]:
            raise vouchers.Conflict("证据或争议状态已更新，请重新观察券分支")
        if action != route(row["facts"]):
            raise StepConflict(agent_view(_view(row)))
        if row["decision"] is None:
            _save_decision(conn, scope, row, action, request_id)
        return _view(row)


def _save_decision(conn, scope, row, action, request_id):
    row.update(decision=action, revision=row["revision"] + 1)
    conn.execute("""UPDATE mixed_voucher_branches SET decision=?, revision=?
        WHERE tenant_id=? AND owner_id=? AND conversation_id=?""", (action, row["revision"], *scope))
    _event(conn, scope, row, request_id, action)


def _event(conn, scope, row, request_id, action):
    conn.execute("""INSERT INTO mixed_voucher_events
        (tenant_id, owner_id, conversation_id, revision, request_id, action, upstream)
        VALUES (?, ?, ?, ?, ?, ?, ?)""", (*scope, row["revision"], request_id, action, json.dumps(row["upstream"])))
