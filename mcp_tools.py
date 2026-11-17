"""MCP 工具的纯实现：不导入 MCP SDK，便于轻量测试。

鉴权边界
--------
MCP 协议本身不携带用户身份，所以 tenant/user 只能由服务端从 Bearer token 解析，
**绝不接受客户端自报**。否则任何能连上 server 的客户端都能读别的租户的订单。

写操作
------
`prepare_refund` 只创建 `pending_approval` 工单，不执行退款。
审批仍走原有的四眼原则与幂等执行链路——MCP 不是后门。
"""

from __future__ import annotations

import json
import hashlib
import logging
import re

import auth
import long_memory
import orders
import refund_service
import refund_tasks
import tool_result
import db
import tickets

THREAD_PREFIX = "mcp-"


def default_thread(user) -> str:
    """Stable namespace; each order now receives a separate task within it."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", user.username)[:40]
    return f"{THREAD_PREFIX}{slug}"


def user_from_headers(headers) -> auth.User:
    """从 ``Authorization: Bearer <token>`` 解析用户。"""
    token = ""
    for key, value in (headers or {}).items():
        if key.lower() == "authorization":
            token = str(value).strip()
            break
    if not token.lower().startswith("bearer "):
        raise PermissionError("缺少 Bearer token：MCP 客户端必须携带用户身份")
    try:
        return auth.decode_access_token(token[7:].strip())
    except ValueError as exc:
        # 统一成鉴权失败：不向客户端区分"格式错误"和"已过期"。
        raise PermissionError("访问令牌无效或已过期") from exc


def get_order_info(user, order_id: str) -> str:
    try:
        order = orders.get_order(user, order_id)
        return tool_result.order_snapshot(order, orders.refund_policy(order))
    except Exception as exc:  # noqa: BLE001
        return tool_result.order_lookup_failure(exc)


def get_refund_task(user, thread_id: str | None = None) -> str:
    if thread_id is None:
        with db.session(tickets.DB_PATH) as conn:
            rows = conn.execute("""SELECT thread_id FROM refund_tasks
                WHERE tenant_id=? AND requester_id=? AND substr(thread_id,1,4)='mcp-'
                ORDER BY updated_at DESC LIMIT 20""", (user.tenant_id, user.username)).fetchall()
        return json.dumps({"tasks": [refund_tasks.get(user, r[0]) for r in rows]}, ensure_ascii=False)
    return json.dumps(
        refund_tasks.get(user, thread_id or default_thread(user)), ensure_ascii=False
    )


def recall_history(user, limit: int = 3) -> str:
    return json.dumps(long_memory.recent_cases(user, limit), ensure_ascii=False)


def search_shop_info(user, question: str, top_k: int = 3) -> str:
    """检索 + 证据核查。检索栈依赖较重，按需导入。"""
    from evidence import cited_evidence, verify_evidence
    from rag import format_evidence
    import retrieval

    try:
        results = retrieval.retrieve(
            question, top_k=top_k, strategy=retrieval.DEFAULT_STRATEGY,
            tenant=user.tenant_id,
        )
        verdict = verify_evidence(question, results)
    except Exception:  # noqa: BLE001
        logging.warning("MCP 检索失败 question=%s", question, exc_info=True)
        return tool_result.retrieval_failure()

    if verdict.sufficient:
        return format_evidence(cited_evidence(verdict, results))
    return tool_result.no_evidence(verdict.missing or "核查结果无效")


def prepare_refund(user, order_id: str, reason: str, thread_id: str | None = None,
                   expected_revision: int | None = None) -> str:
    """登记任务并创建待审批工单；返回 approval_id，不执行退款。"""
    key = hashlib.sha256(json.dumps([user.tenant_id, user.username, order_id.strip()]).encode()).hexdigest()[:24]
    tid = thread_id or default_thread(user) + "-" + key
    try:
        if not tid.startswith("mcp-") or len(tid) > 120:
            raise refund_tasks.TaskConflict("请使用 MCP 返回的任务标识")
        task = refund_tasks.get(user, tid)
        unchanged = task["order_id"] == order_id.strip() and task["reason"] == " ".join(reason.split())
        if task["revision"] and not unchanged and expected_revision is None:
            raise refund_tasks.TaskConflict("修改已有任务必须提供读取到的 expected_revision")
        task = refund_tasks.update(
            user, tid, expected_revision=task["revision"] if expected_revision is None else expected_revision,
            order_id=order_id, reason=reason, intent="apply", source="mcp",
        )
        ticket, _ = refund_service.prepare_ticket(
            user, thread_id=tid, order_id=order_id, reason=reason,
            task_revision=task["revision"], approval_protocol="detached",
        )
        return tool_result.ok(
            approval_id=ticket["id"], status=ticket["status"],
            order_id=ticket["order_id"],
            amount=f"{ticket['amount_cents'] / 100:.2f}",
            required_role=ticket["required_role"], thread_id=tid,
            task_id=tid, task_revision=task["revision"],
            approval_url=f"/tickets/{ticket['id']}/decision",
        )
    except refund_tasks.TaskConflict as exc:
        return tool_result.fail(
            tool_result.CONFLICT,
            "任务已变化；重新读取任务后再提交，不要猜测版本", detail=str(exc),
        )
    except orders.OrderNotFound:
        return tool_result.fail(
            tool_result.NOT_FOUND,
            "订单不存在或不属于当前账号；向用户确认订单号，不要换号重试",
        )
    except orders.RefundDenied as exc:
        return tool_result.fail(
            tool_result.DENIED,
            "订单不满足退款条件；向用户说明原因，不要重试", detail=str(exc),
        )
