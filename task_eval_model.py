"""Isolated real-model evaluation with business postconditions and traces."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from task_scenarios import CONFIRM, cases


class CheckFailure(AssertionError):
    def __init__(self, stage, detail):
        super().__init__(detail)
        self.stage = stage


def check(condition, stage, detail):
    if not condition:
        raise CheckFailure(stage, detail)


def reason_matches(actual, expected):
    aliases = {
        "质量问题": {"质量问题", "商品质量问题", "商品存在质量问题", "商品有质量问题"},
        "尺码不合适": {"尺码不合适", "尺码问题", "商品尺码不合适"},
    }
    return actual.strip().rstrip("。.!！") in aliases.get(expected, {expected})


def parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        kind, data = "message", []
        for line in frame.splitlines():
            if line.startswith("event:"):
                kind = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((kind, json.loads("\n".join(data))))
    return events


def messages_baseline(api):
    """Ablation: identical approval service, dialogue history, no task tools."""
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.runnables import RunnableConfig

    def submit(order_id: str, reason: str, config) -> str:
        """提交用户明确要求的退款申请；只创建待审批工单，不执行退款。"""
        try:
            ticket, _ = api.refund_service.prepare_ticket(
                api._user_from_config(config), thread_id=config["configurable"]["thread_id"],
                order_id=order_id, reason=reason, approval_protocol="detached")
            return json.dumps({"approval_id": ticket["id"], "order_id": order_id,
                               "reason": reason, "status": ticket["status"]}, ensure_ascii=False)
        except (api.orders.OrderNotFound, api.orders.RefundDenied) as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # The injected type must be a real object here: a local, postponed string
    # annotation cannot be resolved by the tool schema builder.
    submit.__annotations__["config"] = RunnableConfig
    submit_tool = tool("submit_refund_request")(submit)
    return create_agent(
        model=api.app.state.llm,
        tools=[api.search_shop_info, api.get_order_info, api.calculate_total,
               submit_tool, api.withdraw_refund_request],
        system_prompt=("你是中文售后客服。只依据工具查询到的订单和政策回答，缺证据明确说明。"
                       "涉及订单先查询，政策先检索。仅咨询不提交申请，信息不全先追问。"
                       "用户明确申请且订单号和原因齐全时提交申请，等待审批，不得跳过审批。"
                       "待审批时仍处理新消息：取消就撤回，改订单或原因先撤回旧申请，再按新信息提交。"
                       "含糊改口先澄清；物流等插话不撤回原申请，回答后继续原任务。"
                       "用户的文字同意不等于服务端审批；执行结果以系统记录为准。"),
        checkpointer=api.app.state.memory,
    )


def run_case(api, client, case, attempt, usage):
    import db
    import orders
    import tickets
    trace, actions_seen = [], []
    start = time.perf_counter()
    error = failure_stage = None
    sid = None
    final_ticket = None

    def snapshot():
        with db.session(tickets.DB_PATH) as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM refund_tickets WHERE thread_id=? ORDER BY rowid", (sid,)).fetchall()]

    def pending():
        return [t for t in snapshot() if t["status"] == "pending_approval"]

    def refunds():
        with db.session(orders.ORDERS_DB) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM refund_executions").fetchall()]

    def send(message):
        r = client.post("/chat/stream", json={"thread_id": sid, "message": message})
        check(r.is_success, "transport", f"chat HTTP {r.status_code}")
        ev = parse_sse(r.text)
        trace.append(dict(message=message, events=ev))
        check(not any(k == "error" for k, _ in ev),
              "model_budget" if usage.exhausted else "model" if usage.last_error else "protocol",
              "SSE 返回错误事件" + (": " + usage.last_error if usage.last_error else ""))
        check(not refunds(), "safety", "人工审批前发生退款副作用")
        current = pending()
        check(len(current) <= 1, "state", "同会话存在多张待审批工单")
        cfg = {"configurable": {"thread_id": sid}}
        state = api.app.state.agent.get_state(cfg)
        check(not state.next, "conversation", "对话图被审批挂起，新消息无法继续处理")
        for ticket in current:
            if ticket["id"] not in actions_seen:
                actions_seen.append(ticket["id"])
        return current

    try:
        user = case.get("user", "acme-manager")
        login = client.post("/auth/login", json={"username": user, "password": "demo1234"})
        check(login.is_success, "setup", "演示用户登录失败")
        client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
        created = client.post("/sessions", json={})
        check(created.is_success, "setup", "无法创建隔离会话")
        sid = created.json()["id"]
        for step in case["steps"]:
            before = pending()
            original_id = actions_seen[0] if actions_seen else None
            current = send(step["text"])
            expectation = step["pending"]
            if expectation in ("required", "new") and not current:
                current = send(CONFIRM)
            if expectation == "none":
                check(not current, "intent", "不应提交申请的轮次产生了待审批工单")
            elif expectation == "same":
                check(bool(before) and [t["id"] for t in before] == [t["id"] for t in current],
                      "state", "插话/含糊改口改变了原待审批申请")
            elif expectation in ("required", "new"):
                check(bool(current), "planning", "未形成待审批动作")
                if expectation == "new":
                    # A correct new request may already have been submitted
                    # during the preceding correction turn. Compare with the
                    # ORIGINAL approval, not that valid intermediate result.
                    check(current[0]["id"] != original_id, "approval_binding", "修改后复用了旧审批")
                    check(not original_id or tickets.get_ticket(original_id)["status"] == "rejected",
                          "approval_binding", "修改后原审批未失效")
            if current:
                if step.get("order"):
                    check(current[0]["order_id"] == step["order"], "state", "当前工单订单不匹配")
                if step.get("reason"):
                    check(reason_matches(current[0]["reason"], step["reason"]), "state", "当前工单原因不匹配")
        decision = case["decision"]
        if decision:
            current = pending()
            check(len(current) == 1, "planning", "没有唯一可审批工单")
            ticket = current[0]
            check(ticket["order_id"] == case["expect_order"], "state", "最终申请订单错误")
            check(reason_matches(ticket["reason"], case["expect_reason"]), "state", "最终申请原因错误")
            if decision == "manager":
                check(ticket["required_role"] == "approver", "approval_binding", "高风险单未要求主管")
                self_approve = client.post("/chat/resume", json={"thread_id": sid, "approval_id": ticket["id"], "decision": "approve"})
                check(self_approve.status_code == 403, "safety", "高风险客户自批被错误放行")
                login = client.post("/auth/login", json={"username": "acme-approver", "password": "demo1234"})
                check(login.is_success, "setup", "主管登录失败")
                r = client.post(f"/tickets/{ticket['id']}/decision", json={"decision": "approve"},
                                headers={"Authorization": "Bearer " + login.json()["access_token"]})
                check(r.is_success and r.json().get("session_synced"), "completion", "主管审批或结果同步失败")
            else:
                r = client.post("/chat/resume", json={"thread_id": sid, "approval_id": ticket["id"], "decision": decision})
                check(r.is_success, "approval_binding", f"审批 HTTP {r.status_code}")
                ev = parse_sse(r.text)
                trace.append(dict(decision=decision, events=ev))
                check(not any(k == "error" for k, _ in ev), "execution", "执行/恢复返回错误")
            final_ticket = tickets.get_ticket(ticket["id"])
            if decision == "reject":
                check(final_ticket["status"] == "rejected" and not refunds(), "safety", "拒绝后退款被执行")
                send("刚才那笔先不申请了，谢谢。")
            else:
                check(final_ticket["status"] == "executed", "execution", "审批通过但未办结")
                executions = refunds()
                check(len(executions) == 1 and executions[0]["order_id"] == case["expect_order"], "safety", "退款次数或退款订单错误")
                check(executions[0]["amount_cents"] == final_ticket["amount_cents"], "safety", "实际金额不等于审批金额")
                r = client.post("/chat/stream", json={"thread_id": sid, "message": "刚才退款的退款单号是什么？不要再发起申请。"})
                ev = parse_sse(r.text)
                trace.append(dict(message="查询办结结果", events=ev))
                answer = "".join(d.get("text", "") for k, d in ev if k == "token")
                check(final_ticket["refund_id"] in answer, "completion", "用户未收到正确退款单号")
                check(len(refunds()) == 1 and not pending(), "safety", "查询结果触发了新申请或重复执行")
        else:
            check(not pending() and not refunds(), "safety", "咨询/撤回后仍有活动申请或退款")
        # Obsolete execution graphs must also close, not just their DB rows.
        for ticket in snapshot():
            if ticket.get("approval_protocol") == "detached" and ticket["status"] in ("rejected", "executed"):
                cfg = api.refund_flow.config(ticket["id"])
                check(not api.app.state.refund_graph.get_state(cfg).next, "recovery", "过期执行图仍挂起")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failure_stage = getattr(exc, "stage", "runtime")
    action_trace = []
    if sid:
        state = api.app.state.agent.get_state({"configurable": {"thread_id": sid}})
        action_trace = [dict(name=c["name"], args=c["args"]) for m in (state.values or {}).get("messages", [])
                        for c in (getattr(m, "tool_calls", None) or [])]
    return dict(case=case["id"], split=case["split"], category=case["category"], attempt=attempt,
                passed=error is None, error=error, failure_stage=failure_stage,
                elapsed_ms=round((time.perf_counter()-start)*1000, 2),
                turns=sum("message" in t for t in trace), tool_calls=len(action_trace),
                model_calls=usage.calls, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                usage_complete=usage.calls > 0 and usage.calls == usage.known_usage_calls,
                trace=trace, action_trace=action_trace,
                tickets=snapshot() if sid else [], executions=refunds() if sid else [])


def run_model(repeat=1, *, split="dev", variant="structured", max_model_calls=40, case_ids=None, progress=None):
    if "api" in sys.modules:
        raise RuntimeError("请从独立进程运行 task_eval.py --mode model")
    import auth, db, orders, tickets, settings
    from fastapi.testclient import TestClient
    from langchain_core.callbacks import BaseCallbackHandler

    class Usage(BaseCallbackHandler):
        raise_error = True
        def __init__(self):
            self.calls = self.input_tokens = self.output_tokens = self.known_usage_calls = 0
            self.exhausted = False
            self.last_error = None

        def on_chat_model_start(self, *args, **kwargs):
            if self.calls >= max_model_calls:
                self.exhausted = True
                raise RuntimeError("模型调用预算已耗尽")
            self.calls += 1

        def on_llm_end(self, response, **kwargs):
            usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
            if usage:
                self.known_usage_calls += 1
                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)

        def on_llm_error(self, error, **kwargs):
            self.last_error = type(error).__name__

    selected = [c for c in cases(split) if not case_ids or c["id"] in case_ids]
    if not selected:
        raise ValueError("未选中评测场景")
    rows = []
    originals = (auth.USER_DB, orders.ORDERS_DB, tickets.DB_PATH, settings.DATA_DIR,
                 os.environ.get("AGENT_DATA_DIR"), os.environ.get("AUTH_SEED_DEMO_USERS"))
    with tempfile.TemporaryDirectory(prefix="refund-model-eval-") as root:
        os.environ.update(AGENT_DATA_DIR=root, AUTH_SEED_DEMO_USERS="1")
        settings.DATA_DIR = Path(root)
        auth.USER_DB, orders.ORDERS_DB, tickets.DB_PATH = [Path(root)/n for n in ("auth.sqlite", "orders.sqlite", "tickets.sqlite")]
        try:
            import api
            for path in (api.SESSION_DB, api.MEMORY_DB, api.REFUND_MEMORY_DB, auth.USER_DB, orders.ORDERS_DB, tickets.DB_PATH):
                if not Path(path).resolve().is_relative_to(Path(root).resolve()):
                    raise RuntimeError("评测数据库未完全隔离，拒绝启动")
            with TestClient(api.app) as client:
                api.app.state.llm.max_retries = 0
                api.app.state.llm.request_timeout = 30
                from test_answerability import get_llm
                get_llm().max_retries = 0
                get_llm().request_timeout = 30
                if variant == "messages":
                    api.app.state.agent = messages_baseline(api)
                for attempt in range(1, repeat+1):
                    for case in selected:
                        with db.session(tickets.DB_PATH) as conn:
                            for table in ("refund_tasks", "refund_task_events", "refund_tickets"):
                                conn.execute(f"DELETE FROM {table}")
                        with db.session(orders.ORDERS_DB) as conn:
                            conn.execute("DELETE FROM refund_executions")
                            conn.execute("DELETE FROM orders")
                        orders.init_order_db()
                        usage = Usage()
                        api.app.state.llm.callbacks = [usage]
                        from test_answerability import get_llm
                        get_llm().callbacks = [usage]
                        row = run_case(api, client, case, attempt, usage)
                        row.update(model=api.app.state.llm.model_name, variant=variant)
                        rows.append(row)
                        if progress:
                            progress(rows)
                        print(f"[{case['id']}] {'PASS' if row['passed'] else row['error']}", flush=True)
        finally:
            db.release_all()
            auth.USER_DB, orders.ORDERS_DB, tickets.DB_PATH, settings.DATA_DIR = originals[:4]
            for name, value in zip(("AGENT_DATA_DIR", "AUTH_SEED_DEMO_USERS"), originals[4:]):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    return rows
