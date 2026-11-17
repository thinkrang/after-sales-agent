"""Browser replay of a mixed request using a scripted LangGraph model."""
from datetime import datetime, timezone
import json
from threading import Lock
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agent_context import ContextBoundary, result_code
from auth import User
import mixed_tools as tools
import mixed_voucher
import refund_tasks
import voucher_disputes


STEPS = ("start", "arrival", "submit", "refresh")
TOOLS = [tools.get_voucher_branch, tools.observe_voucher_branch, tools.choose_voucher_step,
         tools.get_order_info, tools.update_refund_task, tools.submit_refund_request]


class ReplayModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class Action(BaseModel):
    action: Literal["start", "arrival", "submit", "refresh"]


class MixedReplay:
    def __init__(self):
        self.customer = User("acme-user", "演示消费者", "acme", frozenset({"customer"}))
        self.sid = "offline-" + uuid.uuid4().hex
        self.memory = InMemorySaver()
        self.lock = Lock()
        self.finished = 0
        self.trace = []
        self.arrival = datetime.now(timezone.utc).replace(microsecond=0)

    def state(self):
        refund = refund_tasks.get(self.customer, self.sid)
        voucher = mixed_voucher.get(self.customer, self.sid)
        return {
            "next": STEPS[self.finished] if self.finished < len(STEPS) else None,
            "voucher": mixed_voucher.agent_view(voucher) if voucher else None,
            "refund": {key: refund[key] for key in ("order_id", "status", "ticket_id")},
            "trace": self.trace,
        }

    def advance(self, action):
        index = STEPS.index(action)
        if index < self.finished:
            return self.state()
        if index != self.finished:
            raise HTTPException(409, "请先完成前一步")
        row = mixed_voucher.get(self.customer, self.sid)
        order = {"order_id": "A300", "reason": "尺码不合适"}
        if action == "start":
            steps = [
                ("observe_voucher_branch", {"voucher_id": "V200", "store_code": "CENTRAL"}),
                ("choose_voucher_step", {"action": "ask_visit", "expected_revision": 1}),
                ("get_order_info", {"order_id": "A300"}),
                ("update_refund_task", dict(order, intent="apply", expected_revision=0)),
                ("submit_refund_request", dict(order, task_revision=1)),
            ]
            message, answer = ("V200 在中心店用不了。A300 尺码不合适，请申请退款。",
                               "请补充 V200 的到店时间。A300 退款申请已提交，等待你本人确认。")
        else:
            revision = row["revision"]
            observed = {"voucher_id": "V200", "expected_revision": revision}
            if action == "arrival":
                observed["visit_at"] = self.arrival.isoformat()
                decision = "guide_dispute"
                message, answer = (f"我的到店时间是 {self.arrival.isoformat()}。",
                                   "已核对到店资料。平台与商家的记录不一致，可以确认提交争议。")
            elif action == "submit":
                voucher_disputes.submit(self.customer, "V200", "到店无法使用",
                                        store_code="CENTRAL", visit_at=self.arrival)
                decision = "wait_review"
                message, answer = "我已确认提交争议。", "V200 争议已提交，等待人工复核。"
            else:
                if not row["needs_recheck"]:
                    raise HTTPException(409, "请先补传商家流水")
                decision = "wait_review"
                message, answer = ("商家已补传，帮我刷新 V200。",
                                   "已按新流水核对 V200，记录吻合，仍待复核。A300 原工单未变。")
            steps = [("get_voucher_branch", {})]
            if action == "refresh":
                steps.append(("choose_voucher_step", {"action": "wait_review", "expected_revision": revision}))
            steps += [("observe_voucher_branch", observed),
                      ("choose_voucher_step", {"action": decision, "expected_revision": revision + 1})]
        self._invoke(action, steps, message, answer)
        self.finished += 1
        return self.state()

    def _invoke(self, action, steps, message, answer):
        calls = [AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"{action}-{i}", "type": "tool_call"},
        ]) for i, (name, args) in enumerate(steps)]
        graph = create_agent(model=ReplayModel(responses=[*calls, AIMessage(content=answer)]), tools=TOOLS,
                             middleware=[ContextBoundary()], checkpointer=self.memory)
        config = {"configurable": {"thread_id": self.sid, "tenant": self.customer.tenant_id,
                  "user_id": self.customer.username, "roles": sorted(self.customer.roles)}}
        turns = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)["messages"]
        ids = {f"{action}-{i}" for i in range(len(steps))}
        outputs = [msg for msg in turns if isinstance(msg, ToolMessage) and msg.tool_call_id in ids]
        codes = [result_code(msg) for msg in outputs]
        expected = [None, "conflict", None, None] if action == "refresh" else [None] * len(steps)
        if codes != expected:
            raise HTTPException(409, "回放未按预期完成，请重启演示后重试")
        self.trace.append({"action": action, "message": message, "answer": turns[-1].text,
                           "tools": [{"name": msg.name, "code": code, "result": json.loads(msg.content)}
                                     for msg, code in zip(outputs, codes)]})


def install(app: FastAPI):
    replay = MixedReplay()

    @app.get("/demo/mixed")
    def inspect():
        with replay.lock:
            return replay.state()

    @app.post("/demo/mixed")
    def advance(body: Action):
        with replay.lock:
            return replay.advance(body.action)
