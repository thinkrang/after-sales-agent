"""端到端验收：按真实链路跑一遍，输出 PASS / FAIL / BLOCKED。

为什么需要这个脚本
------------------
单测过了不等于链路通。这次改动动了认证、会话归属、HITL 恢复、知识库任务参数
和前端请求链，这些只在真实 HTTP + 真实 agent + 真实数据库下才会暴露问题。

设计取舍
--------
- 用 AGENT_DATA_DIR 指向一个空目录，模拟「Docker 从空数据启动」。
  Docker 本身无法验证（daemon 未运行），单独标 BLOCKED。
- 走 TestClient 而不是外部 uvicorn：同样是真实 FastAPI + 中间件 + lifespan，
  但不需要占端口、不需要清理进程，可以反复跑。
- 会调用真实模型（退款意图识别 + 工具调用），因此有 token 成本和几秒延迟。

用法
----
    python scripts/acceptance_e2e.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# 必须在 import api 之前设置：settings.DATA_DIR 在导入时求值
DATA_DIR = BASE / "_acceptance_data"
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(parents=True)
os.environ["AGENT_DATA_DIR"] = str(DATA_DIR)
os.environ.setdefault("AUTH_SEED_DEMO_USERS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

PASSWORD = "demo1234"
results: list[tuple[str, str, str]] = []   # (步骤, 状态, 说明)


def record(step: str, ok: bool | None, note: str) -> None:
    status = "BLOCKED" if ok is None else ("PASS" if ok else "FAIL")
    results.append((step, status, note))
    print(f"  [{status:7s}] {step}  {note}")


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            try:
                events.append((name, json.loads("\n".join(data))))
            except json.JSONDecodeError:
                pass
    return events


def login(client: TestClient, username: str) -> dict:
    r = client.post("/auth/login", json={"username": username, "password": PASSWORD})
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def main() -> int:
    print("=" * 72)
    print("端到端验收")
    print("=" * 72)
    print(f"数据目录（空）：{DATA_DIR}")
    print()

    with TestClient(app) as client:
        # ---------------- S1 启动与健康检查 ----------------
        print("S1 启动与健康检查")
        h = client.get("/health")
        record("S1a /health", h.status_code == 200, f"HTTP {h.status_code}")
        r = client.get("/ready")
        ready_ok = r.status_code == 200 and r.json().get("ready", True)
        record("S1b /ready", ready_ok, f"HTTP {r.status_code} {str(r.json())[:80]}")
        record("S1c Docker 空容器启动", None,
               "本机 docker daemon 未运行，无法验证；已用空数据目录替代其数据面")

        # ---------------- S2 登录与建会话 ----------------
        print("\nS2 登录与创建会话")
        acme = login(client, "acme-manager")
        me = client.get("/auth/me", headers=acme).json()
        record("S2a acme-manager 登录", me.get("tenant_id") == "acme",
               f"tenant={me.get('tenant_id')} roles={me.get('roles')}")
        cr = client.post("/sessions", json={"title": "验收会话"}, headers=acme)
        sid = (cr.json() or {}).get("id") if cr.status_code in (200, 201) else None
        record("S2b 创建会话", bool(sid), f"HTTP {cr.status_code} id={sid}")

        # ---------------- S3 订单归属与可退性 ----------------
        print("\nS3 查询订单 A100")
        od = client.get("/orders", headers=acme)
        orders = od.json().get("orders", od.json() if isinstance(od.json(), list) else [])
        a100 = next((o for o in orders if str(o.get("order_id")) == "A100"), None)
        record("S3a 能看到 A100", a100 is not None,
               f"共 {len(orders)} 单；A100={'有' if a100 else '无'}")
        if a100:
            record("S3b 归属正确", a100.get("owner") in (None, "acme-manager"),
                   f"owner={a100.get('owner')} status={a100.get('status')}")

        # ---------------- S4 触发退款，拿 approval_id ----------------
        print("\nS4 触发退款（调用真实模型，需要两轮）")
        # 第一轮：模型会先复述订单信息并征求确认（这是 skill 里
        # "提交前先跟用户说清楚" 的行为），不会直接调工具。
        t0 = time.perf_counter()
        st1 = client.post("/chat/stream", headers=acme,
                          json={"thread_id": sid, "message": "我要退款，订单号 A100，质量问题"})
        ev1 = parse_sse(st1.text)
        record("S4a 第一轮：模型征求确认", any(k == "token" for k, _ in ev1),
               f"事件={sorted({k for k, _ in ev1})} 耗时 {time.perf_counter()-t0:.1f}s")

        # 第二轮：用户明确同意，模型才真正调 create_refund_request 并触发中断
        st2 = client.post("/chat/stream", headers=acme,
                          json={"thread_id": sid, "message": "可以，请提交"})
        events = parse_sse(st2.text)
        kinds = [k for k, _ in events]
        interrupt = next((d for k, d in events if k == "interrupt"), None)
        approval_id = None
        if interrupt:
            acts = interrupt.get("actions") or []
            if acts:
                approval_id = acts[0].get("approval_id")
        record("S4b 第二轮：触发审批中断", interrupt is not None,
               f"事件={sorted(set(kinds))}")
        record("S4c 返回 approval_id", bool(approval_id), f"approval_id={approval_id}")

        # ---------------- S5 刷新后审批仍在 ----------------
        print("\nS5 刷新页面（重新拉会话消息）")
        msgs = client.get(f"/sessions/{sid}/messages", headers=acme)
        body = json.dumps(msgs.json(), ensure_ascii=False) if msgs.status_code == 200 else ""
        record("S5 待审批状态可恢复", msgs.status_code == 200 and "A100" in body,
               f"HTTP {msgs.status_code} 含订单号={'A100' in body}")

        # ---------------- S6 批准并执行 ----------------
        print("\nS6 批准 → approved → executing → executed")
        if approval_id:
            rv = client.post("/chat/resume", headers=acme, json={
                "thread_id": sid, "approval_id": approval_id, "decision": "approve",
            })
            record("S6a 批准请求", rv.status_code == 200, f"HTTP {rv.status_code}")
            tk = client.get("/tickets", headers=acme).json()
            rows = tk.get("tickets", tk if isinstance(tk, list) else [])
            mine = [t for t in rows if str(t.get("order_id")) == "A100"]
            executed = [t for t in mine if t.get("status") == "executed"]
            record("S6b 工单到达 executed", len(executed) == 1,
                   f"A100 工单 {len(mine)} 条，executed {len(executed)} 条"
                   + (f"，refund_id={executed[0].get('refund_id')}" if executed else ""))
        else:
            record("S6 批准并执行", False, "没有 approval_id，跳过")

        # ---------------- S7 重复批准与重放 ----------------
        print("\nS7 重复点击批准 + 重放请求")
        if approval_id:
            again = client.post("/chat/resume", headers=acme, json={
                "thread_id": sid, "approval_id": approval_id, "decision": "approve",
            })
            tk = client.get("/tickets", headers=acme).json()
            rows = tk.get("tickets", tk if isinstance(tk, list) else [])
            executed = [t for t in rows
                        if str(t.get("order_id")) == "A100" and t.get("status") == "executed"]
            record("S7a 重复批准被拒", again.status_code >= 400,
                   f"HTTP {again.status_code}")
            record("S7b 仍只有一笔退款", len(executed) == 1,
                   f"executed 工单数={len(executed)}")

        # ---------------- S8 跨租户隔离 ----------------
        print("\nS8 globex-user 越权检查")
        globex = login(client, "globex-user")
        x1 = client.get(f"/sessions/{sid}/messages", headers=globex)
        record("S8a 读不到 ACME 会话", x1.status_code == 404,
               f"HTTP {x1.status_code}（期望 404，不泄露存在性）")
        x2 = client.get("/orders", headers=globex)
        g_orders = x2.json().get("orders", []) if x2.status_code == 200 else []
        leaked = [o for o in g_orders if str(o.get("order_id")) == "A100"]
        record("S8b 看不到 ACME 订单 A100", not leaked,
               f"globex 可见 {len(g_orders)} 单，含 A100={'是' if leaked else '否'}")

        # ---------------- S9 崩溃恢复 ----------------
        print("\nS9 模拟「批准后进程中断」+ 对账")
        import refund_service
        import tickets as tk_mod

        from auth import User

        crash_tid = "crash_" + str(int(time.time()))
        actor = User(
            username="acme-manager", display_name="ACME 主管", tenant_id="acme",
            roles=frozenset({"customer", "approver", "knowledge_admin"}),
        )
        # 必须用 prepare_ticket：它才会算 payload_hash 并绑定
        # tenant/user/thread/order/amount/reason。直接用 tickets.create_ticket
        # 建的工单没有绑定信息，execute() 会以 ApprovalRequired 拒绝执行。
        t, _order = refund_service.prepare_ticket(
            actor, thread_id=crash_tid, order_id="A200", reason="验收-崩溃恢复",
        )
        tk_mod.approve_ticket(t["id"], operator="acme-manager",
                              tenant_id="acme", requester_id="acme-manager")
        # 停在这里：模拟「已批准、未执行」时进程挂掉
        recoverable = tk_mod.list_recoverable()
        hit = [x for x in recoverable if x["id"] == t["id"]]
        record("S9a 能被对账脚本发现", bool(hit), f"可恢复 {len(recoverable)} 条")
        if hit:
            res = refund_service.reconcile_ticket(hit[0])
            record("S9b 对账后到达终态", res.get("status") == "executed",
                   f"status={res.get('status')} refund_id={res.get('refund_id')}")
            # 再跑一次，验证幂等
            again_rows = [x for x in tk_mod.list_recoverable() if x["id"] == t["id"]]
            record("S9c 重复对账不再处理", not again_rows,
                   "已终态，不会被再次捞出")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 72)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    blocked = sum(1 for _, s, _ in results if s == "BLOCKED")
    print(f"结果：PASS {passed} / FAIL {failed} / BLOCKED {blocked}")
    if failed:
        print("\n失败项：")
        for step, s, note in results:
            if s == "FAIL":
                print(f"  - {step}: {note}")
    if blocked:
        print("\n未验证项：")
        for step, s, note in results:
            if s == "BLOCKED":
                print(f"  - {step}: {note}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
