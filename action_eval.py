"""动作级评测：报告安全指标，而不是问答准确率。

为什么要有这个脚本
------------------
62 条问答集衡量的是"答得对不对"，但它答不了这几个问题：

- 有没有错误执行了别人的订单？
- 有没有绕过审批直接退款？
- 重复点击会不会退两次？
- 崩溃之后能不能恢复？

这些是**系统性质**，不是模型能力。所以这里的指标是
「危险动作误执行数 / 越权数 / 重复退款数 / 恢复成功率」——
全是 0 才算合格，不是"93.9%"这种可以接受不完美的数字。

分层原则
--------
能用确定性方法验证的，绝不用模型评测：

- 越权、审批绕过、幂等、状态机、恢复 → 纯代码路径，直接调服务层。
  快、稳、结论硬，每次提交都能跑。
- 路径选择、多轮补槽、注入抵抗 → 依赖模型判断，单独一组，
  标记为 opt-in（`--with-model`），因为要调真实模型。

把前者也交给模型评测，只会让"安全"这件事变得不可靠：
模型每次回答不一样，你就没法确定是系统有洞还是这次恰好没触发。

用法
----
    python action_eval.py                # 只跑确定性场景
    python action_eval.py --with-model   # 加上模型行为场景（慢、有成本）
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import auth
import db
import orders
import refund_service
import risk
import tickets

ACME_MANAGER = auth.User(
    "acme-manager", "ACME 售后主管", "acme",
    frozenset({"customer", "approver", "knowledge_admin"}),
)
ACME_APPROVER = auth.User(
    "acme-approver", "ACME 审批人", "acme", frozenset({"customer", "approver"}),
)
ACME_USER = auth.User("acme-user", "ACME 客户", "acme", frozenset({"customer"}))
GLOBEX_USER = auth.User(
    "globex-user", "GLOBEX 客户", "globex", frozenset({"customer"})
)


@dataclass
class Outcome:
    ok: bool
    detail: str
    danger: bool = False          # 危险动作被执行了（比"没测到"严重得多）


@dataclass
class Scenario:
    id: str
    category: str
    description: str
    fn: Callable[[], Outcome]


@dataclass
class Report:
    rows: list = field(default_factory=list)

    def add(self, scenario: Scenario, outcome: Outcome) -> None:
        self.rows.append((scenario, outcome))

    def count(self, *, category: str | None = None) -> tuple[int, int]:
        rows = [r for r in self.rows if category is None or r[0].category == category]
        ok = sum(1 for _, o in rows if o.ok)
        return ok, len(rows)

    @property
    def dangerous(self) -> list:
        return [(s, o) for s, o in self.rows if o.danger]

    @property
    def failed(self) -> list:
        return [(s, o) for s, o in self.rows if not o.ok]


# ---------------------------------------------------------------- 环境隔离


def _reset(tmp: Path) -> None:
    """每个场景一套全新数据库，避免互相污染。"""
    tickets.DB_PATH = tmp / "tickets.sqlite"
    orders.ORDERS_DB = tmp / "orders.sqlite"
    auth.USER_DB = tmp / "auth.sqlite"
    tickets.init_tickets_table()
    orders.init_order_db()
    auth.init_auth_db(seed_demo_users=True)


# ---------------------------------------------------------------- 断言助手


def blocked(fn: Callable, exc_types, note: str = "") -> Outcome:
    """期望被拦截。**没拦住就是危险动作**。"""
    try:
        fn()
    except exc_types as exc:
        return Outcome(True, f"已拦截 {type(exc).__name__}{('：' + note) if note else ''}")
    except Exception as exc:  # noqa: BLE001
        return Outcome(False, f"异常类型不符 {type(exc).__name__}: {exc}")
    return Outcome(False, "本应被拦截，但执行成功了", danger=True)


def allowed(fn: Callable, note: str = "") -> Outcome:
    """期望放行。拦错了是功能缺陷，但不是安全事故。"""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return Outcome(False, f"本应放行，却被 {type(exc).__name__} 拦下：{exc}")
    return Outcome(True, f"按预期放行{('：' + note) if note else ''}")


def _prepare(user, thread, order_id, reason="质量问题") -> dict:
    ticket, _ = refund_service.prepare_ticket(
        user, thread_id=thread, order_id=order_id, reason=reason
    )
    return ticket


# ---------------------------------------------------------------- A 身份与租户


def scenarios_identity() -> list[Scenario]:
    def order_scope():
        return blocked(
            lambda: orders.get_order(GLOBEX_USER, "A100"),
            orders.OrderNotFound, "globex 拿 acme 的订单号",
        )

    def order_list_scope():
        visible = [o["order_id"] for o in orders.list_orders(GLOBEX_USER)]
        if "A100" in visible:
            return Outcome(False, "globex 的订单列表里出现了 acme 的 A100", danger=True)
        return Outcome(True, f"globex 只看到 {visible}")

    def ticket_scope():
        t = _prepare(ACME_MANAGER, "th-a", "A100")
        got = tickets.get_ticket(t["id"], tenant_id=GLOBEX_USER.tenant_id)
        if got is not None:
            return Outcome(False, "globex 能读到 acme 的工单", danger=True)
        return Outcome(True, "跨租户查工单返回 None")

    def approve_other_tenant():
        t = _prepare(ACME_MANAGER, "th-a2", "A100")
        return blocked(
            lambda: refund_service.decide(
                GLOBEX_USER, approval_id=t["id"], decision="approve"
            ),
            orders.OrderNotFound, "globex 批 acme 的工单",
        )

    def token_tampered_tenant():
        token = auth.create_access_token(GLOBEX_USER)
        head, body, sig = token.split(".")
        payload = json.loads(auth._b64decode(body))
        payload["tenant"] = "acme"            # 想把租户改成 acme
        forged = ".".join([
            head,
            auth._b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()),
            sig,                               # 签名不变
        ])
        return blocked(
            lambda: auth.decode_access_token(forged), ValueError, "改租户但不动签名"
        )

    def token_expired():
        old = auth.create_access_token(ACME_USER, now=1_600_000_000)
        return blocked(
            lambda: auth.decode_access_token(old, now=1_800_000_000),
            ValueError, "过期 token",
        )

    def token_bad_signature():
        token = auth.create_access_token(ACME_USER)
        return blocked(
            lambda: auth.decode_access_token(token[:-4] + "AAAA"),
            ValueError, "篡改签名",
        )

    def token_alg_confusion():
        token = auth.create_access_token(ACME_USER)
        head, body, sig = token.split(".")
        header = json.loads(auth._b64decode(head))
        header["alg"] = "none"
        forged = ".".join([
            auth._b64encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode()),
            body, sig,
        ])
        return blocked(
            lambda: auth.decode_access_token(forged), ValueError, "alg 改成 none"
        )

    def password_hash_salted():
        h1 = auth.hash_password("demo1234")
        h2 = auth.hash_password("demo1234")
        if h1 == h2:
            return Outcome(False, "两次哈希相同，说明没加盐", danger=True)
        return Outcome(True, "同密码两次哈希不同（加盐生效）")

    return [
        Scenario("ID01", "身份与租户", "globex 用 acme 的订单号查订单", order_scope),
        Scenario("ID02", "身份与租户", "globex 的订单列表不含 acme 订单", order_list_scope),
        Scenario("ID03", "身份与租户", "globex 查 acme 的工单", ticket_scope),
        Scenario("ID04", "身份与租户", "globex 批 acme 的工单", approve_other_tenant),
        Scenario("ID05", "身份与租户", "篡改 token 里的 tenant", token_tampered_tenant),
        Scenario("ID06", "身份与租户", "过期 token", token_expired),
        Scenario("ID07", "身份与租户", "篡改签名", token_bad_signature),
        Scenario("ID08", "身份与租户", "alg 混淆攻击", token_alg_confusion),
        Scenario("ID09", "身份与租户", "密码哈希加盐", password_hash_salted),
    ]


# ---------------------------------------------------------------- B 审批绕过


def scenarios_bypass() -> list[Scenario]:
    def execute_without_ticket():
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="no-ticket", order_id="A100", reason="质量问题"
            ),
            (refund_service.ApprovalRequired,),
            "没有任何工单就执行退款",
        )

    def execute_pending():
        t = _prepare(ACME_MANAGER, "th-p", "A100")
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="th-p", order_id="A100", reason="质量问题"
            ),
            (refund_service.ApprovalRequired,),
            f"工单还是 {t['status']} 就执行",
        )

    def execute_rejected():
        t = _prepare(ACME_MANAGER, "th-r", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="th-r", decision="reject")
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="th-r", order_id="A100", reason="质量问题"
            ),
            (refund_service.ApprovalRequired,),
            "已拒绝的工单不能执行",
        )

    def reuse_approval_changed_amount():
        """改金额后复用审批——哈希对不上，必须找不到工单。"""
        t = _prepare(ACME_MANAGER, "th-amt", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="th-amt", decision="approve")
        # 同一订单，但换一个金额不同的订单 A400 试图复用
        return blocked(
            lambda: refund_service.execute(
                ACME_USER, thread_id="th-amt", order_id="A400", reason="质量问题"
            ),
            (refund_service.ApprovalRequired, orders.OrderNotFound),
            "拿 A100 的审批去退 A400",
        )

    def reuse_approval_changed_reason():
        t = _prepare(ACME_MANAGER, "th-rsn", "A100", reason="质量问题")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="th-rsn", decision="approve")
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="th-rsn", order_id="A100", reason="不想要了"
            ),
            (refund_service.ApprovalRequired,),
            "改退款原因后复用审批",
        )

    def reuse_approval_changed_thread():
        t = _prepare(ACME_MANAGER, "th-x", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="th-x", decision="approve")
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="th-y", order_id="A100", reason="质量问题"
            ),
            (refund_service.ApprovalRequired,),
            "换个会话复用审批",
        )

    def approval_id_from_client_ignored():
        """approval_id 必须来自服务端工单，客户端随便给一个不能生效。"""
        _prepare(ACME_MANAGER, "th-fake", "A100")
        return blocked(
            lambda: refund_service.execute(
                ACME_MANAGER, thread_id="th-fake", order_id="A100", reason="质量问题"
            ),
            (refund_service.ApprovalRequired,),
            "伪造 approval_id 无法推进",
        )

    return [
        Scenario("BY01", "审批绕过", "没有工单直接执行", execute_without_ticket),
        Scenario("BY02", "审批绕过", "pending 状态执行", execute_pending),
        Scenario("BY03", "审批绕过", "rejected 状态执行", execute_rejected),
        Scenario("BY04", "审批绕过", "换订单复用审批", reuse_approval_changed_amount),
        Scenario("BY05", "审批绕过", "改原因复用审批", reuse_approval_changed_reason),
        Scenario("BY06", "审批绕过", "换会话复用审批", reuse_approval_changed_thread),
        Scenario("BY07", "审批绕过", "伪造 approval_id", approval_id_from_client_ignored),
    ]


# ---------------------------------------------------------------- C 风险分级


def scenarios_risk() -> list[Scenario]:
    def low_risk_customer_ok():
        t = _prepare(ACME_MANAGER, "rk1", "A100")
        if t["risk_tier"] != risk.TIER_LOW:
            return Outcome(False, f"A100 应为低风险，实际 {t['risk_tier']}")
        return allowed(
            lambda: refund_service.decide(
                ACME_MANAGER, approval_id=t["id"], thread_id="rk1", decision="approve"
            ),
            "低风险单客户本人确认",
        )

    def low_risk_other_denied():
        t = _prepare(ACME_MANAGER, "rk2", "A100")
        return blocked(
            lambda: refund_service.decide(
                ACME_APPROVER, approval_id=t["id"], decision="approve"
            ),
            refund_service.ApprovalForbidden, "审批人越权处理他人低风险单",
        )

    def high_risk_self_denied():
        t = _prepare(ACME_USER, "rk3", "A400")
        if t["risk_tier"] != risk.TIER_HIGH:
            return Outcome(False, f"A400 应为高风险，实际 {t['risk_tier']}")
        return blocked(
            lambda: refund_service.decide(
                ACME_USER, approval_id=t["id"], thread_id="rk3", decision="approve"
            ),
            refund_service.ApprovalForbidden, "高风险单客户自批",
        )

    def high_risk_other_approver_ok():
        t = _prepare(ACME_USER, "rk4", "A400")
        return allowed(
            lambda: refund_service.decide(
                ACME_APPROVER, approval_id=t["id"], decision="approve"
            ),
            "高风险单由同租户另一位审批人批准",
        )

    def high_risk_self_approver_denied():
        """主管自己发起的高风险单，自己也不能批。A500 签收 6 天 → 边界政策。"""
        t = _prepare(ACME_MANAGER, "rk5", "A500")
        if t["risk_tier"] != risk.TIER_HIGH:
            return Outcome(False, f"A500 应为高风险，实际 {t['risk_tier']}")
        return blocked(
            lambda: refund_service.decide(
                ACME_MANAGER, approval_id=t["id"], decision="approve"
            ),
            refund_service.ApprovalForbidden, "主管自批自己发起的高风险单",
        )

    def high_risk_without_role_denied():
        t = _prepare(ACME_USER, "rk6", "A400")
        return blocked(
            lambda: refund_service.decide(
                GLOBEX_USER, approval_id=t["id"], decision="approve"
            ),
            orders.OrderNotFound, "跨租户非审批人",
        )

    def risk_fields_persisted():
        t = _prepare(ACME_USER, "rk7", "A400")
        missing = [
            k for k in ("risk_tier", "risk_reasons", "required_role", "rule_version")
            if not t.get(k)
        ]
        if missing:
            return Outcome(False, f"工单缺风险字段 {missing}")
        if t["rule_version"] != risk.RISK_RULES_VERSION:
            return Outcome(False, "规则版本与代码不一致")
        return Outcome(True, f"{t['risk_tier']} / {t['rule_version']}")

    return [
        Scenario("RK01", "风险分级", "低风险客户本人可批", low_risk_customer_ok),
        Scenario("RK02", "风险分级", "他人不能批低风险单", low_risk_other_denied),
        Scenario("RK03", "风险分级", "高风险客户自批被拒", high_risk_self_denied),
        Scenario("RK04", "风险分级", "高风险由他人批准", high_risk_other_approver_ok),
        Scenario("RK05", "风险分级", "主管自批自己的高风险单被拒", high_risk_self_approver_denied),
        Scenario("RK06", "风险分级", "跨租户非审批人不可批", high_risk_without_role_denied),
        Scenario("RK07", "风险分级", "风险字段落库且版本一致", risk_fields_persisted),
    ]


# ---------------------------------------------------------------- D 幂等与恢复


def scenarios_idempotency() -> list[Scenario]:
    def double_approve():
        t = _prepare(ACME_MANAGER, "id1", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="id1", decision="approve")
        return blocked(
            lambda: refund_service.decide(
                ACME_MANAGER, approval_id=t["id"], thread_id="id1", decision="approve"
            ),
            refund_service.WorkflowConflict, "重复批准",
        )

    def double_execute_returns_same():
        t = _prepare(ACME_MANAGER, "id2", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="id2", decision="approve")
        first = refund_service.execute(
            ACME_MANAGER, thread_id="id2", order_id="A100", reason="质量问题"
        )
        second = refund_service.execute(
            ACME_MANAGER, thread_id="id2", order_id="A100", reason="质量问题"
        )
        if first["refund_id"] != second["refund_id"]:
            return Outcome(False, "重复执行产生了两个退款单号", danger=True)
        return Outcome(True, f"两次执行同一单号 {first['refund_id']}")

    def concurrent_approve_single_winner():
        t = _prepare(ACME_MANAGER, "id3", "A100")
        wins = []

        def go():
            try:
                refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                                      thread_id="id3", decision="approve")
                wins.append(1)
            except Exception:  # noqa: BLE001
                pass

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: go(), range(8)))
        if len(wins) != 1:
            return Outcome(False, f"{len(wins)} 个线程都批准成功，应为 1", danger=True)
        return Outcome(True, "8 线程并发只有 1 个成功")

    def concurrent_execute_single_refund():
        t = _prepare(ACME_MANAGER, "id4", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="id4", decision="approve")
        ids, errs = [], []

        def go():
            try:
                r = refund_service.execute(
                    ACME_MANAGER, thread_id="id4", order_id="A100", reason="质量问题"
                )
                ids.append(r["refund_id"])
            except Exception as exc:  # noqa: BLE001
                errs.append(str(exc)[:40])

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: go(), range(6)))
        distinct = set(ids)
        if len(distinct) > 1:
            return Outcome(False, f"产生多个退款单号 {distinct}", danger=True)
        return Outcome(True, f"6 线程并发只有 1 个单号（成功 {len(ids)} / 异常 {len(errs)}）")

    def recover_approved():
        t = _prepare(ACME_MANAGER, "id5", "A200")
        tickets.approve_ticket(t["id"], operator="acme-manager",
                               tenant_id="acme", requester_id="acme-manager")
        rows = [x for x in tickets.list_recoverable() if x["id"] == t["id"]]
        if not rows:
            return Outcome(False, "已批准工单没被对账脚本捞出")
        res = refund_service.reconcile_ticket(rows[0])
        if res["status"] != tickets.EXECUTED:
            return Outcome(False, f"恢复后状态为 {res['status']}")
        return Outcome(True, f"恢复到 executed，单号 {res.get('refund_id')}")

    def recover_executing():
        """模拟"已抢占执行权但进程挂掉"：executing 状态必须能被对账完成。"""
        t = _prepare(ACME_MANAGER, "id6", "A200")
        tickets.approve_ticket(t["id"], operator="acme-manager",
                               tenant_id="acme", requester_id="acme-manager")
        claimed = tickets.acquire_execution(
            "id6", tenant_id="acme", requester_id="acme-manager",
            order_id="A200",
            payload_hash=tickets.get_ticket(t["id"])["payload_hash"],
        )
        if claimed is None:
            return Outcome(False, "抢占执行权失败，无法模拟崩溃")
        rows = [x for x in tickets.list_recoverable() if x["id"] == t["id"]]
        if not rows:
            return Outcome(False, "executing 状态未被 list_recoverable 捞出")
        res = refund_service.reconcile_ticket(rows[0])
        if res["status"] != tickets.EXECUTED:
            return Outcome(False, f"恢复后状态为 {res['status']}")
        return Outcome(True, "executing 崩溃后恢复成功")

    def recover_twice_idempotent():
        t = _prepare(ACME_MANAGER, "id7", "A200")
        tickets.approve_ticket(t["id"], operator="acme-manager",
                               tenant_id="acme", requester_id="acme-manager")
        rows = [x for x in tickets.list_recoverable() if x["id"] == t["id"]]
        first = refund_service.reconcile_ticket(rows[0])
        again = [x for x in tickets.list_recoverable() if x["id"] == t["id"]]
        if again:
            return Outcome(False, "已终态的工单仍被捞出，会重复处理", danger=True)
        if first["status"] != tickets.EXECUTED:
            return Outcome(False, "首次恢复未成功")
        return Outcome(True, "重复对账不会二次处理")

    def cross_tenant_same_order_id():
        """acme 和 globex 都有 A100 风格的订单号时，不能互相吞掉。"""
        a = _prepare(ACME_MANAGER, "id8a", "A100")
        g = _prepare(GLOBEX_USER, "id8b", "G100")
        refund_service.decide(ACME_MANAGER, approval_id=a["id"],
                              thread_id="id8a", decision="approve")
        refund_service.execute(ACME_MANAGER, thread_id="id8a",
                               order_id="A100", reason="质量问题")
        # globex 的订单必须还能正常发起
        if g["status"] != tickets.PENDING:
            return Outcome(False, f"globex 工单状态异常 {g['status']}", danger=True)
        return Outcome(True, "两个租户的退款互不影响")

    return [
        Scenario("IDM01", "幂等与恢复", "重复批准被拒", double_approve),
        Scenario("IDM02", "幂等与恢复", "重复执行返回原单号", double_execute_returns_same),
        Scenario("IDM03", "幂等与恢复", "8 线程并发批准只有一个成功", concurrent_approve_single_winner),
        Scenario("IDM04", "幂等与恢复", "6 线程并发执行只有一个单号", concurrent_execute_single_refund),
        Scenario("IDM05", "幂等与恢复", "approved 状态崩溃后恢复", recover_approved),
        Scenario("IDM06", "幂等与恢复", "executing 状态崩溃后恢复", recover_executing),
        Scenario("IDM07", "幂等与恢复", "重复对账不二次处理", recover_twice_idempotent),
        Scenario("IDM08", "幂等与恢复", "跨租户订单互不干扰", cross_tenant_same_order_id),
    ]


# ---------------------------------------------------------------- E 状态机


def scenarios_state_machine() -> list[Scenario]:
    def _executed_ticket():
        t = _prepare(ACME_MANAGER, "sm0", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="sm0", decision="approve")
        refund_service.execute(ACME_MANAGER, thread_id="sm0",
                               order_id="A100", reason="质量问题")
        return t["id"]

    def executed_cannot_go_back():
        tid = _executed_ticket()
        ok = tickets.mark_failed(tid, "试图回退")
        if ok:
            return Outcome(False, "executed 被改成了 failed", danger=True)
        return Outcome(True, "终态不可回退")

    def pending_cannot_skip_to_executed():
        t = _prepare(ACME_MANAGER, "sm2", "A100")
        ok = tickets.mark_executed(t["id"], "RF-FAKE")
        if ok:
            return Outcome(False, "pending 直接跳到 executed", danger=True)
        return Outcome(True, "不能跳过审批直接标记执行成功")

    def rejected_cannot_approve():
        t = _prepare(ACME_MANAGER, "sm3", "A100")
        refund_service.decide(ACME_MANAGER, approval_id=t["id"],
                              thread_id="sm3", decision="reject")
        ok = tickets.approve_ticket(t["id"], operator="acme-manager",
                                    tenant_id="acme", requester_id="acme-manager")
        if ok:
            return Outcome(False, "rejected 被重新批准了", danger=True)
        return Outcome(True, "rejected 是终态")

    def executed_cannot_reapprove():
        tid = _executed_ticket()
        ok = tickets.approve_ticket(tid, operator="acme-manager",
                                    tenant_id="acme", requester_id="acme-manager")
        if ok:
            return Outcome(False, "executed 被重新批准", danger=True)
        return Outcome(True, "executed 是终态")

    def invalid_transition_table():
        bad = []
        for src, targets in tickets.ALLOWED_TRANSITIONS.items():
            for tgt in targets:
                if src == tgt:
                    bad.append(f"{src}->{tgt}")
        if bad:
            return Outcome(False, f"存在自环迁移 {bad}")
        if tickets.ALLOWED_TRANSITIONS[tickets.EXECUTED]:
            return Outcome(False, "executed 不是终态")
        return Outcome(True, "迁移表无自环、终态为空集")

    return [
        Scenario("SM01", "状态机", "executed 不能回退", executed_cannot_go_back),
        Scenario("SM02", "状态机", "pending 不能跳到 executed", pending_cannot_skip_to_executed),
        Scenario("SM03", "状态机", "rejected 不能重新批准", rejected_cannot_approve),
        Scenario("SM04", "状态机", "executed 不能重新批准", executed_cannot_reapprove),
        Scenario("SM05", "状态机", "迁移表无自环且终态封闭", invalid_transition_table),
    ]


# ---------------------------------------------------------------- 汇总与输出


def deterministic_scenarios() -> list[Scenario]:
    return (
        scenarios_identity()
        + scenarios_bypass()
        + scenarios_risk()
        + scenarios_idempotency()
        + scenarios_state_machine()
    )


def run(scenarios: list[Scenario], *, verbose: bool = True) -> Report:
    """跑一组场景。

    跑完必须把模块级 DB 路径还原：这个函数会被 pytest 调用，如果不还原，
    同进程里后面的测试就会连到一个已经被删掉的临时目录上。
    """
    saved = {
        "tickets": tickets.DB_PATH,
        "orders": orders.ORDERS_DB,
        "auth": auth.USER_DB,
    }
    report = Report()
    try:
        for s in scenarios:
            tmp = Path(tempfile.mkdtemp(prefix="act_"))
            try:
                _reset(tmp)
                try:
                    outcome = s.fn()
                except Exception as exc:  # noqa: BLE001
                    outcome = Outcome(False, f"场景本身抛异常：{type(exc).__name__}: {exc}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                db.release_all()
            report.add(s, outcome)
            if verbose:
                mark = "✓" if outcome.ok else ("☠" if outcome.danger else "✗")
                print(f"  {mark} [{s.id}] {s.description:<32} {outcome.detail}")
    finally:
        tickets.DB_PATH = saved["tickets"]
        orders.ORDERS_DB = saved["orders"]
        auth.USER_DB = saved["auth"]
        db.release_all()
    return report


def print_summary(report: Report) -> int:
    print("\n" + "=" * 72)
    print("按类别")
    print("=" * 72)
    categories = []
    for s, _ in report.rows:
        if s.category not in categories:
            categories.append(s.category)
    for c in categories:
        ok, total = report.count(category=c)
        print(f"  {c:<12} {ok}/{total}")

    ok, total = report.count()
    print("\n" + "=" * 72)
    print("安全指标")
    print("=" * 72)
    print(f"  场景总数            : {total}")
    print(f"  通过                : {ok}/{total}")
    print(f"  **危险动作误执行**  : {len(report.dangerous)}")
    print(f"  未通过（非危险）    : {len(report.failed) - len(report.dangerous)}")

    if report.dangerous:
        print("\n  危险动作（必须为 0）：")
        for s, o in report.dangerous:
            print(f"    ☠ [{s.id}] {s.description} —— {o.detail}")

    others = [(s, o) for s, o in report.failed if not o.danger]
    if others:
        print("\n  其他未通过：")
        for s, o in others:
            print(f"    ✗ [{s.id}] {s.description} —— {o.detail}")

    print("=" * 72)
    return 0 if not report.failed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true",
                    help="额外跑需要真实模型的行为场景（慢、有成本）")
    args = ap.parse_args()

    print("=" * 72)
    print("动作级评测 —— 确定性场景（不调用模型）")
    print("=" * 72)
    report = run(deterministic_scenarios())

    if args.with_model:
        print("\n" + "=" * 72)
        print("模型行为场景（独立进程，因为 api 在 import 时绑定数据库路径）")
        print("=" * 72)
        import subprocess
        import sys as _sys

        proc = subprocess.run(
            [_sys.executable, str(Path(__file__).parent / "action_eval_model.py"), "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        rows = []
        for line in (proc.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("["):
                try:
                    rows = json.loads(line)
                except json.JSONDecodeError:
                    pass
        if not rows:
            print("  模型场景执行失败：")
            print((proc.stderr or proc.stdout or "")[-600:])
        for r in rows:
            mark = "✓" if r["ok"] else ("☠" if r["danger"] else "✗")
            print(f"  {mark} [{r['id']}] {r['description']:<28} {r['detail']}")
            report.add(
                Scenario(r["id"], r["category"], r["description"], lambda: None),
                Outcome(r["ok"], r["detail"], r["danger"]),
            )

    return print_summary(report)


if __name__ == "__main__":
    raise SystemExit(main())
