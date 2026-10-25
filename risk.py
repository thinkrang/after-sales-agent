"""退款风险分级：确定性规则引擎。

为什么不用模型判断风险
----------------------
风控决策必须**可复现、可审计、可回滚**。如果用 LLM 判断"这笔退款风险高不高"：

- 同一笔订单两次评估可能给出不同结论，审计时无法解释；
- 调一次模型要几百毫秒，而这条判断是纯算术；
- 规则变了无法追溯——你只能翻 prompt 历史，翻不到"当时为什么放行"。

所以这里是纯函数：给定订单 + 政策判定 + 用户近期退款次数，输出确定的
分级结果。每条规则带一个稳定的 code，记录在工单上，事后可以逐条复核。

`RISK_RULES_VERSION` 随规则变化递增。工单里存下这个版本号，
就能回答"这张单是按哪一版规则批的"。

一条被删掉的规则（留个记录）
----------------------------
最初还有第四条：「已签收 + 非质量原因 → 无理由退货 → 高风险」。
理由看起来成立：无理由退货商家要承担运费和库存损失。

但跑测试立刻暴露问题：**七天无理由退货是电商的标准流程**，这条规则把
绝大多数正常退款都标成高风险，等于让主管审批所有单子——分级就白做了。

风险规则的门槛不是"看起来有道理"，而是"能不能把少数异常从大量正常里挑出来"。
一条把所有单子都命中的规则，等价于没有规则。所以删了。

分级与角色
----------
- `low`   → `customer`：客户本人确认即可（现有流程）
- `high`  → `approver`：必须有审批角色，且**不能是发起人本人**

四眼原则（four-eyes）只在 high 档生效：低风险退款让客户点一下确认是合理的，
但如果所有退款都要两个人批，客服系统就没法用了。风险分级的意义就在这里——
把审批资源集中到真正需要的单子上。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

# 规则版本：任何阈值或规则变化都必须改这个值
RISK_RULES_VERSION = "2026-10-25.1"

TIER_LOW = "low"
TIER_HIGH = "high"

ROLE_CUSTOMER = "customer"
ROLE_APPROVER = "approver"

# 单笔金额超过 500 元转主管。这个阈值应该来自业务方，不是拍脑袋——
# 这里写死是因为演示项目没有配置中心，真实系统应放进租户级配置。
HIGH_AMOUNT_CENTS = 50_000

# 距退款截止不足 24 小时：政策上仍然可退，但属于边界情形，
# 让机器单独决定容易产生争议。
WINDOW_EDGE_HOURS = 24

# 同一用户 24 小时内已执行退款达到这个数量，视为异常频次
RECENT_REFUND_LIMIT = 2
RECENT_WINDOW_HOURS = 24

@dataclass(frozen=True)
class RiskAssessment:
    tier: str
    required_role: str
    reasons: tuple[str, ...]
    rule_version: str

    @property
    def needs_manager(self) -> bool:
        return self.required_role == ROLE_APPROVER

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "required_role": self.required_role,
            "reasons": list(self.reasons),
            "rule_version": self.rule_version,
        }


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def assess(
    order: dict,
    decision: dict,
    *,
    reason: str = "",
    recent_executed: int = 0,
    now: datetime | None = None,
) -> RiskAssessment:
    """评估单笔退款的风险等级。

    参数
    ----
    order             订单记录（含 amount_cents / status / refundable_until）
    decision          orders.refund_policy() 的返回值
    reason            用户填写的退款原因
    recent_executed   该用户近 RECENT_WINDOW_HOURS 小时内已执行成功的退款笔数
    now               便于测试注入固定时间
    """
    now = now or datetime.now()
    reasons: list[str] = []

    amount_cents = int(order.get("amount_cents") or 0)

    # R1 大额：超过自动审批上限
    if amount_cents >= HIGH_AMOUNT_CENTS:
        reasons.append(
            f"金额 {amount_cents / 100:.2f} 元达到或超过自动审批上限 "
            f"{HIGH_AMOUNT_CENTS / 100:.0f} 元"
        )

    # R2 边界政策：仍在退款期内，但离截止很近
    until = _parse(order.get("refundable_until"))
    if until is not None:
        remaining = until - now
        if timedelta(0) < remaining <= timedelta(hours=WINDOW_EDGE_HOURS):
            hours = remaining.total_seconds() / 3600
            reasons.append(f"距退款期限仅剩 {hours:.1f} 小时，属边界政策")

    # R3 异常频次：短期内反复退款
    if recent_executed >= RECENT_REFUND_LIMIT:
        reasons.append(
            f"近 {RECENT_WINDOW_HOURS} 小时已有 {recent_executed} 笔退款执行成功"
        )

    if reasons:
        return RiskAssessment(
            tier=TIER_HIGH,
            required_role=ROLE_APPROVER,
            reasons=tuple(reasons),
            rule_version=RISK_RULES_VERSION,
        )
    return RiskAssessment(
        tier=TIER_LOW,
        required_role=ROLE_CUSTOMER,
        reasons=(),
        rule_version=RISK_RULES_VERSION,
    )
