"""风险分级规则的单测。

规则是纯函数，所以这些测试不需要数据库、不需要模型，毫秒级。
重点不是"能不能跑"，而是**每条规则的边界和理由是否准确**——
风控规则写错一个边界，放行的就是不该放行的单子。
"""

from datetime import datetime, timedelta

import pytest

import risk

NOW = datetime(2026, 10, 25, 12, 0, 0)
ELIGIBLE = {"eligible": True, "code": "eligible", "reason": "可退"}


def order(amount_cents: int, *, days_left: float = 6.0, status: str = "delivered") -> dict:
    """构造订单。days_left 是距退款截止还剩几天，负数表示已过期。"""
    until = NOW + timedelta(days=days_left)
    return {
        "order_id": "A100",
        "amount_cents": amount_cents,
        "status": status,
        "refundable_until": until.strftime("%Y-%m-%d %H:%M:%S"),
    }


def assess(o, reason="质量问题", recent=0):
    return risk.assess(o, ELIGIBLE, reason=reason, recent_executed=recent, now=NOW)


# ---------------------------------------------------------------- 低风险基线


def test_small_amount_quality_reason_is_low():
    a = assess(order(12_900))
    assert a.tier == risk.TIER_LOW
    assert a.required_role == risk.ROLE_CUSTOMER
    assert a.reasons == ()
    assert a.needs_manager is False


# ---------------------------------------------------------------- R1 大额


def test_amount_at_threshold_is_high():
    """等于阈值就要转主管——边界取闭区间，避免"刚好 500 元"漏网。"""
    a = assess(order(risk.HIGH_AMOUNT_CENTS))
    assert a.tier == risk.TIER_HIGH
    assert a.required_role == risk.ROLE_APPROVER
    assert any("自动审批上限" in r for r in a.reasons)


def test_amount_just_below_threshold_is_low():
    a = assess(order(risk.HIGH_AMOUNT_CENTS - 1))
    assert a.tier == risk.TIER_LOW


# ---------------------------------------------------------------- R2 边界政策


def test_near_expiry_is_high():
    a = assess(order(12_900, days_left=0.5))
    assert a.tier == risk.TIER_HIGH
    assert any("边界政策" in r for r in a.reasons)


def test_plenty_of_time_left_is_low():
    a = assess(order(12_900, days_left=6))
    assert a.tier == risk.TIER_LOW


def test_expired_order_does_not_trigger_edge_rule():
    """已过期由 refund_policy 直接判不可退，不该被算成"边界"。

    否则会把"本来就退不了"的单子误标成高风险，浪费审批资源。
    """
    a = assess(order(12_900, days_left=-1))
    assert not any("边界政策" in r for r in a.reasons)


# ---------------------------------------------------------------- R3 异常频次


def test_frequent_refunds_are_high():
    a = assess(order(12_900), recent=risk.RECENT_REFUND_LIMIT)
    assert a.tier == risk.TIER_HIGH
    assert any("笔退款执行成功" in r for r in a.reasons)


def test_one_recent_refund_is_still_low():
    a = assess(order(12_900), recent=risk.RECENT_REFUND_LIMIT - 1)
    assert a.tier == risk.TIER_LOW


# ---------------------------------------------------------------- 被删掉的规则（回归）


@pytest.mark.parametrize("reason", ["不想要了", "尺码不合适", "买错了", ""])
def test_no_reason_return_is_not_high_risk(reason):
    """七天无理由退货是电商标准流程，不该算高风险。

    曾经有过一条「已签收 + 非质量原因 → 高风险」的规则，结果是绝大多数
    正常退款都被标成高风险，主管要审所有单子，分级就失去意义了。
    这条测试防止它被加回来。
    """
    a = assess(order(12_900), reason=reason)
    assert a.tier == risk.TIER_LOW


def test_unshipped_order_is_not_high_risk():
    a = assess(order(12_900, status="pending"), reason="不想要了")
    assert a.tier == risk.TIER_LOW


# ---------------------------------------------------------------- 组合与确定性


def test_multiple_rules_accumulate_reasons():
    a = assess(order(128_000, days_left=0.5), reason="不想要了", recent=3)
    assert a.tier == risk.TIER_HIGH
    assert len(a.reasons) >= 3, f"应该命中多条规则，实际只有 {a.reasons}"


def test_assessment_is_deterministic():
    """同样的输入必须给同样的结论——风控结论要能复现，否则审计没法做。"""
    o = order(128_000)
    first = assess(o, reason="不想要了", recent=2)
    second = assess(o, reason="不想要了", recent=2)
    assert first == second


def test_rule_version_is_recorded():
    a = assess(order(12_900))
    assert a.rule_version == risk.RISK_RULES_VERSION
    assert a.rule_version  # 不能是空串，否则工单上无法追溯


def test_as_dict_is_json_friendly():
    import json

    payload = assess(order(128_000)).as_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False))["tier"] == risk.TIER_HIGH
