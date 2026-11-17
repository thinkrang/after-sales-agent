"""Structured tool results: a code the model can act on, not a sentence it must guess.

工具返回值是给模型做决策的，不是给人读的。一句"查询失败"模型只能瞎猜：
是重试、是换个订单号、还是该问用户？所以失败也返回结构化信封。

契约：失败一律返回信封（ok/code/retryable/hint），成功返回该工具的原生载荷
（订单是字段，检索是证据文本）。只有失败需要模型做决策，成功不需要额外解释。

只依赖 orders（不碰检索栈），因此轻量测试可以直接导入，不必装 chromadb。
"""
from __future__ import annotations

import json

import orders

# 封闭枚举：新增码必须同时说明模型该做什么，否则模型无法据此决策。
NOT_FOUND = "not_found"
DENIED = "denied"
TRANSIENT = "transient"
CONFLICT = "conflict"
INVALID_ARGS = "invalid_args"
NO_EVIDENCE = "no_evidence"
OUTCOME_UNKNOWN = "outcome_unknown"

CODES = frozenset({NOT_FOUND, DENIED, TRANSIENT, CONFLICT, INVALID_ARGS, NO_EVIDENCE, OUTCOME_UNKNOWN})

# 只有临时故障值得重试。not_found 重试一百次还是找不到，
# 反而会触发循环边界，把一次本可以追问的交互变成一次"停止"。
# no_evidence 同理：同一问法再查一次，检索结果不会变。
RETRYABLE = frozenset({TRANSIENT})


def ok(**data) -> str:
    return json.dumps({"ok": True, **data}, ensure_ascii=False)


def fail(code: str, hint: str, **data) -> str:
    if code not in CODES:
        raise ValueError(f"未知的工具失败码：{code}")
    return json.dumps(
        {"ok": False, "code": code, "retryable": code in RETRYABLE, "hint": hint, **data},
        ensure_ascii=False,
    )


def order_snapshot(order: dict, decision: dict) -> str:
    """成功结果保持原有字段名，避免已有提示词和前端解析失效。"""
    return ok(
        order_id=order["order_id"],
        product_name=order["product_name"],
        amount=f"{order['amount_cents'] / 100:.2f}",
        status=order["status"],
        refundable=decision["eligible"],
        reason=decision["reason"],
    )


def order_lookup_failure(exc: Exception) -> str:
    """把异常翻译成模型能据以决策的失败码。

    订单不存在是"确定没有"，不是"暂时查不到"——这两者的下一步动作相反：
    前者要问用户核对订单号，后者可以再试一次。
    """
    if isinstance(exc, orders.OrderNotFound):
        return fail(
            NOT_FOUND,
            "订单不存在或不属于当前账号；向用户确认订单号，不要猜测、不要换号重试",
        )
    return fail(
        TRANSIENT,
        "订单服务暂时不可用，可以重试一次；仍失败则告知用户稍后再试或转人工",
    )


def no_evidence(reason: str) -> str:
    """检索到了文档，但没有足以回答的证据。

    这不是故障，是"知识库确实没写"。所以不可重试——同一问法再查一次，
    召回结果不会变；正确动作是明确告知无法确认，或让用户补充条件。
    """
    return fail(
        NO_EVIDENCE,
        "知识库没有足以回答的可靠证据；明确告知用户无法确认，不要编造，"
        "也不要用同一问法重试",
        reason=reason,
    )


def retrieval_failure() -> str:
    """检索链路本身出错（索引缺失、下游超时等），与"没有证据"是两回事。

    不接收异常对象：下游报错可能带内部地址或 SQL，返回给模型的是稳定提示，
    细节只进日志。
    """
    return fail(
        TRANSIENT,
        "知识库检索暂时不可用，可以重试一次；仍失败则告知用户稍后再试或转人工",
    )
