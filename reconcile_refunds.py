"""Recover refunds approved before a process/network failure.

Usage:
    python reconcile_refunds.py

The script is safe to run repeatedly. Each recovery reuses the original
approval payload and ticket id as the downstream idempotency key.

失败语义
--------
对账不是"重试到成功为止"。工单带 attempts 计数，重试到
``tickets.MAX_EXECUTION_ATTEMPTS`` 次仍失败才转 ``failed``。

为什么可以这样终结：下游退款按工单 ID 去重，**如果它真的成功过，
重试会返回原记录而不是继续失败**。所以持续失败说明确实没退成，
这时候终结是安全的，不会出现"工单说失败但钱已经退了"。

退出码：
    0  全部恢复成功，或没有需要恢复的工单
    1  有工单恢复失败（会打印 attempts 和是否已转 failed）
"""

from __future__ import annotations

import logging

import refund_service
import tickets


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    tickets.init_tickets_table()

    rows = tickets.list_recoverable()
    if not rows:
        print("没有需要恢复的退款工单")
        return 0

    print(f"发现 {len(rows)} 条待恢复工单\n")
    failures = 0
    for ticket in rows:
        label = (
            f"{ticket['id']} (订单 {ticket['order_id']}, "
            f"{ticket['status']}, 已尝试 {ticket['attempts']} 次)"
        )
        try:
            result = refund_service.reconcile_ticket(ticket)
            print(f"✓ {label} -> {result['status']} {result.get('refund_id') or ''}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logging.exception("工单 %s 对账失败", ticket["id"])
            # 重新读一次：这次失败可能已经把工单推到 failed（尝试次数到上限）
            current = tickets.get_ticket(ticket["id"]) or {}
            status = current.get("status", "?")
            attempts = current.get("attempts", "?")
            if status == tickets.FAILED:
                tail = "，已转 failed，需人工核实"
            else:
                tail = f"，仍可重试（上限 {tickets.MAX_EXECUTION_ATTEMPTS} 次）"
            print(f"✗ {label} -> {type(exc).__name__}: {exc}")
            print(f"   当前状态 {status}，已尝试 {attempts} 次{tail}")

    print()
    if failures:
        print(f"完成：{len(rows) - failures} 条恢复成功，{failures} 条失败")
        return 1
    print(f"完成：{len(rows)} 条全部恢复成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
