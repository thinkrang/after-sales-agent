"""Recover refunds approved before a process/network failure.

Usage:
    python reconcile_refunds.py

The script is safe to run repeatedly. Each recovery reuses the original
approval payload and ticket id as the downstream idempotency key.
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

    failures = 0
    for ticket in rows:
        try:
            result = refund_service.reconcile_ticket(ticket)
            print(
                f"已恢复工单 {ticket['id']} -> "
                f"{result['status']} {result.get('refund_id') or ''}"
            )
        except Exception as exc:
            failures += 1
            logging.exception("工单 %s 对账失败", ticket["id"])
            print(f"恢复失败 {ticket['id']}: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
