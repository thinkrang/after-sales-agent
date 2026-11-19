"""Prune conversation checkpoints for sessions that have been idle for N days.

Usage:
    python -m scripts.prune_checkpoints                     # dry run, 30 days
    python -m scripts.prune_checkpoints --days 7 --apply
    python -m scripts.prune_checkpoints --orphans --apply   # also threads with no session row
    python -m scripts.prune_checkpoints --apply --vacuum

只删会话历史，不删业务事实
--------------------------
订单、退款工单、退款执行记录都在各自的库里，脚本不碰。删掉的只是"聊天原文"
和侧边栏条目——留着侧边栏却点进去空白，比直接删掉更让人困惑。

审批图状态（refund_memory.sqlite）也不动：`refund_flow.settle()` 依赖它
从中断处恢复流程，清掉会让待审批工单无法收尾。

安全性
------
默认 dry run，只打印将被删除的条数；`--apply` 才写库。
先删 writes 再删 checkpoints，避免留下没有 checkpoint 的孤立写入。
服务运行中执行可能遇到 database is locked，建议低峰期或停服执行。
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

import settings
import tickets

# 和 api._now() 保持同一格式，否则字符串比较会得出错误的新旧关系。
SESSION_FMT = "%Y-%m-%d %H:%M:%S"


def cutoff(days: int) -> str:
    if days < 1:
        raise ValueError("保留天数必须大于 0")
    return (datetime.now() - timedelta(days=days)).strftime(SESSION_FMT)


def expired_sessions(conn: sqlite3.Connection, before: str, tenant: str | None = None) -> list[dict]:
    sql = "SELECT id, title, updated_at, tenant_id, user_id FROM sessions WHERE updated_at < ?"
    params: list[str] = [before]
    if tenant:
        sql += " AND tenant_id = ?"
        params.append(tenant)
    return [dict(r) for r in conn.execute(sql + " ORDER BY updated_at", params)]


def orphan_threads(memory: sqlite3.Connection, known: set[str]) -> list[str]:
    """有 checkpoint 但没有会话行——通常是评测或迁移留下的。"""
    rows = memory.execute("SELECT DISTINCT thread_id FROM checkpoints")
    return sorted(r[0] for r in rows if r[0] not in known)


def counts(memory: sqlite3.Connection, thread_id: str) -> tuple[int, int]:
    cp = memory.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (thread_id,)).fetchone()[0]
    wr = memory.execute("SELECT COUNT(*) FROM writes WHERE thread_id=?", (thread_id,)).fetchone()[0]
    return cp, wr


def prune(
    sessions_db: Path,
    memory_db: Path,
    *,
    days: int = 30,
    tenant: str | None = None,
    apply: bool = False,
    orphans: bool = False,
    vacuum: bool = False,
    tickets_db: Path | None = None,
) -> dict:
    before = cutoff(days)
    if not Path(sessions_db).exists():
        raise FileNotFoundError(f"会话库不存在：{sessions_db}")
    if not Path(memory_db).exists():
        raise FileNotFoundError(f"记忆库不存在：{memory_db}")

    report: dict = {"cutoff": before, "applied": apply, "threads": [], "protected": [],
                    "orphans": [], "checkpoints": 0, "writes": 0, "sessions": 0}

    with closing(sqlite3.connect(sessions_db, timeout=10)) as sconn, \
         closing(sqlite3.connect(memory_db, timeout=10)) as mconn:
        sconn.row_factory = sqlite3.Row
        ticket_path = tickets_db if tickets_db is not None else Path(sessions_db).with_name("tickets.sqlite")
        if tickets_db is not None and not ticket_path.is_file():
            raise FileNotFoundError(f"工单库不存在：{ticket_path}")
        expired = []
        for row in expired_sessions(sconn, before, tenant):
            if ticket_path.is_file() and not tickets.close_conversation(
                row["tenant_id"], row["user_id"], row["id"], path=ticket_path, apply=apply
            ):
                report["protected"].append(row["id"])
                continue
            expired.append(row)
        known = {r[0] for r in sconn.execute("SELECT id FROM sessions")}
        extra = orphan_threads(mconn, known) if orphans else []

        for row in expired:
            cp, wr = counts(mconn, row["id"])
            report["threads"].append({"id": row["id"], "title": row["title"],
                                      "updated_at": row["updated_at"],
                                      "checkpoints": cp, "writes": wr})
            report["checkpoints"] += cp
            report["writes"] += wr
        for tid in extra:
            cp, wr = counts(mconn, tid)
            report["orphans"].append({"id": tid, "checkpoints": cp, "writes": wr})
            report["checkpoints"] += cp
            report["writes"] += wr

        report["sessions"] = len(expired)

        if not apply:
            return report

        targets = [row["id"] for row in expired] + extra
        with mconn:  # 一个事务，失败整体回滚
            for tid in targets:
                mconn.execute("DELETE FROM writes WHERE thread_id=?", (tid,))
                mconn.execute("DELETE FROM checkpoints WHERE thread_id=?", (tid,))
        with sconn:
            for row in expired:
                sconn.execute("DELETE FROM sessions WHERE id=?", (row["id"],))

        if vacuum:
            mconn.execute("VACUUM")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30, help="保留最近多少天（默认 30）")
    parser.add_argument("--tenant", help="只清理指定租户")
    parser.add_argument("--apply", action="store_true", help="真正删除；默认只预览")
    parser.add_argument("--orphans", action="store_true", help="同时清理没有会话行的线程")
    parser.add_argument("--vacuum", action="store_true", help="清理后回收空间")
    parser.add_argument("--sessions", type=Path, default=settings.DATA_DIR / "sessions.sqlite")
    parser.add_argument("--memory", type=Path, default=settings.DATA_DIR / "chat_memory.sqlite")
    parser.add_argument("--tickets", type=Path, help="自定义工单库；默认查找会话库旁的 tickets.sqlite")
    args = parser.parse_args()

    try:
        report = prune(args.sessions, args.memory, days=args.days, tenant=args.tenant,
                       apply=args.apply, orphans=args.orphans, vacuum=args.vacuum, tickets_db=args.tickets)
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        parser.error(str(exc))

    mode = "已删除" if report["applied"] else "预览（加 --apply 才会删除）"
    print(f"截止时间：{report['cutoff']}　模式：{mode}\n")
    for row in report["threads"]:
        print(f"  {row['id']:<24} {row['updated_at']}  checkpoint {row['checkpoints']:>3}  "
              f"writes {row['writes']:>3}  {row['title'][:20]}")
    for row in report["orphans"]:
        print(f"  {row['id']:<24} (无会话行)          checkpoint {row['checkpoints']:>3}  "
              f"writes {row['writes']:>3}")
    print(f"\n线程 {len(report['threads'])} 个（另有无会话行 {len(report['orphans'])} 个），"
          f"checkpoint {report['checkpoints']} 条，writes {report['writes']} 条，"
          f"会话行 {report['sessions']} 条")
    if report["protected"]:
        print(f"保留 {len(report['protected'])} 个仍有退款待处理的会话")
    if not report["threads"] and not report["orphans"]:
        print("没有需要清理的会话。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
