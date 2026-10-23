"""SQLite 连接工厂：统一开启 WAL 与忙等超时。

为什么需要这个模块
------------------
SQLite 默认 `journal_mode=DELETE`，写操作会阻塞读。单机做并发服务时，
两个请求同时写就会有一方拿到 `database is locked`。

WAL（Write-Ahead Logging）让读写不互相阻塞：写者追加到 `-wal` 文件，
读者读主库快照。这是 SQLite 做并发服务的标准配置，且设置一次后
对数据库文件持久生效。

`busy_timeout` 解决的是另一个问题：多个写者同时到达时，后来者不应该
立刻失败，而应该等一会儿重试。SQLite 默认是 0（立刻报错），并发下
几乎必然失败。给 5 秒缓冲，足够覆盖本项目毫秒级的写入。

注意
----
WAL 允许多个读者 + 一个写者并发。它不解决"多个写者"的问题——
写者之间仍然串行。真要高并发写入，得换 PostgreSQL。
这条限制写在这里，是因为它决定了本项目的并发上限。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


def enable_wal(path: str | Path) -> None:
    """把数据库切到 WAL 模式。**只在初始化时调用一次。**

    为什么不能放在 connect() 里
    ---------------------------
    `PRAGMA journal_mode=WAL` 需要短暂获取排他锁。并发建连时多个线程同时
    执行它会互相冲突，实测报 `attempt to write a readonly database`。
    而 journal_mode 是写进数据库文件头的持久属性——设一次就一直有效，
    每次连接都设既没必要也不安全。
    """
    try:
        conn = sqlite3.connect(path, timeout=10)
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        conn.close()
        if str(mode).lower() != "wal":
            print(f"[db] {path} 未能切换到 WAL（当前 {mode}），并发写入会退化为串行")
    except sqlite3.Error as exc:
        print(f"[db] {path} 开启 WAL 失败：{exc}")


def connect(path: str | Path, timeout: float = 10.0) -> sqlite3.Connection:
    """打开连接并套上**每连接**级别的并发 PRAGMA。

    timeout 是 Python 层的连接超时；busy_timeout 是 SQLite 层的忙等，
    两者作用不同，都保留。
    """
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    # NORMAL 在 WAL 下足够安全（崩溃最多丢最后一个事务，不会损坏库），
    # 换来明显更快的写入。FULL 是默认值，对日志类写入过于保守。
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# 长连接池：每个数据库保持一个连接不关闭，用于钉住 -wal / -shm 文件
_keepers: dict[str, sqlite3.Connection] = {}


def hold(path: str | Path) -> None:
    """保持一个长连接，确保 WAL 的共享内存文件始终存在。

    为什么需要
    ----------
    压测实测：8 个线程同时对同一张表做 UPDATE，7 个报
    `attempt to write a readonly database`；但只要每个线程先做一次 SELECT，
    失败数就变成 0。

    原因是**并发首次写入时，多个连接在竞争创建 -wal / -shm 文件**。
    Windows 上这个竞争会随机失败，SQLite 把失败表现为"数据库只读"。
    提前建好这两个文件并保持打开，后续连接就能直接复用，不再竞争。

    这是 SQLite 单机并发的一个已知陷阱——文档里不会明说，只能压出来。
    """
    key = str(Path(path).resolve())
    if key in _keepers:
        return
    conn = connect(path)
    # 一次读操作就会触发 WAL 文件创建（写入路径才需要，但读也会建立 shm）
    conn.execute("SELECT 1").fetchone()
    _keepers[key] = conn


def release_all() -> None:
    """关闭所有长连接。测试收尾时调用。"""
    for conn in _keepers.values():
        try:
            conn.close()
        except sqlite3.Error:
            pass
    _keepers.clear()


@contextmanager
def session(path: str | Path, timeout: float = 10.0):
    """一次事务的完整生命周期：提交、回滚、**关闭**。

    为什么需要它
    ------------
    Python 的 `with sqlite3.connect(...) as conn` 只提交/回滚，**不关闭连接**。
    原先代码全用这种写法，导致每次请求泄漏一个连接句柄——压测几百个请求后
    会积累大量打开的 WAL 句柄，Windows 上表现为随机的
    `attempt to write a readonly database`。

    这个上下文管理器把"提交 + 关闭"绑在一起，调用方不用记得 close。
    """
    conn = connect(path, timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
