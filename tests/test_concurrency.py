"""并发与数据库连接层测试。

这个文件守两件事：
    1. `db` 模块的 PRAGMA / 连接生命周期正确（不泄漏、有 WAL）
    2. **系统没有被串行化** —— 这是回归测试的核心

第 2 条尤其重要：曾经代码里有一把全局锁包住整个 agent 调用，
把吞吐压到并发数的 1/N。这个测试会在有人重新引入全局锁时失败。
"""

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import db


# ---------------------------------------------------------------- 连接层


def test_hold_creates_wal_files(tmp_path):
    """hold() 之后 -wal / -shm 应该存在——它们是并发写入的前提。"""
    path = tmp_path / "t.db"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.execute("INSERT INTO t VALUES (1, 'x')")
    c.commit()
    c.close()

    db.enable_wal(path)
    db.hold(path)
    try:
        assert (tmp_path / "t.db-wal").exists(), "WAL 文件没建出来"
        assert (tmp_path / "t.db-shm").exists(), "共享内存文件没建出来"
    finally:
        db.release_all()


def test_wal_mode_is_persistent(tmp_path):
    """journal_mode 是写进文件头的持久属性，重开连接仍然是 WAL。"""
    path = tmp_path / "t.db"
    sqlite3.connect(path).close()
    db.enable_wal(path)

    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()


def test_session_commits(tmp_path):
    path = tmp_path / "t.db"
    with db.session(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT v FROM t WHERE id=1").fetchone()[0] == "hello"
    conn.close()


def test_session_rolls_back_on_error(tmp_path):
    path = tmp_path / "t.db"
    with db.session(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")

    with pytest.raises(RuntimeError):
        with db.session(path) as conn:
            conn.execute("INSERT INTO t VALUES (1, 'x')")
            raise RuntimeError("boom")

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0, "异常后应该回滚"
    conn.close()


def test_session_closes_connection(tmp_path):
    """连接必须被关闭。不关的话压测几百个请求就会耗尽句柄。"""
    path = tmp_path / "t.db"
    with db.session(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        leaked = conn

    with pytest.raises(sqlite3.ProgrammingError):
        leaked.execute("SELECT 1")


def test_busy_timeout_is_set(tmp_path):
    path = tmp_path / "t.db"
    with db.session(path) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


# ---------------------------------------------------------------- 并发写入


def test_concurrent_writes_succeed(tmp_path):
    """16 线程 × 3 轮并发写，一次都不能失败。

    这条守的是 WAL 文件竞争问题：如果 hold() 没被调用，
    并发首次写入会报 `attempt to write a readonly database`。
    """
    path = tmp_path / "t.db"
    with db.session(path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'init')")

    db.enable_wal(path)
    db.hold(path)

    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for _ in range(3):
                with db.session(path) as conn:
                    conn.execute("UPDATE t SET v = ? WHERE id = 1", (str(n),))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(worker, range(16)))
    finally:
        db.release_all()

    assert not errors, f"并发写失败 {len(errors)} 次：{errors[:3]}"


# ---------------------------------------------------------------- 串行化回归


class _StubAgent:
    """固定延迟的假 agent。延迟恒定，唯一的变量就是并发控制。"""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.peak = 0
        self._active = 0

    def invoke(self, payload, config=None, **kwargs):
        self._active += 1
        self.peak = max(self.peak, self._active)
        try:
            time.sleep(self.delay)
            return {"messages": [SimpleNamespace(content="ok")]}
        finally:
            self._active -= 1


def test_agent_calls_are_not_serialized():
    """5 个并发请求的总耗时应该接近单次延迟，而不是 5 倍。

    曾经这里有一把全局锁，把 agent 调用串行化（并行度 1.00x）。
    如果哪天有人又把锁加回来，这条测试会失败。
    """
    import httpx

    from api import app

    delay = 0.25
    n = 5

    async def run() -> tuple[float, _StubAgent]:
        async with app.router.lifespan_context(app):
            stub = _StubAgent(delay)
            app.state.agent = stub

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://test", timeout=60) as client:
                async def one(i: int):
                    return await client.post(
                        "/chat", json={"thread_id": f"conc_{i}", "message": "hi"}
                    )

                start = time.perf_counter()
                results = await asyncio.gather(*(one(i) for i in range(n)))
                elapsed = time.perf_counter() - start

            assert all(r.status_code == 200 for r in results), "并发请求不该失败"
            return elapsed, stub

    elapsed, stub = asyncio.run(run())

    # 完全串行需要 n * delay = 1.25s；允许一些线程调度开销
    assert elapsed < delay * n * 0.6, (
        f"并发耗时 {elapsed:.2f}s 接近串行的 {delay * n:.2f}s，说明又被串行化了"
    )
    assert stub.peak > 1, f"agent 并发峰值只有 {stub.peak}，请求没有真正重叠"
