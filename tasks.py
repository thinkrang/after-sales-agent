"""文档索引的异步任务层。

为什么要把索引做成异步
----------------------
上传一个 PDF 要做：解码 → 解析 → 切分 → 逐块 embedding → 写向量库。
20 篇文档的规模下也要几秒，真实场景几百页 PDF 能到几十秒。
同步做的话 HTTP 请求一直挂着，前端只能转圈，网关还可能超时断开。

异步之后：HTTP 立刻返回 task_id，前端轮询进度，用户体验完全不同。

三个设计要点
------------
1. **幂等**：doc_id = sha256(文件内容)。同一文件重复上传，
   `add_knowledge_document` 内部会识别 duplicate；任务重试同理，
   重试不会产生第二份索引。这是"至少一次投递"下的必要保护。

2. **只重试瞬时错误**：文件格式错误（ValueError）重试一百次也不会好，
   直接抛出；网络/IO 抖动才重试。

3. **降级**：Redis 不可用时退化为进程内后台线程。
   这不是偷懒——它保证项目在没有 broker 的环境里也能跑通演示。
   生产必须用真 broker，进程内方案进程一挂任务就丢了。
"""

import base64
import logging
import threading
import time
import uuid

from celery_app import BROKER_URL, celery_app
from rag import add_knowledge_document
from retrieval import refresh_corpus

# 降级模式下把任务状态放在内存里。进程重启即丢，仅用于单机演示。
_LOCAL_TASKS: dict[str, dict] = {}
_local_lock = threading.Lock()

# broker 探测结果缓存，避免每个请求都去 ping 一次 Redis
_broker_state = {"checked_at": 0.0, "ok": False}
BROKER_CHECK_TTL = 10.0


# ---------------------------------------------------------------- 任务定义


@celery_app.task(bind=True, name="knowledge.index_document", max_retries=3)
def index_document(self, filename: str, content_b64: str, doc_id: str) -> dict:
    """解析文档并写入向量库。

    参数里传 base64 而不是 bytes：Celery 用 JSON 序列化，
    bytes 不能直接过。项目限制单文件 5MB，base64 膨胀 33% 可接受。
    """
    raw = base64.b64decode(content_b64)

    self.update_state(state="PROGRESS", meta={"stage": "解析并写入向量库"})

    try:
        result = add_knowledge_document(filename=filename, file_bytes=raw)
    except ValueError:
        # 文件格式/编码问题，重试无意义
        raise
    except Exception as exc:
        logging.warning("索引任务失败，准备重试（第 %s 次）：%s", self.request.retries + 1, exc)
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

    # 新增文档后要让 BM25 的语料缓存失效，否则关键词检索看不到新文档
    refresh_corpus()
    return result


# ---------------------------------------------------------------- broker 探测


def _broker_available() -> bool:
    now = time.time()
    if now - _broker_state["checked_at"] < BROKER_CHECK_TTL:
        return _broker_state["ok"]

    ok = False
    try:
        import redis

        client = redis.Redis.from_url(
            BROKER_URL,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        ok = True
    except Exception as exc:
        logging.info("Redis 不可用，索引任务降级为进程内执行：%s", exc)

    _broker_state.update(checked_at=now, ok=ok)
    return ok


# ---------------------------------------------------------------- 派发与查询


def _run_inline(task_id: str, filename: str, content: bytes) -> None:
    try:
        result = add_knowledge_document(filename=filename, file_bytes=content)
        refresh_corpus()
        with _local_lock:
            _LOCAL_TASKS[task_id] = {"status": "SUCCESS", "result": result}
    except Exception as exc:
        logging.exception("进程内索引任务失败")
        with _local_lock:
            _LOCAL_TASKS[task_id] = {"status": "FAILURE", "error": str(exc)}


def submit_index_task(filename: str, content: bytes, doc_id: str) -> tuple[str, str]:
    """派发索引任务，返回 (task_id, mode)。mode 为 celery 或 inline。"""
    payload = base64.b64encode(content).decode("ascii")

    if _broker_available():
        try:
            async_result = index_document.delay(filename, payload, doc_id)
            return async_result.id, "celery"
        except Exception as exc:
            logging.warning("派发 Celery 任务失败，降级为进程内执行：%s", exc)

    task_id = "local_" + uuid.uuid4().hex[:12]
    with _local_lock:
        _LOCAL_TASKS[task_id] = {"status": "STARTED", "stage": "解析并写入向量库"}

    threading.Thread(
        target=_run_inline,
        args=(task_id, filename, content),
        daemon=True,
    ).start()

    return task_id, "inline"


def get_task_status(task_id: str) -> dict:
    """查询任务状态。Celery 任务和降级任务走不同路径。"""
    if task_id.startswith("local_"):
        with _local_lock:
            state = dict(_LOCAL_TASKS.get(task_id) or {})
        if not state:
            return {"task_id": task_id, "status": "PENDING", "mode": "inline"}
        return {"task_id": task_id, "mode": "inline", **state}

    from celery.result import AsyncResult

    res = AsyncResult(task_id, app=celery_app)
    payload = {"task_id": task_id, "status": res.state, "mode": "celery"}

    if res.state == "SUCCESS":
        payload["result"] = res.result
    elif res.state == "FAILURE":
        payload["error"] = str(res.info)
    elif res.state == "PROGRESS":
        payload["stage"] = (res.info or {}).get("stage")

    return payload
