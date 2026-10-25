"""知识库管理接口测试：异步索引、去重、列表、删除。

上传接口已改为异步：POST /knowledge/upload 立刻返回 task_id，
索引在后台执行（Celery，或在 Redis 不可用时降级为进程内线程）。
所以每个用例都要轮询任务状态，不能像以前那样直接断言返回体。

这些用例会真实读写 Chroma 集合，每条都要自己清理，
否则跑几次测试知识库里就全是垃圾文档。
"""

import time
import hashlib

import pytest

TEST_DOC = "测试文档：这是一篇用于自动化测试的临时文档。\n内容仅用于验证上传链路。"


def _upload(client, filename: str, content: str, mime="text/plain"):
    return client.post(
        "/knowledge/upload",
        files={"file": (filename, content.encode("utf-8"), mime)},
    )


def _wait_task(client, task_id: str, timeout: float = 60.0) -> dict:
    """轮询任务直到终态。索引要跑 embedding，给它足够时间。"""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/knowledge/tasks/{task_id}").json()
        if last["status"] in ("SUCCESS", "FAILURE"):
            return last
        time.sleep(0.2)
    raise AssertionError(f"任务超时未完成，最后状态：{last}")


def _upload_and_wait(client, filename: str, content: str) -> dict:
    """上传并等到索引完成，返回任务终态。"""
    r = _upload(client, filename, content)
    assert r.status_code == 200, r.text
    return _wait_task(client, r.json()["task_id"])


def _delete_if_exists(client, filename):
    docs = client.get("/knowledge/documents").json()["documents"]
    for d in docs:
        if d["filename"] == filename and d.get("doc_id"):
            client.delete(f"/knowledge/documents/{d['doc_id']}")


# ---------------------------------------------------------------- 列表


def test_list_returns_documents(client):
    body = client.get("/knowledge/documents").json()
    assert body["total"] == len(body["documents"])
    assert body["total"] > 0
    assert all(d.get("filename") for d in body["documents"])


# ---------------------------------------------------------------- 异步上传


def test_upload_returns_task_id_immediately(client):
    """接口必须立刻返回 task_id，不能等索引跑完。"""
    name = "pytest_async_contract.txt"
    _delete_if_exists(client, name)

    r = _upload(client, name, TEST_DOC)
    assert r.status_code == 200

    body = r.json()
    assert body["task_id"]
    assert body["status"] == "queued"
    assert body["doc_id"] == hashlib.sha256(
        b"acme\0" + TEST_DOC.encode("utf-8")
    ).hexdigest()
    # mode 说明走的哪条路：celery（有 Redis）或 inline（降级）
    assert body["mode"] in ("celery", "inline")

    _wait_task(client, body["task_id"])
    _delete_if_exists(client, name)


def test_upload_indexes_document(client):
    name = "pytest_upload_case.txt"
    _delete_if_exists(client, name)

    done = _upload_and_wait(client, name, TEST_DOC)
    assert done["status"] == "SUCCESS"
    assert done["result"]["filename"] == name

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]
    assert name in names

    _delete_if_exists(client, name)


def test_upload_same_content_twice_is_deduplicated(client):
    """同样内容再传一次应识别为重复，不能重复入库。

    幂等键是租户 + 内容哈希，所以两次 doc_id 必须一致，
    第二次任务的 result.duplicate 应为 True。
    """
    name = "pytest_dedup_case.txt"
    _delete_if_exists(client, name)

    first = _upload_and_wait(client, name, TEST_DOC)
    second = _upload_and_wait(client, name, TEST_DOC)

    assert first["result"]["duplicate"] is False
    assert second["result"]["duplicate"] is True
    assert first["result"]["doc_id"] == second["result"]["doc_id"]

    _delete_if_exists(client, name)


def test_task_status_of_unknown_id(client):
    body = client.get("/knowledge/tasks/local_doesnotexist").json()
    assert body["status"] == "PENDING"


def test_task_id_length_is_validated(client):
    r = client.get("/knowledge/tasks/" + "x" * 100)
    assert r.status_code == 400


# ---------------------------------------------------------------- 参数校验


@pytest.mark.parametrize("filename", ["bad.exe", "bad.docx", "bad.zip"])
def test_upload_unsupported_extension_rejected(client, filename):
    assert _upload(client, filename, "x").status_code == 400


def test_upload_empty_file_rejected(client):
    assert _upload(client, "empty.txt", "").status_code in (400, 422)


def test_upload_too_large_rejected(client):
    """5MB 上限，超了要 413 而不是等后台任务失败。"""
    big = "x" * (5 * 1024 * 1024 + 10)
    assert _upload(client, "big.txt", big).status_code == 413


# ---------------------------------------------------------------- 删除


def test_delete_document(client):
    name = "pytest_delete_case.txt"
    _delete_if_exists(client, name)

    done = _upload_and_wait(client, name, TEST_DOC)
    doc_id = done["result"]["doc_id"]

    assert client.delete(f"/knowledge/documents/{doc_id}").status_code == 200

    names = [d["filename"] for d in client.get("/knowledge/documents").json()["documents"]]
    assert name not in names


def test_delete_missing_document_returns_404(client):
    r = client.delete("/knowledge/documents/" + "0" * 64)
    assert r.status_code == 404
