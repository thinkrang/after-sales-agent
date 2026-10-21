"""会话管理接口测试：增删改查 + 会话隔离。"""


def test_create_returns_id_and_default_title(client):
    r = client.post("/sessions", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("chat_")
    assert body["title"] == "新对话"

    client.delete(f"/sessions/{body['id']}")


def test_create_with_custom_title(client):
    r = client.post("/sessions", json={"title": "自定义标题"})
    assert r.json()["title"] == "自定义标题"
    client.delete(f"/sessions/{r.json()['id']}")


def test_list_contains_created_session(client, session_id):
    ids = [s["id"] for s in client.get("/sessions").json()["sessions"]]
    assert session_id in ids


def test_rename(client, session_id):
    r = client.patch(f"/sessions/{session_id}", json={"title": "改过的名字"})
    assert r.status_code == 200
    assert r.json()["title"] == "改过的名字"

    found = [s for s in client.get("/sessions").json()["sessions"] if s["id"] == session_id]
    assert found[0]["title"] == "改过的名字"


def test_rename_missing_session_returns_404(client):
    r = client.patch("/sessions/not_exist_xxx", json={"title": "x"})
    assert r.status_code == 404


def test_rename_rejects_empty_title(client, session_id):
    r = client.patch(f"/sessions/{session_id}", json={"title": ""})
    assert r.status_code == 422  # pydantic 校验拦截


def test_delete_removes_from_list(client):
    sid = client.post("/sessions", json={}).json()["id"]
    assert client.delete(f"/sessions/{sid}").status_code == 200

    ids = [s["id"] for s in client.get("/sessions").json()["sessions"]]
    assert sid not in ids


def test_delete_is_idempotent(client):
    """删两次不该报错 —— 前端重试、用户连点都不应该炸。"""
    sid = client.post("/sessions", json={}).json()["id"]
    assert client.delete(f"/sessions/{sid}").status_code == 200
    assert client.delete(f"/sessions/{sid}").status_code == 200


def test_messages_of_missing_session_returns_empty(client):
    r = client.get("/sessions/never_existed_xxx/messages")
    assert r.status_code == 200
    assert r.json()["messages"] == []
    assert r.json()["pending"] == []


def test_session_isolation(client, fake_agent):
    """核心用例：两个会话的历史不能互相串。

    这是多会话产品最容易出的 bug —— 用错 thread_id 或者缓存没按会话隔离，
    用户就会看到别人的对话。
    """
    from conftest import history

    a = client.post("/sessions", json={}).json()["id"]
    b = client.post("/sessions", json={}).json()["id"]

    fake_agent.by_thread[a] = history("A 的问题", "A 的回答")
    fake_agent.by_thread[b] = history("B 的问题")

    msg_a = client.get(f"/sessions/{a}/messages").json()["messages"]
    msg_b = client.get(f"/sessions/{b}/messages").json()["messages"]

    assert [m["text"] for m in msg_a] == ["A 的问题", "A 的回答"]
    assert [m["text"] for m in msg_b] == ["B 的问题"]

    client.delete(f"/sessions/{a}")
    client.delete(f"/sessions/{b}")
