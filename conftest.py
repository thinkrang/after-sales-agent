"""pytest 公共夹具。

两个关键设计
------------
1. client 是 session 级的：TestClient 进入时会触发 FastAPI 的 lifespan，
   而 lifespan 要加载 embedding 模型（约 11 秒）。每个测试重建一次太浪费。

2. 提供一个 FakeAgent 替换 app.state.agent，用来测 SSE 协议和审批流程。
   这样接口测试不需要真实调用大模型 —— 毫秒级、零成本、结果确定。
   真实模型的端到端测试用 @pytest.mark.llm 标记，默认不跑。
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_ticket_db(tmp_path_factory):
    """Point all runtime SQLite databases at isolated temporary files.

    没有这道隔离，测 SSE 协议时会真的往 tickets.sqlite 写工单
    （_stream_agent 发现待审批动作就会建单），跑几次测试数据库里
    就全是 A1 / C3 这种假订单。
    """
    import auth
    import orders
    import tickets

    root = tmp_path_factory.mktemp("runtime-db")
    originals = {
        "tickets": tickets.DB_PATH,
        "auth": auth.USER_DB,
        "orders": orders.ORDERS_DB,
    }
    tickets.DB_PATH = root / "tickets_test.sqlite"
    auth.USER_DB = root / "auth_test.sqlite"
    orders.ORDERS_DB = root / "orders_test.sqlite"
    tickets.init_tickets_table()
    orders.init_order_db()
    yield root
    tickets.DB_PATH = originals["tickets"]
    auth.USER_DB = originals["auth"]
    orders.ORDERS_DB = originals["orders"]
    import db

    db.release_all()


@pytest.fixture(scope="session")
def client(isolate_ticket_db):
    """整个测试会话共用一个 TestClient（只加载一次模型）。"""
    from fastapi.testclient import TestClient
    from api import app

    app_module = __import__("api")
    app_module.SESSION_DB = isolate_ticket_db / "sessions_test.sqlite"
    app_module.MEMORY_DB = isolate_ticket_db / "memory_test.sqlite"

    with TestClient(app) as c:
        response = c.post(
            "/auth/login",
            json={"username": "acme-manager", "password": "demo1234"},
        )
        assert response.status_code == 200, response.text
        c.headers["Authorization"] = (
            f"Bearer {response.json()['access_token']}"
        )
        yield c


@pytest.fixture
def login_as(client):
    """Return a helper that issues a token without mutating global client state."""

    def _login(username: str, password: str = "demo1234") -> dict[str, str]:
        response = client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        assert response.status_code == 200, response.text
        return {
            "Authorization": f"Bearer {response.json()['access_token']}"
        }

    return _login


@pytest.fixture
def session_id(client):
    """建一个临时会话，测试结束后删掉，不污染真实会话列表。"""
    sid = client.post("/sessions", json={}).json()["id"]
    yield sid
    client.delete(f"/sessions/{sid}")


# ---------------------------------------------------------------- 假 Agent


class FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeTask:
    def __init__(self, interrupts=None, name="HumanInTheLoopMiddleware.after_model"):
        self.name = name
        self.interrupts = interrupts or []


class FakeSnapshot:
    def __init__(self, tasks=None, values=None):
        self.tasks = tasks or []
        self.values = values or {}
        self.next = ("HumanInTheLoopMiddleware.after_model",) if self.tasks else ()


class FakeAgent:
    """可编排的假 Agent。

    script: 一次 stream 要产出的事件序列，元素是 (chunk, metadata)
    by_thread: thread_id -> 历史消息，用于测试会话隔离
    pending: thread_id -> 待审批动作列表，模拟"图停在中断点"
    """

    def __init__(self):
        self.script = []
        self.by_thread = {}
        self.pending = {}
        self.calls = []

    def stream(self, payload, config=None, stream_mode=None):
        self.calls.append(payload)
        for chunk, metadata in self.script:
            yield chunk, metadata

    def get_state(self, config):
        tid = config["configurable"]["thread_id"]
        messages = self.by_thread.get(tid, [])
        actions = self.pending.get(tid)
        tasks = [FakeTask([FakeInterrupt({"action_requests": actions})])] if actions else []
        return FakeSnapshot(tasks=tasks, values={"messages": messages})


@pytest.fixture
def fake_agent():
    """把 app.state.agent 换成 FakeAgent，测试结束后还原。"""
    from api import app

    original = app.state.agent
    fake = FakeAgent()
    app.state.agent = fake
    try:
        yield fake
    finally:
        app.state.agent = original


# ---------------------------------------------------------------- 构造消息的便捷函数


def token_chunks(text: str, node: str = "model"):
    """把一段文本拆成若干 token 块，模拟流式输出。"""
    from langchain_core.messages import AIMessageChunk

    out = []
    for ch in text:
        out.append((AIMessageChunk(content=ch), {"langgraph_node": node}))
    return out


def tool_message(name: str, text: str):
    from langchain_core.messages import ToolMessage

    return (ToolMessage(content=text, name=name, tool_call_id="call_1"),
            {"langgraph_node": "tools"})


def history(*texts):
    """构造一段历史消息（交替 human/ai）。"""
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = []
    for i, text in enumerate(texts):
        msgs.append(HumanMessage(content=text) if i % 2 == 0 else AIMessage(content=text))
    return msgs


# ---------------------------------------------------------------- SSE 解析


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本解析成 [(event, data), ...]。"""
    import json

    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = "message"
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            try:
                events.append((name, json.loads("\n".join(data_lines))))
            except json.JSONDecodeError:
                pass
    return events


@pytest.fixture
def sse():
    return parse_sse
