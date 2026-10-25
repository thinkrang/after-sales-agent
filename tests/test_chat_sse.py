"""聊天 SSE 协议与审批流程测试。

用 FakeAgent 替换真实 Agent，所以不调大模型、不花钱、结果确定。
测的是"协议对不对"：事件顺序、事件内容、异常分支。
"""

from conftest import token_chunks, tool_message
from auth import User
import refund_service


def _prepare_pending(sid: str, order_id: str = "A100", reason: str = "质量问题"):
    user = User(
        "acme-manager",
        "ACME 售后主管",
        "acme",
        frozenset({"customer", "approver"}),
    )
    ticket, _ = refund_service.prepare_ticket(
        user,
        thread_id=sid,
        order_id=order_id,
        reason=reason,
    )
    return ticket


def _post_stream(client, sid, message="测试问题"):
    return client.post("/chat/stream", json={"thread_id": sid, "message": message})


def test_stream_emits_status_tokens_done(client, fake_agent, session_id, sse):
    fake_agent.script = token_chunks("你好")

    events = sse(_post_stream(client, session_id).text)
    kinds = [k for k, _ in events]

    assert kinds[0] == "status"
    assert kinds[-1] == "done"
    assert kinds.count("token") == 2  # "你" "好"

    answer = "".join(d.get("text", "") for k, d in events if k == "token")
    assert answer == "你好"


def test_stream_emits_tool_event(client, fake_agent, session_id, sse):
    fake_agent.script = [tool_message("search_shop_info", "检索到的内容")] + token_chunks("答")

    events = sse(_post_stream(client, session_id).text)
    tools = [d for k, d in events if k == "tool"]

    assert len(tools) == 1
    assert tools[0]["name"] == "search_shop_info"
    assert "检索到的内容" in tools[0]["text"]


def test_done_carries_thread_id_and_title(client, fake_agent, session_id, sse):
    fake_agent.script = token_chunks("好")

    events = sse(_post_stream(client, session_id, "这是标题来源").text)
    done = [d for k, d in events if k == "done"][0]

    assert done["thread_id"] == session_id
    # 会话标题应当被首条消息覆盖
    assert done["title"] == "这是标题来源"


def test_stream_emits_interrupt_instead_of_done(client, fake_agent, session_id, sse):
    """有待审批动作时，必须以 interrupt 结尾，而不是 done。

    这是前端判断"要不要弹卡片"的唯一依据。发错了卡片就不会出现，
    用户会以为程序卡住。
    """
    fake_agent.script = token_chunks("准备提交")
    ticket = _prepare_pending(session_id)
    fake_agent.pending[session_id] = [
        {
            "name": "create_refund_request",
            "args": {"order_id": "A100", "reason": "质量问题"},
            "description": "需要审批",
        }
    ]

    events = sse(_post_stream(client, session_id).text)
    kinds = [k for k, _ in events]

    assert "interrupt" in kinds
    assert "done" not in kinds

    payload = [d for k, d in events if k == "interrupt"][0]
    assert payload["thread_id"] == session_id
    assert payload["actions"][0]["name"] == "create_refund_request"
    assert payload["actions"][0]["approval_id"] == ticket["id"]
    assert payload["actions"][0]["args"]["order_id"] == "A100"


def test_pending_survives_page_reload(client, fake_agent, session_id):
    """刷新页面后必须还能拿到待审批动作，否则会话卡死。"""
    fake_agent.pending[session_id] = [
        {
            "name": "create_refund_request",
            "args": {"order_id": "A100", "reason": "质量问题"},
            "description": "d",
        }
    ]
    _prepare_pending(session_id)

    body = client.get(f"/sessions/{session_id}/messages").json()
    assert len(body["pending"]) == 1
    assert body["pending"][0]["args"]["order_id"] == "A100"


def test_resume_without_pending_returns_409(client, fake_agent, session_id):
    """防重复提交：没有待审批就必须 409，否则写操作会被执行两次。"""
    fake_agent.pending.pop(session_id, None)

    r = client.post(
        "/chat/resume",
        json={
            "thread_id": session_id,
            "approval_id": "not-a-pending-record",
            "decision": "approve",
        },
    )
    assert r.status_code == 409


def test_resume_approve_streams_answer(client, fake_agent, session_id, sse):
    fake_agent.pending[session_id] = [
        {
            "name": "create_refund_request",
            "args": {"order_id": "A100", "reason": "质量问题"},
            "description": "d",
        }
    ]
    ticket = _prepare_pending(session_id)
    fake_agent.script = token_chunks("已提交")

    r = client.post(
        "/chat/resume",
        json={
            "thread_id": session_id,
            "approval_id": ticket["id"],
            "decision": "approve",
        },
    )
    assert r.status_code == 200

    events = sse(r.text)
    answer = "".join(d.get("text", "") for k, d in events if k == "token")
    assert answer == "已提交"


def test_resume_reject_is_accepted(client, fake_agent, session_id, sse):
    fake_agent.pending[session_id] = [
        {
            "name": "create_refund_request",
            "args": {"order_id": "A100", "reason": "质量问题"},
            "description": "d",
        }
    ]
    ticket = _prepare_pending(session_id)
    fake_agent.script = token_chunks("好的")

    r = client.post("/chat/resume", json={
        "thread_id": session_id,
        "approval_id": ticket["id"],
        "decision": "reject",
        "message": "信息不全",
    })
    assert r.status_code == 200

    # 注意：不能直接断言 "好的" in r.text —— 流式输出把字拆成了多个事件，
    # 原始 SSE 文本里不存在连续的 "好的"。必须解析事件再拼回去。
    events = sse(r.text)
    answer = "".join(d.get("text", "") for k, d in events if k == "token")
    assert answer == "好的"


def test_resume_rejects_invalid_decision(client, session_id):
    r = client.post(
        "/chat/resume",
        json={
            "thread_id": session_id,
            "approval_id": "not-a-pending-record",
            "decision": "maybe",
        },
    )
    assert r.status_code == 422


def test_chat_rejects_illegal_thread_id(client):
    """thread_id 会拼进 SQL 和文件路径，必须白名单校验。"""
    r = client.post("/chat/stream", json={"thread_id": "../../etc/passwd", "message": "x"})
    assert r.status_code == 422


def test_chat_rejects_empty_message(client, session_id):
    r = client.post("/chat/stream", json={"thread_id": session_id, "message": ""})
    assert r.status_code == 422


def test_chat_rejects_overlong_message(client, session_id):
    r = client.post("/chat/stream", json={"thread_id": session_id, "message": "x" * 2001})
    assert r.status_code == 422
