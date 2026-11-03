import json
import os
import subprocess
import sys

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import db
import orders
import refund_flow
import refund_service
import refund_tasks
import tickets
from auth import User


@pytest.fixture
def flow_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tickets, "DB_PATH", tmp_path / "tickets.sqlite")
    monkeypatch.setattr(orders, "ORDERS_DB", tmp_path / "orders.sqlite")
    tickets.init_tickets_table()
    orders.init_order_db()
    user = User("acme-manager", "客户", "acme", frozenset({"customer"}))
    refund_tasks.update(user, "conversation", expected_revision=0, order_id="A100", reason="质量问题", intent="apply")
    ticket, _ = refund_service.prepare_ticket(user, thread_id="conversation", order_id="A100", reason="质量问题", task_revision=1, approval_protocol="detached")
    yield user, ticket, tmp_path / "flow.sqlite"
    db.release_all()


def test_real_process_restart_resumes_approved_workflow(flow_db):
    user, ticket, memory_path = flow_db
    with SqliteSaver.from_conn_string(str(memory_path)) as saver:
        graph = refund_flow.build(saver)
        refund_flow.settle(graph, ticket)
        assert graph.get_state(refund_flow.config(ticket["id"])).next
    refund_service.decide(user, approval_id=ticket["id"], thread_id="conversation", decision="approve")
    env = dict(os.environ, AGENT_DATA_DIR=str(memory_path.parent), PYTHONIOENCODING="utf-8")
    code = """
import json, sys
from langgraph.checkpoint.sqlite import SqliteSaver
import refund_flow, tickets
with SqliteSaver.from_conn_string(sys.argv[1]) as saver:
    graph = refund_flow.build(saver)
    ticket = refund_flow.settle(graph, tickets.get_ticket(sys.argv[2]))
    print(json.dumps(dict(status=ticket['status'], refund_id=ticket['refund_id'], pending=bool(graph.get_state(refund_flow.config(ticket['id'])).next))))
"""
    proc = subprocess.run([sys.executable, "-c", code, str(memory_path), ticket["id"]],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["status"] == "executed" and result["refund_id"] and not result["pending"]
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1


def test_lost_result_then_checkpoint_reopen_does_not_repeat_refund(flow_db, monkeypatch):
    user, ticket, memory_path = flow_db
    with SqliteSaver.from_conn_string(str(memory_path)) as saver:
        graph = refund_flow.build(saver)
        refund_flow.settle(graph, ticket)
        refund_service.decide(user, approval_id=ticket["id"], thread_id="conversation", decision="approve")
        original = tickets.mark_executed

        def crash(*args, **kwargs):
            raise OSError("下游成功、工单完成前退出")

        with monkeypatch.context() as patch:
            patch.setattr(tickets, "mark_executed", crash)
            with pytest.raises(OSError):
                refund_flow.settle(graph, tickets.get_ticket(ticket["id"]))
        assert tickets.get_ticket(ticket["id"])["status"] == "executing"
    with SqliteSaver.from_conn_string(str(memory_path)) as saver:
        graph = refund_flow.build(saver)
        final = refund_flow.settle(graph, tickets.get_ticket(ticket["id"]))
        assert final["status"] == "executed"
        assert not graph.get_state(refund_flow.config(ticket["id"])).next
    with db.session(orders.ORDERS_DB) as conn:
        assert conn.execute("SELECT COUNT(*) FROM refund_executions").fetchone()[0] == 1
