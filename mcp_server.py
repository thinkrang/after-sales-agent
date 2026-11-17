"""把售后能力暴露为 MCP server（官方 Python SDK 2.x，协议 2026-07-28）。

启动
----
    python mcp_server.py                                  # stdio，本地调试
    python mcp_server.py --transport streamable-http --port 8765

鉴权
----
HTTP 传输：客户端必须带 ``Authorization: Bearer <token>``，服务端从 token
解析 tenant/user，不接受客户端自报归属。

stdio 传输没有 HTTP header，本地调试用环境变量 ``SHOP_MCP_TOKEN`` 注入身份；
生产不要用 stdio 对外暴露。

写操作
------
``prepare_refund`` 只创建待审批工单。审批与执行仍在原服务里，MCP 不能绕过四眼原则。
"""

from __future__ import annotations

import argparse
import os

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

import mcp_tools

server = MCPServer(
    name="shop-after-sales",
    version="1.0.0",
    instructions=(
        "售后工具服务。所有工具都以调用者身份执行，租户由 Bearer token 决定。"
        "prepare_refund 只创建待审批工单，不会执行退款；审批结果以工单状态为准。"
    ),
)


def _user(ctx: Context):
    """stdio 用环境变量兜底，HTTP 必须走请求头。"""
    if ctx.headers is None:
        token = os.getenv("SHOP_MCP_TOKEN", "")
        return mcp_tools.user_from_headers(
            {"authorization": f"Bearer {token}"} if token else {}
        )
    return mcp_tools.user_from_headers(ctx.headers)


@server.tool(description="查询当前用户自己的订单与退款资格（只读）")
def get_order_info(order_id: str, ctx: Context) -> str:
    return mcp_tools.get_order_info(_user(ctx), order_id)


@server.tool(description="检索店铺知识库并核查证据；证据不足时明确返回 no_evidence")
def search_shop_info(question: str, ctx: Context) -> str:
    return mcp_tools.search_shop_info(_user(ctx), question)


@server.tool(description="读取当前用户的退款任务状态、缺失字段与下一步建议")
def get_refund_task(ctx: Context, task_id: str | None = None) -> str:
    return mcp_tools.get_refund_task(_user(ctx), task_id)


@server.tool(description="查询当前用户最近三条已结束的售后记录（只读，带工单来源）")
def recall_after_sales_history(ctx: Context) -> str:
    return mcp_tools.recall_history(_user(ctx))


@server.tool(description="登记退款并创建待审批工单；不会执行退款")
def prepare_refund(order_id: str, reason: str, ctx: Context,
                   task_id: str | None = None, expected_revision: int | None = None) -> str:
    return mcp_tools.prepare_refund(_user(ctx), order_id, reason, task_id, expected_revision)


def main() -> int:
    parser = argparse.ArgumentParser(description="售后 MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"),
                        default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    import auth
    import orders
    import tickets
    auth.init_auth_db(seed_demo_users=os.getenv("AUTH_SEED_DEMO_USERS", "1") != "0")
    tickets.init_tickets_table()
    orders.init_order_db()

    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
