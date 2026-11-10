# MCP 工具服务

工具不应该只属于某一个 agent。这个项目把售后能力抽成一层 MCP server：
任何支持 MCP 的客户端都能调用，但**拿不到租户之外的数据，也绕不过审批**。

实现见 `mcp_server.py`（协议层）与 `mcp_tools.py`（业务实现，不依赖 SDK）。

## 鉴权边界

MCP 协议本身不携带用户身份，所以：

| 传输 | 身份来源 | 说明 |
|---|---|---|
| `streamable-http` | `Authorization: Bearer <token>` 请求头 | 服务端解析 token 得到 tenant/user |
| `stdio` | 环境变量 `SHOP_MCP_TOKEN` | 仅本地调试；stdio 没有 HTTP 头，生产不要对外暴露 |

**客户端不能自报租户。** 如果允许 `{"tenant": "acme"}` 这种参数，
任何能连上 server 的进程都能读别人订单——这正是本项目其他部分一直在守的边界。

token 无效、过期、格式错误统一返回"访问令牌无效或已过期"，
不向客户端区分具体原因。

## 工具清单

| 工具 | 类型 | 说明 |
|---|---|---|
| `get_order_info` | 只读 | 订单、金额、退款资格；跨租户返回 `not_found` |
| `search_shop_info` | 只读 | 混合检索 + 证据核查；证据不足返回 `no_evidence` |
| `get_refund_task` | 只读 | 当前退款任务、缺失字段、下一步建议 |
| `recall_after_sales_history` | 只读 | 最近三条已结束售后记录，带工单来源 |
| `prepare_refund` | **写** | 登记任务并创建 `pending_approval` 工单，**不执行退款** |

失败一律返回结构化信封（`code` / `retryable` / `hint`），
和主服务的工具契约一致：`not_found`、`no_evidence` 不重试，`transient` 可重试一次。

## 启动与连接

```bash
# HTTP 模式（推荐）
python mcp_server.py --transport streamable-http --host 127.0.0.1 --port 8765

# stdio 模式（本地调试）
SHOP_MCP_TOKEN="<token>" python mcp_server.py
```

客户端配置（以 HTTP 为例）：

```json
{
  "mcpServers": {
    "shop-after-sales": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

token 通过 `POST /login` 获取，和 Web 端同一套身份系统。

## 写操作为什么安全

`prepare_refund` 只做两件事：把订单和原因写入任务表，然后创建 `pending_approval` 工单。
它不调用退款执行，也不改工单状态。审批仍走四眼原则，执行仍以工单 ID 为幂等键。

所以 MCP 是**多一个入口**，不是多一条后门。

## 已验证

- 单元测试 `tests/test_mcp_tools.py`：无 token 拒绝、token 决定租户、
  跨租户查单返回 `not_found`、写操作只创建待审批且执行次数为 0、线程名稳定。
- 端到端：启动 streamable-http 服务，用官方客户端带 token 调用 5 个工具，
  `A100` 返回订单与退款资格，`A999` 返回 `not_found`。

## 已知限制

- 没有接 OAuth 授权服务器，token 由本项目签发；接入第三方客户端时需要补授权流程。
- stdio 模式靠环境变量注入身份，只适合单人本地调试。
- 没有做调用频率限制；对外开放前需要在传输层加。
- `search_shop_info` 依赖完整检索栈（chromadb + 模型），
  MCP 服务启动时需要能加载索引。
