# 退款任务、持续对话与评测

## 两条持久化流程

`create_agent` 负责理解用户消息和调用查询/任务工具；`refund_flow.py` 的 StateGraph
负责审批等待与执行。新申请的 `submit_refund_request` 正常返回工单，不把对话图停在
未完成的 tool call 上。审批等待时，用户仍可自然语言改口、取消、补充原因和插话。

执行图使用独立 `refund_memory.sqlite`；对话仍使用 `chat_memory.sqlite`。
执行图恢复只改变控制流，不能授权：节点必须重查服务端工单，未批准即禁止退款。
数据库中已批准但执行图尚未建立的工单，可以重建流程；下游成功后崩溃则复用原幂等键恢复。
审批结果由服务端写回对话，稳定 message id 避免重试重复插入，无需模型复述支付结果。

旧 `create_refund_request` 和 HITL middleware 仅兼容升级前的会话中断，不用于新申请。

## 任务状态

`refund_tasks` 保存租户、发起人、会话、订单、原因、consult/apply 意图、版本、
取消标记和关联工单；`refund_task_events` 保存版本快照、来源与聊天轮次 id。
表由启动迁移无损创建，不重置原数据库。

- `get_refund_task` 返回 revision、missing_fields、状态和 next_action。
- `update_refund_task` 用 expected_revision 防旧请求覆盖新信息。
- 改订单且未明确提供新原因时清空原因；撤回后必须明确 apply 才重新打开。
- 修改和拒绝旧待审批位于同一事务；已批准/执行中不可修改，已执行不可撤回。
- 查询物流/支付等插话不修改原任务；含糊改口先澄清，由真实模型场景检查。
- 原审批失效后，执行图会被关闭；关闭失败只记录错误，不能恢复旧审批授权。
- 页面顶部显示进度与变更时间线，卡片支持明确修改与撤回，聊天也能处理这些意图。

单会话锁保护同一进程内的对话 checkpoint 和审批结果更新；不同会话可并发。
锁有固定大小，不无限增长。多 API worker 的 checkpoint 并发仍需分布式会话锁或路由黏性；
数据库条件迁移和唯一约束继续作为跨进程副作用安全边界。

## 接口

- `GET /sessions/{sid}/refund-task`：任务和下一步建议。
- `PATCH /sessions/{sid}/refund-task`：提交 expected_revision 与新字段，或 cancel=true。
- `GET /sessions/{sid}/refund-task/history?after=0`：归属隔离的审计记录和 next_cursor，每页最多 50 条。

版本冲突/已执行返回 409；不可访问的会话/订单返回 404；未知字段返回 422。
租户与用户来自登录上下文，不能由请求体指定。SSE done 和 interrupt 均带 refund_task。
SSE interrupt 只表示存在待审批卡片，不表示新对话图不可继续。

## 评测层次

### 确定性业务回归

```bash
python task_eval.py --repeat 3 --output work/task-system.json
```

8 个场景检查补槽、咨询、改订单、改原因、撤回、批准后不可改、下游成功但响应丢失、
关闭连接后重读。独立临时数据库，不写业务数据。结果只能表述为业务回归通过率。

`tests/test_detached_tasks.py` 使用脚本模型和真实 LangGraph/HTTP/SQLite，验证待审批
继续对话、改口撤回、旧流程关闭、原始 resume 不能授予权限和流程重建。
`tests/test_refund_flow.py` 另外用真正的 Python 子进程验证重启恢复，用故障注入验证
下游成功后、工单写回前崩溃不重复退款。

### 真实模型多轮任务

```bash
python task_eval.py --mode model --variant structured --split dev --repeat 3 --output work/structured-dev.json
python task_eval.py --mode model --variant structured --split holdout --repeat 3 --output work/structured-holdout.json
```

需要完整 requirements、可用 embedding/知识索引和有效 DEEPSEEK_API_KEY，会产生 API 费用。
全部业务数据库在临时目录，启动前逐一验证路径。使用现有知识索引，不重建业务订单/工单。
默认只跑 dev；`--case ID` 可限制场景；`--max-model-calls 40` 限制单场景模型调用，
主模型与证据核查共同计数。评测模型请求不自动重试、单请求超时 30 秒。

数据集 `refund-dialogue-v2`：8 个开发场景 + 4 个指定 holdout 合成场景。
包括多轮补槽、待审批插话、自然语言改订单/原因、取消、拒绝、高风险主管审批、
插话后恢复、撤回后换新单、含糊改口不擅改、要求绕过审批。
修改/取消用自然语言驱动真实 Agent，不借用 PATCH 接口代替模型理解。

判定依据：每轮审批前无退款副作用；正确工单订单/原因；插话保留原审批；改口使旧审批失效；
高风险自批被拒；最终仅一次、正确金额退款；对话图无悬空；过期执行图关闭；用户收到正确退款单号。
不是以回答命中某个关键词判定完整业务成功。

### 对照实验

```bash
python task_eval.py --mode model --variant messages --split dev --repeat 3 --output work/messages-dev.json
python task_eval.py --mode model --variant structured --split dev --repeat 3 --baseline work/messages-dev.json --output work/paired-dev.json
```

messages 变体保留同一远程模型、对话 checkpoint、订单查询、检索、待审批撤回工具、
审批/风控/幂等服务，但去掉结构化任务工具和 Skill 流程提示，仅根据聊天历史组织任务。
它是组件组合对照，不能把差异全归因于某个字段或某一句提示。

比较要求模式、数据集 SHA256、模型、场景和重复次数一致，且变体不同。
同时报告获益与退步案例，不删除失败运行、不选择性比较子集。

## 报告与复现

JSON 包含数据集版本/指纹、代码 commit/dirty/source_sha256、分组与逐场景重复结果、
失败阶段、P50/P95、聊天轮数、工具参数轨迹、主模型与证据核查调用次数、token 数、工单和执行记录。
错误归因为 intent / planning / state / approval_binding / safety / execution / completion /
conversation / recovery / model / model_budget / protocol / transport / setup / runtime。
模型/网络失败计入分母；`usage_complete=false` 时 token 不完整，不能声称精确成本；不计算货币费用。
耗时不含服务启动预热。没有成功率置信区间或显著性结论，重复次数少时只作开发诊断。

指定 holdout 仍是人工编写的合成场景，非独立用户采集，也不代表生产分布。
修改场景或依据 holdout 结果调参后，应重新划分并报告，不能继续声称未见测试集。
真实模型结果须实际运行后填写，脚本模型/业务单测不代表真实模型成功率。

## 本轮边界

未接入政策版本与执行资格绑定、独立支付服务、未知支付结果人工复核状态。
现有模拟下游与重试上限语义保持原实现，不宣称生产支付保障。

## 本次运行结果

2026-11-03：结构化 dev 16/16，对照 dev 16/16，结构化指定 holdout 4/4。
没有观察到成功率提升，结构化方案开销更高；详细数值与局限见
[评测摘要](evaluation/task-v2-summary.json) 和 README。报告按运行开始时的源文件摘要记录，
完整逐轮记录保留在本地 work 目录，公开摘要不包含原始消息或数据库转储。
