# 第一期：修复缺口与验证证据

日期：2026-11-10。基于 `b4d9d5a` 加本轮工作区改动；未自动提交或推送。

## 修复了什么

| 缺口 | 修复与验证边界 |
|---|---|
| 高风险申请发起人点击拒绝被四眼规则拦截 | 区分批准与撤回；本人可拒绝仍待审批的申请，高风险本人批准继续拒绝。覆盖普通客户、兼有审批角色的客户、错误会话、同租户他人和跨租户审批人。旧审批卡片 HTTP 链路也有回归。 |
| 待审批队列先 LIMIT 后过滤 | 状态、租户、所需角色、排除本人均在 SQL LIMIT 之前应用。`actionable_only=true` 返回本租户当前人可批的单；完整审计列表保留。 |
| 正文反查向量 ID | 使用 Chroma 稳定 chunk ID；融合前再核验可见租户。相同正文跨租户不会改写来源。 |
| BM25 继续使用旧缓存 | 同时失效语料和打分器；索引写入后原子发布共享版本，API 与 Celery 进程可发现版本变化。依赖共享索引目录，未解决多机 Chroma 同步和并发写一致性。 |
| reranker 候选数超过底层上限 | 内部最多允许 100 个候选，公开 retrieve/HTTP 的 top_k 仍限制到 10；用可控打分器覆盖 top-10 召回 30 个候选。空候选不再调用模型。 |
| 调参脚本只看 R@5、遗漏 0.4:0.6 | dev 扫描完整候选，以实际线上 top-3 的 R@3 优先、MRR@10 次优选择。保存逐题结果、数据与索引指纹；固定权重再评指定留出题。 |
| 用 top-10 的前三冒充线上 rerank top-3 | 评测独立调用 top-3 和 top-10；R@3/Hit@1 来自真实 top-3 请求，R@5/MRR 为另一次 top-10 诊断。 |
| 模型评测子进程崩溃仍可能全绿 | 缺输出、缺场景、重复 ID、错误类型、非零退出和超时均记失败；聊天 error 不再计作抵抗注入成功。 |
| DeepSeek 不支持默认 JSON schema 输出 | 真实接口发现 400；证据核查显式使用 function_calling + Pydantic。单独验证有证据放行、无证据拒答，避免用核查器故障冒充正确拒答。 |
| 重复测试、文档成绩冲突、Windows 输出失败 | 删除同名测试覆盖，纠正 11 条留出题、区分历史成绩；动作报告使用 GBK 可显示的 PASS/FAIL/DANGER，SQLite 连接先关闭再删除临时目录。 |
| 验收会覆盖固定数据目录 | 新建独立临时目录；索引和上传目录支持环境变量隔离。保留原始日志和报告，不覆盖业务数据库。 |
| Docker 构建包含本地虚拟环境、排除 Skill 正文 | 排除 work/outputs 与私密配置，显式保留运行时 Skill Markdown。 |

## 复现

```bash
python scripts/verify_project.py --output work/verification
python scripts/verify_project.py --with-model --output work/verification-model
```

默认不调用远程模型；`--with-model` 另跑证据核查真实接口、8 条模型行为和 HTTP 验收，产生 API 费用。
每次新建 `run-*`，保存 Python/依赖、命令、逐步耗时、源码指纹、逐文件清单及 JSON/XML。
manifest 包括根目录、scripts、tests 和 Skill 源码；执行前后源码变化会使总验收返回失败。

本轮 Docker 命令（PowerShell，项目根目录执行；需已有离线 embedding 缓存）：

```powershell
docker build -t after-sales-verification:priority-one-20261110 .
$verifyRoot = (New-Item -ItemType Directory ./work/docker-verify -Force).FullName
$verifyCache = Join-Path $env:USERPROFILE '.cache/huggingface/hub'
docker run --rm --env-file .env --mount "type=bind,source=$verifyCache,target=/root/.cache/huggingface/hub,readonly" --mount "type=bind,source=$verifyRoot,target=/verification" after-sales-verification:priority-one-20261110 python scripts/verify_project.py --output /verification
$verifyRun = Get-ChildItem -LiteralPath $verifyRoot -Directory -Filter 'run-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$verifyIndex = '/verification/' + $verifyRun.Name + '/vector_db'
docker run --rm --env-file .env --mount "type=bind,source=$verifyCache,target=/root/.cache/huggingface/hub,readonly" --mount "type=bind,source=$verifyRoot,target=/verification" -e "AGENT_VECTOR_DB_DIR=$verifyIndex" after-sales-verification:priority-one-20261110 python scripts/acceptance_e2e.py --output /verification/acceptance.json
```

本机使用已安装完整依赖的 `work/test-env/Scripts/python.exe`。pip check 通过不等同于重新安装全部依赖；干净安装以 Docker 构建结果为准。

## 结果口径

原始 JSON 与环境说明见 [汇总](evaluation/verification-2026-11-10.json)，逐项报告在
`docs/evaluation/verification-2026-11-10/`。完整日志、XML 与依赖列表另保存在本地交付包。

| 验证层次 | Windows / Python 3.13 | Docker / Python 3.12 |
|---|---|---|
| 从空目录重建索引 | 152 chunks | 152 chunks |
| 完整 pytest（不调用真实模型） | 283 passed / 1 skipped / 2 deselected | 283 passed / 1 skipped / 2 deselected |
| 确定性动作安全场景 | 36/36，危险误执行 0 | 36/36，危险误执行 0 |
| 业务任务回归（8 场景各 3 次） | 24/24 | 24/24 |
| 真实 DeepSeek 行为场景（单次） | 8/8 | 未重复跑这组；另跑 HTTP 业务闭环 |
| 真实证据核查接口 | 有证据/无证据 2/2 | HTTP 中实际调用核查器，无接口 400 |
| S1–S14 HTTP/ASGI 检查点 | 32 PASS / 0 FAIL / 1 BLOCKED（Docker 步骤） | 33 PASS / 0 FAIL / 0 BLOCKED |
| dev R@3：纯向量 → 混合 | 61/67 → 65/67 | 61/67 → 65/67 |
| 指定留出题 R@3：纯向量 → 混合 | 11/11 → 10/11 | 11/11 → 10/11 |

Docker 镜像从 requirements 安装完整依赖，使用只读的已下载 embedding 缓存与全新运行数据。
镜像 `after-sales-verification:priority-one-20261110` 的源码 manifest 与宿主机一致；
已检查不包含 `.env` 和本地 work/，且包含 Skill 正文。镜像 ID 见汇总。
容器未带 `.git`，不能靠其内部 `commit=null` 判断版本；外部 Docker 报告记录宿主机 commit、
源码指纹、镜像 ID 和 manifest 校对结果。内部脚本未挂 Docker socket，daemon 检测值不代表未在容器中运行。

HTTP 验收使用真实 FastAPI lifespan、LangGraph、SQLite 和 DeepSeek；通过 TestClient 执行，
没有验证浏览器交互、外部 uvicorn 网络链路或完整 Redis/Celery 多容器拓扑。
本地完整回归后仅修正了验收脚本的 Docker 检测提示，该最终脚本已在镜像中运行。

首次运行中，8 条行为场景和旧 HTTP 验收都可能显示通过，但日志出现证据核查接口 400。
该次结果保留用于诊断，不作为核查器正常工作的证明。修复后必须重跑，且报告有证据/无证据两条真实请求。

152 篇是受控合成知识库。检索 dev 67 条，指定留出题 11 条；留出题已被历史评审检查，不能称作未见测试集。
当前开发集选择 0.6:0.4；指定留出题纯向量 11/11、混合召回 10/11，退步题为 P03。
不能从这份小样本宣称泛化能力提升，也不能把单测、确定性业务回归和模型成功率相加。

真实 CrossEncoder 权重未下载，完整模型精排仍未验收；候选边界只由可控打分器验证。
模拟支付网关、单进程会话锁和本地 SQLite 的既有生产边界仍适用。

## 简历可使用的能力表述

实现服务端租户/角色授权、确定性风险分级与四眼审批，并通过审批载荷绑定、条件状态迁移和下游幂等控制退款副作用；覆盖撤回、重复执行与崩溃恢复等失败路径。

构建区分业务回归、真实模型行为与 HTTP 全链路的验证流程，保存代码/数据指纹与逐场景结果；定位并修复评测假绿和模型接口不兼容，记录检索开发集获益及留出题退步。

具体通过数须与本文件最终结果及归档报告同时引用。不要写“生产零事故”“安全率 100%”或“全面提升 RAG 准确率”。
