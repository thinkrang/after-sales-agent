"""Typed, domain-constrained plans proposed by the chat model.

The model selects tasks, arguments and dependencies. Authority, approval and
execution nodes are supplied by the server; no arbitrary Python/tool names.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PLAN_INSTRUCTIONS = (
    "用户同时处理两笔退款时，先 get_refund_plan，再用 create_refund_plan 提议结构化计划。"
    "规划包含至多两个 tasks，每笔 steps 是 lookup_order -> check_policy -> request_refund，"
    "不同任务不互相等待；无需在计划前逐笔调用旧订单工具，调度器会校验归属与资格。"
    "明确申请 intent=apply，仅咨询 intent=consult，缺失订单或原因留空并追问。"
    "本会话一旦有计划，所有退款修改/补充/取消必须用 edit_refund_plan_task 指定 task_id 和"
    "tasks[].state.revision，不再调用单任务 update_refund_task 或 submit_refund_request。"
    "含糊的更正先追问。审批与执行由服务端控制，模型不能批准。等待审批不阻塞其他申请。"
    "已完成结果不得宣称撤回，计划里的等待、失败和完成必须如实分别说明。"
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskDraft(Contract):
    key: str = Field(min_length=1, max_length=32, pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    order_id: str = Field(default="", max_length=64)
    reason: str = Field(default="", max_length=500)
    intent: Literal["consult", "apply"] = "consult"


class StepDraft(Contract):
    key: str = Field(min_length=1, max_length=32, pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    task: str = Field(max_length=32)
    action: Literal["lookup_order", "check_policy", "request_refund"]
    depends_on: list[str] = Field(default_factory=list, max_length=6)


class PlanDraft(Contract):
    goal: str = Field(min_length=1, max_length=500)
    tasks: list[TaskDraft] = Field(min_length=1, max_length=2)
    steps: list[StepDraft] = Field(min_length=3, max_length=6)

    @model_validator(mode="after")
    def validate_graph(self):
        tasks = {t.key: t for t in self.tasks}
        steps = {s.key: s for s in self.steps}
        if len(tasks) != len(self.tasks) or len(steps) != len(self.steps):
            raise ValueError("任务和步骤标识必须唯一")
        orders = [t.order_id.strip() for t in self.tasks if t.order_id.strip()]
        if len(orders) != len(set(orders)):
            raise ValueError("同一计划不能重复申请同一订单")
        visiting, visited = set(), set()

        def visit(key):
            if key not in steps:
                raise ValueError("依赖引用了不存在的步骤")
            if key in visiting:
                raise ValueError("步骤依赖不能有环")
            if key in visited:
                return
            visiting.add(key)
            for dep in steps[key].depends_on:
                visit(dep)
            visiting.remove(key)
            visited.add(key)

        for step in self.steps:
            visit(step.key)
            if step.task not in tasks:
                raise ValueError("步骤必须属于计划中的任务")
            for dep in step.depends_on:
                other = steps[dep]
                if other.task != step.task and (
                    step.action == "request_refund" or other.action == "request_refund"
                ):
                    raise ValueError("审批申请不得与其他任务形成阻塞依赖")
        for task in tasks:
            branch = [s for s in self.steps if s.task == task]
            actions = {s.action: s for s in branch}
            if len(branch) != 3 or len(actions) != 3:
                raise ValueError("每笔任务必须且只能包含查订单、查资格、申请退款三个步骤")
            if actions["lookup_order"].key not in actions["check_policy"].depends_on:
                raise ValueError("资格检查必须依赖本任务的订单查询")
            if actions["check_policy"].key not in actions["request_refund"].depends_on:
                raise ValueError("退款申请必须依赖本任务的资格检查")
        return self


def refund_plan(goal: str, tasks: list[dict]) -> PlanDraft:
    """Deterministic baseline/fixtures; the chat path supplies PlanDraft itself."""
    steps = []
    for task in tasks:
        key = task["key"]
        for action, suffix, deps in (
            ("lookup_order", "order", []),
            ("check_policy", "policy", [key + "_order"]),
            ("request_refund", "request", [key + "_policy"]),
        ):
            steps.append(dict(key=key + "_" + suffix, task=key, action=action, depends_on=deps))
    return PlanDraft(goal=goal, tasks=tasks, steps=steps)
