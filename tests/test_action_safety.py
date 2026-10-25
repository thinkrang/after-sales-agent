"""把动作级评测接进 pytest。

CI 里只跑确定性场景（不调模型），因为安全性质不该依赖模型这次表现如何。
任何"危险动作被执行"都直接让构建失败——这类问题不能用"通过率"妥协。
"""

import action_eval


def test_deterministic_action_scenarios():
    report = action_eval.run(action_eval.deterministic_scenarios(), verbose=False)

    dangerous = [(s.id, s.description, o.detail) for s, o in report.dangerous]
    assert not dangerous, f"有危险动作被执行：{dangerous}"

    failed = [
        (s.id, s.description, o.detail) for s, o in report.failed
    ]
    assert not failed, f"场景未通过：{failed}"

    ok, total = report.count()
    assert ok == total == len(action_eval.deterministic_scenarios())


def test_every_category_is_covered():
    """防止以后删场景把某一类安全性覆盖掉。"""
    cats = {s.category for s in action_eval.deterministic_scenarios()}
    assert cats == {"身份与租户", "审批绕过", "风险分级", "幂等与恢复", "状态机"}
