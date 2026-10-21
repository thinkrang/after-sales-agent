"""Skill 机制测试。

覆盖三件事：
    1. 目录扫描与 frontmatter 解析的正确性（含各种脏数据）
    2. 渐进加载真的省了体积（否则这套机制就是花架子）
    3. 两个工具的行为——尤其是加载不存在的 skill 时不能崩
"""

import pytest

import skills
from skills.registry import _parse_frontmatter


# ---------------------------------------------------------------- 真实目录


def test_discovers_three_skills():
    items = skills.discover()
    names = {s.name for s in items}
    assert names == {"refund_flow", "knowledge_query", "complaint_escalation"}


def test_every_skill_has_name_and_description():
    for item in skills.discover():
        assert item.name
        assert len(item.description) > 10, f"{item.name} 的 description 太短，模型判断不了"


def test_catalog_lists_all_names():
    text = skills.catalog()
    for item in skills.discover():
        assert item.name in text
        assert item.description in text


def test_catalog_is_much_smaller_than_full_text():
    """渐进加载的价值就在这里。如果比值接近 1，说明 skill 太小，机制没意义。"""
    s = skills.stats()
    assert s["skill_count"] >= 3
    assert s["ratio"] > 5, f"目录/正文 只有 {s['ratio']}x，省得不够"
    assert s["saved_chars"] > 1000


# ---------------------------------------------------------------- 查找


def test_get_by_name():
    skill = skills.get("refund_flow")
    assert skill is not None
    assert "退款流程" in skill.content


def test_get_is_case_insensitive():
    assert skills.get("REFUND_FLOW") is not None
    assert skills.get("  Refund_Flow  ") is not None


def test_get_unknown_returns_none():
    assert skills.get("does_not_exist") is None
    assert skills.get("") is None


# ---------------------------------------------------------------- 内容


def test_content_excludes_frontmatter():
    """给模型的是正文。frontmatter 再给一遍就是浪费 token。"""
    for item in skills.discover():
        assert not item.content.startswith("---")
        assert "description:" not in item.content[:80]


def test_content_is_substantial():
    for item in skills.discover():
        assert len(item.content) > 400, f"{item.name} 正文太短，撑不起一个流程"


def test_raw_keeps_frontmatter():
    for item in skills.discover():
        assert item.raw.startswith("---")
        assert item.raw != item.content


# ---------------------------------------------------------------- 解析器


def test_parse_without_frontmatter():
    meta, body = _parse_frontmatter("# 标题\n正文内容")
    assert meta == {}
    assert body == "# 标题\n正文内容"


def test_parse_strips_quotes():
    meta, body = _parse_frontmatter('---\nname: "quoted"\ndesc: \'single\'\n---\nbody')
    assert meta["name"] == "quoted"
    assert meta["desc"] == "single"
    assert body == "body"


def test_parse_ignores_comments_and_blank_lines():
    meta, _ = _parse_frontmatter("---\n# 注释\n\nname: x\n---\nbody")
    assert meta == {"name": "x"}


def test_parse_ignores_lines_without_colon():
    meta, _ = _parse_frontmatter("---\nname: x\n这行没有冒号\n---\nbody")
    assert meta == {"name": "x"}


# ---------------------------------------------------------------- 扫描边界


def _write(root, dirname, text):
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


def test_skips_skill_without_description(tmp_path):
    """没有 description 的 skill 等于不可被发现，应该直接跳过。"""
    _write(tmp_path, "no_desc", "---\nname: no_desc\n---\n正文")
    _write(tmp_path, "good", "---\nname: good\ndescription: 有描述的技能\n---\n正文")

    items = skills.registry.discover(tmp_path)
    assert [s.name for s in items] == ["good"]


def test_skips_directory_without_skill_md(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    _write(tmp_path, "real", "---\nname: real\ndescription: 真技能\n---\n正文")

    items = skills.registry.discover(tmp_path)
    assert [s.name for s in items] == ["real"]


def test_skips_underscore_and_hidden_dirs(tmp_path):
    """skills/ 里会放 registry.py 这类模块，不能被当成技能。"""
    _write(tmp_path, "_private", "---\nname: p\ndescription: 不该出现\n---\nx")
    _write(tmp_path, ".hidden", "---\nname: h\ndescription: 不该出现\n---\nx")
    _write(tmp_path, "visible", "---\nname: v\ndescription: 应该出现\n---\nx")

    items = skills.registry.discover(tmp_path)
    assert [s.name for s in items] == ["v"]


def test_falls_back_to_dirname_when_name_missing(tmp_path):
    _write(tmp_path, "fallback_name", "---\ndescription: 没有 name 字段\n---\n正文")
    items = skills.registry.discover(tmp_path)
    assert items[0].name == "fallback_name"


def test_empty_root_returns_empty(tmp_path):
    assert skills.registry.discover(tmp_path) == ()
    assert skills.registry.catalog(tmp_path) == "（当前没有可用技能）"


# ---------------------------------------------------------------- 工具层


def test_list_skills_tool():
    from api import list_skills

    out = list_skills.invoke({})
    assert "refund_flow" in out
    assert "load_skill" in out


def test_load_skill_tool_returns_content():
    from api import load_skill

    out = load_skill.invoke({"name": "refund_flow"})
    assert "退款流程" in out
    assert not out.startswith("---"), "不该把 frontmatter 也给模型"


def test_load_skill_tool_handles_unknown():
    """加载不存在的 skill 不能抛异常——模型会瞎猜名字。"""
    from api import load_skill

    out = load_skill.invoke({"name": "not_a_skill"})
    assert "没有名为" in out
    assert "refund_flow" in out, "应该顺带告诉模型有哪些可用"
