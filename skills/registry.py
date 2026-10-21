"""Skill 注册表：扫描目录、解析 frontmatter、按需加载正文。

设计要点
--------
**渐进加载（progressive disclosure）** 是这个模块存在的理由。

把全部流程知识塞进 system prompt 的问题：
    - 每轮请求都付固定 token 成本，哪怕这轮根本用不到
    - skill 从 3 个涨到 20 个时，prompt 会膨胀到不可控
    - 加一个 skill 要改代码

改成「目录 + 按需加载」之后：
    - system prompt 只留一行提示，说明有技能可用
    - 模型先看目录（name + description），判断相关才加载正文
    - 加 skill = 加一个目录，不动代码

代价是多一次工具调用往返。所以 description 必须写准——
它是模型唯一的判断依据，写模糊了就会漏加载或多加载。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent

# frontmatter 只支持 `key: value` 单行，够用了。
# 不引 pyyaml：只有两个字段，加一个依赖不划算。
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    path: Path

    @property
    def raw(self) -> str:
        """整份文件，含 frontmatter。"""
        return self.path.read_text(encoding="utf-8")

    @property
    def content(self) -> str:
        """只有正文，去掉 frontmatter。

        给模型的是正文——name/description 它已经从目录里看过了，
        重复给一遍纯属浪费 token。
        """
        _, body = _parse_frontmatter(self.raw)
        return body.strip()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """把 `---\\nkey: value\\n---\\n正文` 拆成 (meta, 正文)。"""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")

    return meta, match.group(2)


def _discover(root: Path) -> tuple[SkillMeta, ...]:
    """扫描 root 下每个子目录的 SKILL.md。

    跳过 __pycache__、下划线开头的目录，以及没有 SKILL.md 的目录——
    这样在 skills/ 里放 helper 模块（如本文件）不会被误当成 skill。
    """
    skills: list[SkillMeta] = []

    if not root.is_dir():
        return ()

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue

        skill_file = child / "SKILL.md"
        if not skill_file.is_file():
            continue

        meta, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        name = meta.get("name", "").strip() or child.name
        description = meta.get("description", "").strip()

        if not description:
            # 没有 description 的 skill 等于不可被发现，直接跳过并留痕
            continue

        skills.append(SkillMeta(name=name, description=description, path=skill_file))

    return tuple(skills)


@lru_cache(maxsize=8)
def discover(root: Path | None = None) -> tuple[SkillMeta, ...]:
    return _discover(root or SKILLS_DIR)


def clear_cache() -> None:
    """测试里改了目录后调用，避免 lru_cache 返回旧结果。"""
    discover.cache_clear()


def get(name: str, root: Path | None = None) -> SkillMeta | None:
    target = name.strip().lower()
    for skill in discover(root):
        if skill.name.lower() == target:
            return skill
    return None


def catalog(root: Path | None = None) -> str:
    """给模型看的技能清单。只给 name + description，不给正文。"""
    skills = discover(root)
    if not skills:
        return "（当前没有可用技能）"

    lines = []
    for skill in skills:
        lines.append(f"- {skill.name}：{skill.description}")
    return "\n".join(lines)


def stats(root: Path | None = None) -> dict:
    """量化渐进加载省了多少字——用来验证这个机制真的有意义。"""
    skills = discover(root)
    catalog_chars = len(catalog(root))
    full_chars = sum(len(s.content) for s in skills)
    return {
        "skill_count": len(skills),
        "catalog_chars": catalog_chars,
        "full_chars": full_chars,
        "saved_chars": full_chars - catalog_chars,
        "ratio": round(full_chars / catalog_chars, 1) if catalog_chars else 0,
    }
