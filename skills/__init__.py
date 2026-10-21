"""Skill 包：渐进加载的能力单元。

对外只暴露四个函数，工具层不需要知道目录结构：
    discover()  列出所有技能元信息
    get(name)   按名取一个技能
    catalog()   生成给模型看的清单文本
    stats()     统计目录与全量正文的体积差
"""

from skills.registry import (
    SkillMeta,
    catalog,
    clear_cache,
    discover,
    get,
    stats,
)

__all__ = [
    "SkillMeta",
    "catalog",
    "clear_cache",
    "discover",
    "get",
    "stats",
]
