"""守护「只有一条检索路径」这个架构约束。

背景
----
2026-10-25 发现过一个隐蔽问题：评测脚本走 retrieval.STRATEGIES（无租户过滤），
线上 agent 走 rag.debug_retrieval（有租户过滤）。两条路径的策略、top_k、
过滤条件都不一样，结果是**按评测调出来的权重对线上毫无影响**——
指标看着变好了，用户那边一点没变。

这种问题的可怕之处在于它不会报错。测试全绿、指标上涨，
但测的不是用户走的那条路。

所以这里用静态检查把约束钉死，而不是靠"记得别再分叉"。

为什么用 AST 而不是 import
--------------------------
retrieval / rag 会连带加载 chromadb、sentence-transformers、jieba，
CI 只装轻量依赖，import 会直接失败。用 AST 解析源码，零依赖、秒级完成。
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 每个策略都必须接受 tenant —— 它是安全边界，不能有"忘了传"的口子
STRATEGY_FUNCS = [
    "vector_retrieval",
    "bm25_retrieval",
    "hybrid_retrieval",
    "hybrid_rerank_retrieval",
    "retrieve",
]


def _funcs(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _arg_names(node: ast.FunctionDef) -> set[str]:
    args = node.args
    names = {a.arg for a in args.args + args.kwonlyargs}
    return names


def test_every_strategy_accepts_tenant():
    """所有检索策略都要能接收 tenant，否则线上调用会漏掉租户过滤。"""
    funcs = _funcs(ROOT / "retrieval.py")
    missing = [name for name in STRATEGY_FUNCS if name not in funcs]
    assert not missing, f"retrieval.py 缺少函数：{missing}"

    no_tenant = [
        name for name in STRATEGY_FUNCS if "tenant" not in _arg_names(funcs[name])
    ]
    assert not no_tenant, (
        f"这些策略没有 tenant 参数：{no_tenant}。"
        "tenant 是安全边界，缺了就会检索到别人租户的文档。"
    )


def test_agent_search_uses_unified_retrieval():
    """线上检索工具必须走 retrieval.retrieve，不能直接调 rag.debug_retrieval。

    debug_retrieval 是底层原语（只管 Chroma + 租户过滤），不做策略选择。
    直接调它等于绕过了 retrieval.py 里的权重和策略配置——
    这正是之前"调权重没用"的根因。
    """
    src = (ROOT / "api.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "search_shop_info":
            target = node
            break
    assert target is not None, "api.py 里找不到 search_shop_info"

    body_src = ast.get_source_segment(src, target) or ""

    assert "retrieval.retrieve(" in body_src, (
        "search_shop_info 没有调用 retrieval.retrieve —— "
        "线上检索又和评测分叉了，此时按评测调权重不会影响用户看到的答案。"
    )
    assert "debug_retrieval(" not in body_src, (
        "search_shop_info 直接调了 rag.debug_retrieval，绕过了策略层。"
        "它会跳过 RRF 权重和 DEFAULT_STRATEGY，让评测结论对不上线上行为。"
    )


def test_default_strategy_is_registered():
    """DEFAULT_STRATEGY 必须是 STRATEGIES 里存在的键。

    用 AST 读常量，避免 import 重依赖。
    这条防的是：改了环境变量名或默认值，但忘了同步策略表，
    线上第一次检索就 500。
    """
    src = (ROOT / "retrieval.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    default_value = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "DEFAULT_STRATEGY":
                    # os.getenv("RETRIEVAL_STRATEGY", "hybrid").strip() or "hybrid"
                    for sub in ast.walk(node.value):
                        if (
                            isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "getenv"
                            and len(sub.args) >= 2
                            and isinstance(sub.args[1], ast.Constant)
                        ):
                            default_value = sub.args[1].value
    assert default_value is not None, "解析不出 DEFAULT_STRATEGY 的默认值"

    keys = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "STRATEGIES":
                    keys = {
                        k.value
                        for k in node.value.keys
                        if isinstance(k, ast.Constant)
                    }
    assert keys, "解析不出 STRATEGIES 的键"
    assert default_value in keys, (
        f"DEFAULT_STRATEGY={default_value!r} 不在 STRATEGIES {sorted(keys)} 里"
    )
