"""多租户权限过滤测试。

核心要验证的不是"能过滤"，而是**"过滤发生在检索前"**：
同一句问题，不同租户拿到的第一条结果必须是各自可见的文档，
而不是先取全量 top-k 再筛掉（那样候选池会被别人的数据占满）。
"""

import pytest

from rag import (
    DEFAULT_TENANT,
    debug_retrieval,
    tenant_filter,
    tenant_from_filename,
    visible_tenants,
)

ACME_DOC = "tenant_acme_price_agreement.txt"
GLOBEX_DOC = "tenant_globex_price_agreement.txt"


# ---------------------------------------------------------------- 纯函数


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("tenant_acme_price.txt", "acme"),
        ("tenant_globex_contract.txt", "globex"),
        ("tenant_ACME_upper.txt", "acme"),
        ("product_info.txt", DEFAULT_TENANT),
        ("refund_policy.txt", DEFAULT_TENANT),
        ("tenant_.txt", DEFAULT_TENANT),
    ],
)
def test_tenant_from_filename(filename, expected):
    assert tenant_from_filename(filename) == expected


def test_visible_tenants_includes_public():
    """自己的租户必须能同时看到公共文档，否则连商品价格都查不到。"""
    assert set(visible_tenants("acme")) == {"acme", DEFAULT_TENANT}
    assert visible_tenants(None) == [DEFAULT_TENANT]
    assert visible_tenants(DEFAULT_TENANT) == [DEFAULT_TENANT]


def test_tenant_filter_shape():
    assert tenant_filter(None) == {"tenant": DEFAULT_TENANT}
    condition = tenant_filter("acme")
    assert condition["tenant"]["$in"] == ["acme", DEFAULT_TENANT]


# ---------------------------------------------------------------- 检索行为


def test_acme_sees_own_doc_not_globex():
    results = debug_retrieval("卫衣的协议价是多少", top_k=5, tenant="acme")

    sources = [r["source"] for r in results]
    assert ACME_DOC in sources
    assert GLOBEX_DOC not in sources
    # 协议价文档应该排第一 —— 说明它真的参与了检索，不是被筛掉的
    assert sources[0] == ACME_DOC


def test_globex_sees_own_doc_not_acme():
    results = debug_retrieval("卫衣的协议价是多少", top_k=5, tenant="globex")

    sources = [r["source"] for r in results]
    assert GLOBEX_DOC in sources
    assert ACME_DOC not in sources
    assert sources[0] == GLOBEX_DOC


def test_public_user_sees_neither_tenant_doc():
    """不带租户时两篇协议价都不可见 —— 这正是迁移必须做的原因：
    没有 tenant 字段的老数据会以同样方式静默消失。"""
    results = debug_retrieval("卫衣的协议价是多少", top_k=5, tenant=None)

    sources = [r["source"] for r in results]
    assert ACME_DOC not in sources
    assert GLOBEX_DOC not in sources
    # 但仍能看到公共商品信息
    assert any("product_info" in s for s in sources)


def test_tenant_does_not_starve_public_docs():
    """检索前过滤不能让结果只剩租户文档。

    租户文档只有 1 篇，k=3 时应该还能召回 2 篇公共文档。
    如果是"检索后过滤"，很可能这 3 个名额全被别的东西占掉了。
    """
    results = debug_retrieval("卫衣多少钱", top_k=3, tenant="acme")
    tenants = [r["tenant"] for r in results]
    assert DEFAULT_TENANT in tenants


def test_retrieval_respects_top_k_with_filter():
    results = debug_retrieval("卫衣", top_k=2, tenant="acme")
    assert len(results) <= 2
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
