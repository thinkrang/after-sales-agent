"""给已有向量块补齐 tenant 元数据。

为什么必须有这一步
------------------
检索前过滤用的是 `where={"tenant": {"$in": [...]}}`。
Chroma 的 where 只匹配「字段存在且值符合」的文档 ——
**没有 tenant 字段的老数据会被直接排除**。

所以开启权限过滤前，必须把存量数据补上 tenant，否则用户会突然
发现自己连商品价格都查不到了（而且是静默失败，很难排查）。

这个脚本幂等，可以重复跑。
"""

from pathlib import Path

from rag import (
    DEFAULT_TENANT,
    get_vector_store,
    tenant_from_filename,
)


def main():
    store = get_vector_store()
    collection = store._collection

    raw = collection.get(include=["metadatas"])
    ids = raw.get("ids") or []
    metas = raw.get("metadatas") or []

    print(f"共 {len(ids)} 个向量块")

    update_ids = []
    update_metas = []
    stats: dict[str, int] = {}

    for chunk_id, meta in zip(ids, metas):
        meta = dict(meta or {})

        # 已经有的不覆盖 —— 尊重显式设置的租户
        if meta.get("tenant"):
            stats[meta["tenant"]] = stats.get(meta["tenant"], 0) + 1
            continue

        tenant = tenant_from_filename(meta.get("source", ""))
        meta["tenant"] = tenant
        update_ids.append(chunk_id)
        update_metas.append(meta)
        stats[tenant] = stats.get(tenant, 0) + 1

    if update_ids:
        # 分批更新，避免一次提交太多
        batch = 200
        for start in range(0, len(update_ids), batch):
            collection.update(
                ids=update_ids[start:start + batch],
                metadatas=update_metas[start:start + batch],
            )
        print(f"已更新 {len(update_ids)} 个向量块")
    else:
        print("所有向量块都已有 tenant 字段，无需更新")

    print("\n租户分布：")
    for tenant, count in sorted(stats.items()):
        print(f"  {tenant:<12} {count} 个块")

    # 校验：不该再有缺失
    raw2 = collection.get(include=["metadatas"])
    missing = [
        (m or {}).get("source", "?")
        for m in (raw2.get("metadatas") or [])
        if not (m or {}).get("tenant")
    ]
    if missing:
        print(f"\n⚠ 仍有 {len(missing)} 个块缺少 tenant：{missing[:5]}")
    else:
        print(f"\n✓ 全部向量块都有 tenant 字段（默认租户：{DEFAULT_TENANT}）")


if __name__ == "__main__":
    main()
