"""重建向量索引，并保留此前通过 /knowledge/upload 上传的文档。

为什么需要这个脚本
------------------
rag.build_index() 里有一行 vector_store.reset_collection()，
它会清空整个集合。对纯演示项目没问题，但只要你上传过文档，
直接跑 build_index() 就会把这些上传文档的索引一起抹掉
（文件还在 uploads/ 里，但检索不到了）。

本脚本的做法：
1. 先把集合里 type == "uploaded" 的文档连同原文和元信息读出来
2. 清空集合
3. 重新索引 knowledge/*.txt
4. 把上传文档原样写回

用法
----
    python rebuild_index.py

注意：重建前请先停掉正在运行的服务，避免两个进程同时写 Chroma。
"""

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import get_vector_store, load_documents, tenant_from_filename

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 上传文档的判定方式：metadata 里有合法的 sha256 doc_id。
# 注意：集合里并不存在 "type" 字段，"uploaded" 是 list_knowledge_documents()
# 根据 doc_id 计算出来的。之前误用 type 过滤，导致一篇都没匹配上，
# 随后 reset_collection() 把上传文档一起清空了 —— 这是个真实的教训：
# 过滤条件要对着真实数据验证，不能对着"看起来应该有"的字段写。
import re

DOC_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


def is_uploaded(meta: dict) -> bool:
    if not meta:
        return False
    doc_id = meta.get("doc_id")
    return isinstance(doc_id, str) and DOC_ID_PATTERN.fullmatch(doc_id) is not None


def main():
    store = get_vector_store()

    # ---------- 1. 备份已上传文档 ----------
    raw = store.get(include=["documents", "metadatas"])
    uploaded = []
    seen = set()
    knowledge_docs = load_documents()
    known_sources = {doc.metadata.get("source") for doc in knowledge_docs}
    known_content = {
        hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        for doc in knowledge_docs
    }

    for text, meta in zip(raw["documents"], raw["metadatas"]):
        if not is_uploaded(meta):
            continue
        if meta.get("source") in known_sources:
            continue
        if hashlib.sha256((text or "").encode("utf-8")).hexdigest() in known_content:
            continue
        key = (meta.get("doc_id"), meta.get("source"))
        if key in seen:
            continue
        seen.add(key)
        # 补 tenant：老数据可能没有这个字段，缺了会被权限过滤静默排除
        meta = dict(meta)
        if not meta.get("tenant"):
            meta["tenant"] = tenant_from_filename(meta.get("source", ""))
        uploaded.append(Document(page_content=text, metadata=meta))

    print(f"[1/4] 备份到 {len(uploaded)} 篇上传文档")
    for doc in uploaded:
        print(f"        - {doc.metadata.get('source')}")

    # ---------- 2. 读取 knowledge/*.txt ----------
    docs = knowledge_docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/4] knowledge/ 共 {len(docs)} 篇 → 切成 {len(chunks)} 个文本块")

    # ---------- 3. 重建 ----------
    store.reset_collection()
    store.add_documents(chunks)
    if uploaded:
        store.add_documents(uploaded)
    print("[3/4] 集合已重建")

    # ---------- 4. 校验 ----------
    check = store.get(include=["metadatas"])
    sources = sorted({
        (m.get("source") or m.get("filename") or "?")
        for m in check["metadatas"]
    })
    print(f"[4/4] 校验：{len(check['ids'])} 个文本块，{len(sources)} 个来源")
    for s in sources:
        print(f"        - {s}")


if __name__ == "__main__":
    main()
