import hashlib
import os
import re
from threading import Lock
from pathlib import Path
from functools import lru_cache
from io import BytesIO
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from pypdf import PdfReader

# 1. 项目路径
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
UPLOAD_DIR = BASE_DIR / "uploads"
index_lock = Lock()
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


# 2. 初始化 Embedding 和向量数据库
@lru_cache(maxsize=1)
def get_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name="shop_knowledge",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )

    return vector_store


# 3. 读取知识库文档

# ---------------------------------------------------------------- 多租户

DEFAULT_TENANT = "public"

# 文件名约定：tenant_<租户名>_xxx.txt 归属该租户，其余为公共文档。
# 用文件名而不是 sidecar 配置，是因为这样零配置、一眼能看懂，
# 加文档时不用同步维护另一份映射表。
_TENANT_PREFIX = "tenant_"


def tenant_from_filename(filename: str) -> str:
    """从文件名推断租户。tenant_acme_price.txt -> acme

    用 stem（去掉扩展名）再解析，否则 tenant_.txt 会被解析成 ".txt"。
    租户名必须是纯字母数字下划线，不符合的一律落回 public ——
    宁可少给权限，也不能因为文件名奇怪就造出一个奇怪的租户。
    """
    name = Path(filename).name
    if not name.startswith(_TENANT_PREFIX):
        return DEFAULT_TENANT

    rest = Path(name).stem[len(_TENANT_PREFIX):]
    tenant = rest.split("_", 1)[0].strip().lower()

    if not tenant or not re.fullmatch(r"[a-z0-9]+", tenant):
        return DEFAULT_TENANT
    return tenant


def visible_tenants(tenant: str | None) -> list[str]:
    """某个用户能看到的租户集合：自己的 + 公共的。

    注意公共文档用 DEFAULT_TENANT 标记，所以任何租户都要带上它，
    否则用户会连商品价格这类公共信息都查不到。
    """
    if not tenant or tenant == DEFAULT_TENANT:
        return [DEFAULT_TENANT]
    return [tenant, DEFAULT_TENANT]


def tenant_filter(tenant: str | None) -> dict | None:
    """构造 Chroma 的 where 条件。

    这就是「检索前过滤」：把权限条件交给向量库，只在可见子集里做 ANN。
    换成「检索后过滤」的话，top-k 会被无权限文档占满，
    筛完可能一条不剩 —— 候选池被别人的数据挤占了。
    """
    tenants = visible_tenants(tenant)
    if len(tenants) == 1:
        return {"tenant": tenants[0]}
    return {"tenant": {"$in": tenants}}


def load_documents():
    documents = []

    files = sorted(
        path
        for suffix in ("*.txt", "*.md")
        for path in KNOWLEDGE_DIR.glob(suffix)
    )

    if not files:
        raise FileNotFoundError(
            "knowledge 文件夹中没有找到 TXT 文档"
        )

    for file in files:
        text = file.read_text(encoding="utf-8")

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file.name,
                    "tenant": tenant_from_filename(file.name),
                },
            )
        )

    return documents


# 4. 文档切分和向量化
def build_index():

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("没有可索引的文档内容")

    vector_store = get_vector_store()

    # 演示项目：重建整个知识库
    vector_store.reset_collection()
    vector_store.add_documents(chunks)

    print(f"知识库构建完成，共 {len(chunks)} 个文本块")


# 5. 检索相关知识
def search_knowledge(question: str) -> str:

    vector_store = get_vector_store()

    # 检查知识库是否为空
    if not vector_store.get(limit=1)["ids"]:
        return "知识库为空，请先运行 rag.py 构建索引。"

    results = vector_store.similarity_search(
        query=question,
        k=3,
    )

    if not results:
        return "没有找到相关文档。"

    # 返回检索内容和来源
    passages = []

    for doc in results:
        source = doc.metadata.get("source", "未知文档")
        page = doc.metadata.get("page")

        if page is not None:
            source_label = f"{source}，第 {page} 页"
        else:
            source_label = source

        passages.append(
            f"【来源：{source_label}】\n{doc.page_content}"
        )

    return "\n\n".join(passages)
def format_evidence(results: list[dict]) -> str:
    """把 debug_retrieval 的结果格式化成带来源标注的文本。"""
    if not results:
        return "没有找到相关文档。"

    passages = []

    for item in results:
        source = item.get("source", "未知文档")
        page = item.get("page")

        if page is not None:
            source_label = f"{source}，第 {page} 页"
        else:
            source_label = source

        passages.append(f"【来源：{source_label}】\n{item['content']}")

    return "\n\n".join(passages)

def add_txt_document(filename:str,file_bytes)->dict:
    """将上传的TXT文档增量添加到Chroma."""
    #解码文件
    try:
        text=file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("请上传UTF-8编码的TXT文件")
    if not text.strip():
        raise ValueError("文件内容不能为空")
    doc_id=hashlib.sha256(file_bytes).hexdigest()
    safe_name=Path(
        filename.replace("\\","/")
    ).name
    document=Document(
        page_content=text,
        metadata={
            "source":safe_name,
            "doc_id":doc_id,
        }
    )
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks=splitter.split_documents([document])
    chunk_ids=[
        f"{doc_id}_{i}"
        for i in range(len(chunks))
    ]
    store=get_vector_store()
    with index_lock:
# 检查是否已经上传相同内容
        exiting=store.get(
            where={"doc_id":doc_id},
            limit=1,
        )
        if exiting["ids"]:
            return{
                "filename":safe_name,
                "doc_id":doc_id,
                "chunks":0,
                "duplicate":True,
            }
        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )
        saved_path=UPLOAD_DIR/f"{doc_id}.txt"
        saved_path.write_bytes(file_bytes)
        try:
            store.add_documents(
                documents=chunks,
                ids=chunk_ids,
            )
        except Exception:
            saved_path.unlink(missing_ok=True)
            try:
                store.delete(ids=chunk_ids)
            except Exception:
                pass
            raise
        print(
        f"[RAG] 新增文件：{safe_name}，"
        f"生成 {len(chunks)} 个文本块"
    )

        return {
            "filename": safe_name,
            "doc_id": doc_id,
            "chunks": len(chunks),
            "duplicate": False,
        }

# 查询知识库文档列表
def list_knowledge_documents(tenant: str | None = None):

    store = get_vector_store()

    get_kwargs = {"include": ["metadatas"]}
    if tenant:
        get_kwargs["where"] = tenant_filter(tenant)
    with index_lock:
        records = store.get(**get_kwargs)

    documents = {}

    for metadata in records.get("metadatas", []):

        if not metadata:
            continue

        filename = metadata.get("source", "未知文档")
        doc_id = metadata.get("doc_id")

        # 有合法 doc_id 的，是通过上传接口建立的文档
        uploaded = (
            isinstance(doc_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", doc_id)
            is not None
        )

        if uploaded:
            key = f"uploaded:{doc_id}"
        else:
            # 兼容之前手动建立的知识库文档
            key = f"initial:{filename}"

        if key not in documents:
            documents[key] = {
                "filename": filename,
                "doc_id": doc_id if uploaded else None,
                "type": "uploaded" if uploaded else "initial",
                "tenant": metadata.get("tenant") or tenant_from_filename(filename),
                "chunks": 0,
            }

        documents[key]["chunks"] += 1

    return sorted(
        documents.values(),
        key=lambda item: item["filename"].lower()
    )


# 删除指定的上传文档及向量
def delete_knowledge_document(
    doc_id: str,
    *,
    tenant: str | None = None,
):

    # 防止非法 ID
    if not re.fullmatch(r"[0-9a-f]{64}", doc_id):
        raise ValueError("非法文档 ID")

    store = get_vector_store()

    with index_lock:

        # 查找属于该文档的所有向量
        records = store.get(
            where=(
                {"$and": [{"doc_id": doc_id}, {"tenant": tenant}]}
                if tenant
                else {"doc_id": doc_id}
            )
        )

        ids = records["ids"]

        if not ids:
            return False

        # 先删除 Chroma 向量
        store.delete(ids=ids)

        # 再删除本地保存的原始文件
    for ext in (".txt", ".md", ".pdf"):
        file_path = UPLOAD_DIR / f"{doc_id}{ext}"
        file_path.unlink(missing_ok=True)

    return True

def add_knowledge_document(
    filename: str,
    file_bytes: bytes,
    tenant: str | None = None,
) -> dict:
    """将 TXT、Markdown、PDF 文档增量写入 Chroma。

    tenant 指定文档归属。不传则按文件名约定推断
    （tenant_<租户名>_xxx -> 该租户），再不行落到 public。
    """

    safe_name = Path(filename.replace("\\", "/")).name
    ext = Path(safe_name).suffix.lower()

    # 显式传入优先，否则按文件名约定推断
    owner = (tenant or "").strip().lower() or tenant_from_filename(safe_name)

    if ext not in {".txt", ".md", ".pdf"}:
        raise ValueError("只支持 TXT、Markdown 和 PDF 文件")

    if not file_bytes or len(file_bytes) > 5 * 1024 * 1024:
        raise ValueError("文件不能为空，且不能超过 5MB")

    # Include the tenant in the idempotency key. Hashing content alone makes a
    # tenant-B upload look like a duplicate of the same bytes in tenant A,
    # causing the second tenant to receive an unretrievable document.
    doc_id = hashlib.sha256(
        owner.encode("utf-8") + b"\0" + file_bytes
    ).hexdigest()

    documents = []

    # ---------- 1. PDF 解析 ----------
    if ext == ".pdf":
        try:
            reader = PdfReader(BytesIO(file_bytes))

            if reader.is_encrypted:
                raise ValueError("暂不支持加密 PDF")

            if len(reader.pages) > 20:
                raise ValueError("演示版最多支持 20 页 PDF")

            for page_num, page in enumerate(reader.pages, start=1):

                text = page.extract_text() or ""

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": safe_name,
                            "doc_id": doc_id,
                            "page": page_num,
                            "file_ext": ext,
                            "tenant": owner,
                        },
                    )
                )

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"PDF 解析失败，请检查文件是否有效：{e}"
            ) from e

    # ---------- 2. TXT / Markdown 解析 ----------
    else:
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("请使用 UTF-8 编码的文本文件")

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": safe_name,
                        "doc_id": doc_id,
                        "file_ext": ext,
                        "tenant": owner,
                    },
                )
            )

    if not documents:
        raise ValueError(
            "没有提取到有效文字。扫描版 PDF 暂不支持。"
        )

    # ---------- 3. 统一文本切分 ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(documents)

    if len(chunks) > 200:
        raise ValueError(
            "文档内容过长，演示版最多支持 200 个文本块"
        )

    chunk_ids = [
        f"{doc_id}_{i}"
        for i in range(len(chunks))
    ]

    # ---------- 4. 写入 Chroma ----------
    store = get_vector_store()

    with index_lock:

        existing = store.get(
            where={"doc_id": doc_id},
            limit=1,
        )

        # 内容相同则不重复建立索引
        if existing["ids"]:
            return {
                "filename": safe_name,
                "doc_id": doc_id,
                "chunks": 0,
                "duplicate": True,
            }

        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_path = UPLOAD_DIR / f"{doc_id}{ext}"

        # 保存原始文档
        saved_path.write_bytes(file_bytes)

        try:
            # 增量写入，不清空旧知识
            store.add_documents(
                documents=chunks,
                ids=chunk_ids,
            )

        except Exception:
            # 索引失败时尽可能回滚
            saved_path.unlink(missing_ok=True)

            try:
                store.delete(ids=chunk_ids)
            except Exception:
                pass

            raise

    print(
        f"[RAG] 文档上传成功：{safe_name}，"
        f"生成 {len(chunks)} 个文本块"
    )

    return {
        "filename": safe_name,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "duplicate": False,
    }

# RAG 检索调试函数
def debug_retrieval(
    question: str,
    top_k: int = 3,
    tenant: str | None = None,
) -> list[dict]:
    """
    返回 Chroma 的真实检索结果：
    文档来源、排名、距离、页码和文本内容。

    tenant 不为空时，只在「该租户 + 公共」的文档里检索。
    """

    question = question.strip()

    if not question:
        raise ValueError("测试问题不能为空")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k 必须在 1 到 10 之间")

    # 复用原来的 Chroma 向量数据库
    store = get_vector_store()

    where = tenant_filter(tenant)

    # 与上传/删除共用锁
    with index_lock:
        matches = store.similarity_search_with_score(
            query=question,
            k=top_k,
            filter=where,
        )

    results = []

    for rank, (doc, distance) in enumerate(matches, start=1):

        results.append({
            "rank": rank,
            "source": doc.metadata.get("source", "未知文档"),
            "tenant": doc.metadata.get("tenant", DEFAULT_TENANT),
            "page": doc.metadata.get("page"),
            "distance": round(float(distance), 6),
            "content": doc.page_content,
        })

    return results
# 6. 单独测试 RAG
if __name__ == "__main__":

    print("正在构建知识库...")
    build_index()

    question = "衣服买了八天，不喜欢了还能退货吗？"

    print(f"\n测试问题：{question}")
    print("\n检索结果：")

    print(search_knowledge(question))
