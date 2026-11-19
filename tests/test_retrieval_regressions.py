"""用可控文档/打分器验证权限与候选边界，不下载远程模型。"""
import subprocess
import sys

import pytest
from langchain_core.documents import Document

import rag
import retrieval
from evaluation import evaluate_rag
from evaluation import tune_rrf


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VECTOR_DB_DIR", tmp_path)
    retrieval.refresh_corpus()
    yield tmp_path
    retrieval.refresh_corpus()


def test_duplicate_text_does_not_switch_tenant_or_source(monkeypatch, isolated_index):
    class Store:
        def get(self, **kwargs):
            return dict(ids=["public-id", "secret-id"], documents=["same", "same"],
                        metadatas=[dict(source="public.txt", tenant="public"),
                                   dict(source="private.txt", tenant="globex")])
    monkeypatch.setattr(retrieval, "get_vector_store", Store)
    monkeypatch.setattr(retrieval, "vector_retrieval", lambda *a, **k: [
        dict(id="public-id", content="same"), dict(id="secret-id", content="same")])
    monkeypatch.setattr(retrieval, "bm25_retrieval", lambda *a, **k: [])
    result = retrieval.hybrid_retrieval("same", 3, tenant="acme")
    assert [(r["id"], r["source"], r["tenant"]) for r in result] == [
        ("public-id", "public.txt", "public")]


def test_bm25_refresh_and_other_process_revision(monkeypatch, isolated_index):
    current = ["old"]
    class Store:
        def get(self, **kwargs):
            return dict(ids=current[:], documents=current[:],
                        metadatas=[dict(source=x, tenant="public") for x in current])
    class Scorer:
        def __init__(self, docs):
            self.docs = docs
        def get_scores(self, query):
            return [1 if query == d else 0 for d in self.docs]
    monkeypatch.setattr(retrieval, "get_vector_store", Store)
    monkeypatch.setattr(retrieval, "tokenize", lambda text: [text])
    monkeypatch.setattr(retrieval, "BM25Okapi", Scorer)
    assert retrieval.bm25_retrieval("old", 3)[0]["id"] == "old"
    current[:] = ["new"]
    retrieval.refresh_corpus()
    assert retrieval.bm25_retrieval("old", 3) == []
    assert retrieval.bm25_retrieval("new", 3)[0]["id"] == "new"
    # 模拟 Celery 另一个进程发布版本，API 不调用 refresh 也能发现。
    current[:] = ["worker-added"]
    subprocess.run([sys.executable, "-c",
                    "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('worker-v2', encoding='ascii')",
                    str(isolated_index / ".corpus-revision")], check=True)
    assert retrieval.bm25_retrieval("new", 3) == []
    assert retrieval.bm25_retrieval("worker-added", 3)[0]["id"] == "worker-added"


def test_rerank_top10_can_retrieve_30_candidates(monkeypatch, isolated_index):
    seen = []
    class Store:
        def get(self, **kwargs):
            return dict(ids=[f"id{i}" for i in range(30)], documents=[f"doc{i}" for i in range(30)],
                        metadatas=[dict(source=f"s{i}", tenant="public") for i in range(30)])
        def similarity_search_with_score(self, *, query, k, filter):
            seen.append(k)
            return [(Document(id=f"id{i}", page_content=f"doc{i}",
                              metadata=dict(source=f"s{i}", tenant="public")), 0.1) for i in range(k)]
    class Reranker:
        def predict(self, pairs):
            return list(range(len(pairs)))
    store = Store()
    monkeypatch.setattr(rag, "get_vector_store", lambda: store)
    monkeypatch.setattr(retrieval, "get_vector_store", lambda: store)
    monkeypatch.setattr(retrieval, "bm25_retrieval", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "_reranker", Reranker)
    rows = retrieval.retrieve("query", top_k=10, strategy="hybrid_rerank")
    assert seen == [30] and len(rows) == 10
    assert rows[0]["id"] == "id29"
    with pytest.raises(ValueError):
        retrieval.retrieve("query", top_k=11)


def test_evaluation_uses_actual_top3_not_top10_prefix(monkeypatch):
    calls = []
    def retrieve(question, top_k, **kwargs):
        calls.append(top_k)
        return [dict(rank=1, source="expected" if top_k == 10 else "wrong")]
    monkeypatch.setattr(evaluate_rag, "retrieve", retrieve)
    report = evaluate_rag.run_strategy("hybrid_rerank", [dict(id="test", type="easy", question="q", expected_sources=["expected"])])
    assert calls == [3, 10]
    assert report["recall3"] == report["hit1"] == 0
    assert report["recall"] == report["mrr"] == 1


def test_zero_weight_does_not_add_keyword_only_documents():
    assert retrieval._rrf_fuse([["vector"], ["keyword"]], [1, 0]) == [("vector", 1 / 61)]


def test_tuning_restores_weights_when_retrieval_fails(monkeypatch):
    original = retrieval.VECTOR_WEIGHT, retrieval.BM25_WEIGHT
    def fail(*a, **k):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(tune_rrf, "run_strategy", fail)
    with pytest.raises(RuntimeError):
        tune_rrf.score(0.4, 0.6, [])
    assert (retrieval.VECTOR_WEIGHT, retrieval.BM25_WEIGHT) == original


def test_empty_collection_does_not_crash_bm25(monkeypatch, isolated_index):
    class Store:
        def get(self, **kwargs):
            return dict(ids=[], documents=[], metadatas=[])
    monkeypatch.setattr(retrieval, "get_vector_store", Store)
    assert retrieval.bm25_retrieval("new", 3) == []
