"""Tests for the vector store abstraction (S3 refactor).

Testing strategy (per ROADMAP test table):
- ``Document`` / ``SearchResult`` / ``VectorConfig`` dataclasses -> real asserts.
- ``InMemoryVectorStore`` -> real assertions with a deterministic embedding
  provider (no hash-random, no mock loops). Verifies C2 (no provider raises)
  and C3 (delete consistency for the in-memory backend).
- ``FAISSVectorStore`` -> ``@integration``, honestly skipped when faiss/numpy
  are not installed; when installed, runs against the real faiss+numpy and
  verifies C3 (delete keeps count/search consistent). No mock loops.
- ``ChromaVectorStore`` -> ``@integration``, honestly skipped when chromadb is
  not installed; when installed, runs against a real ephemeral Chroma client.
- ``VectorEngine`` -> real assertions on backend selection (``auto`` honors
  Chroma > FAISS > InMemory), C2 (no provider -> explicit raise), indexing and
  search through a deterministic provider, and hybrid search fusion.

Removed vs. the previous file: the ``TestVectorStoreAbstract`` class that
exercised ABC ``pass`` statements to inflate coverage, the Chroma/FAISS
``patch.dict('sys.modules')`` mock loops that validated "the mock was called"
rather than real behavior, and the ``TestNumpyNotAvailable`` import-reload
coverage test.
"""

from __future__ import annotations

import pytest

from vertai.core.vector import (
    ChromaVectorStore,
    Document,
    FAISSVectorStore,
    InMemoryVectorStore,
    SearchResult,
    VectorConfig,
    VectorEngine,
)

from tests._helpers import (
    HAS_CHROMA,
    HAS_FAISS,
    DeterministicEmbeddingProvider,
    requires_extra,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDocument:
    def test_auto_id_from_content(self) -> None:
        doc = Document(content="hello", metadata={"src": "a"})
        assert doc.content == "hello"
        assert doc.metadata == {"src": "a"}
        assert len(doc.doc_id) == 12

    def test_same_content_same_id(self) -> None:
        assert Document(content="x").doc_id == Document(content="x").doc_id

    def test_custom_id_preserved(self) -> None:
        assert Document(content="x", doc_id="custom").doc_id == "custom"

    def test_default_metadata_empty(self) -> None:
        assert Document(content="x").metadata == {}


class TestSearchResult:
    def test_fields(self) -> None:
        r = SearchResult(document=Document(content="x"), score=0.5, distance=0.5)
        assert r.score == 0.5
        assert r.distance == 0.5


class TestVectorConfig:
    def test_defaults(self) -> None:
        c = VectorConfig()
        assert c.top_k == 5
        assert c.collection_name == "default"


# ---------------------------------------------------------------------------
# InMemoryVectorStore (real, no mocks)
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    def _store_with_docs(
        self, contents: list[str], dim: int = 8
    ) -> InMemoryVectorStore:
        provider = DeterministicEmbeddingProvider(dimension=dim)
        store = InMemoryVectorStore()
        docs = [Document(content=c) for c in contents]
        embeddings = provider.embed(contents)
        store.add(docs, embeddings)
        return store

    def test_add_and_count(self) -> None:
        store = self._store_with_docs(["a", "b", "c"])
        assert store.count() == 3

    def test_add_mismatched_lengths_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            store.add([Document(content="a")], [[0.1, 0.2], [0.3, 0.4]])

    def test_search_returns_results(self) -> None:
        store = self._store_with_docs(
            ["python programming", "java development", "cooking recipes"]
        )
        provider = DeterministicEmbeddingProvider()
        qv = provider.embed("python programming")[0]
        results = store.search(qv, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_empty_store(self) -> None:
        store = InMemoryVectorStore()
        assert store.search([0.1, 0.2], top_k=5) == []

    def test_search_top_k_limit(self) -> None:
        store = self._store_with_docs([f"doc {i}" for i in range(10)])
        qv = store._vectors[store._documents[next(iter(store._documents))].doc_id]
        results = store.search(qv, top_k=3)
        assert len(results) == 3

    def test_search_orders_by_score_desc(self) -> None:
        store = self._store_with_docs(["alpha alpha", "alpha", "zzz zzz"])
        provider = DeterministicEmbeddingProvider()
        qv = provider.embed("alpha alpha")[0]
        results = store.search(qv, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete_removes_and_keeps_consistency(self) -> None:
        store = self._store_with_docs(["one", "two", "three"])
        ids = list(store._documents.keys())
        store.delete(ids[:2])
        # C3 invariant: count and search reflect the deletion.
        assert store.count() == 1
        provider = DeterministicEmbeddingProvider()
        qv = provider.embed("one")[0]
        results = store.search(qv, top_k=5)
        assert all(r.document.doc_id not in ids[:2] for r in results)

    def test_delete_nonexistent_is_noop(self) -> None:
        store = self._store_with_docs(["one"])
        store.delete(["does-not-exist"])
        assert store.count() == 1

    def test_zero_vector_similarity_is_zero(self) -> None:
        # Cosine similarity with a zero vector returns 0.0 (no division error).
        assert InMemoryVectorStore._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# VectorEngine (real, deterministic provider)
# ---------------------------------------------------------------------------


class TestVectorEngine:
    def test_memory_store_creation(self) -> None:
        engine = VectorEngine(store_type="memory")
        assert isinstance(engine.store, InMemoryVectorStore)

    def test_index_and_search_with_provider(self) -> None:
        provider = DeterministicEmbeddingProvider()
        engine = VectorEngine(
            store_type="memory", embedding_provider=provider
        )
        ids = engine.index_documents(
            [Document(content="machine learning"), Document(content="deep learning")]
        )
        assert len(ids) == 2
        assert engine.count() == 2
        results = engine.search("learning", top_k=2)
        assert len(results) > 0

    def test_no_provider_raises_c2(self) -> None:
        # C2: no hash-random fallback; explicit raise on use.
        engine = VectorEngine(store_type="memory")
        with pytest.raises(RuntimeError, match="No EmbeddingProvider"):
            engine.index_documents([Document(content="x")])
        with pytest.raises(RuntimeError, match="No EmbeddingProvider"):
            engine.search("x")

    def test_embedding_fn_wrapped_as_provider(self) -> None:
        def fn(text: str) -> list[float]:
            return [float(len(text))]

        engine = VectorEngine(store_type="memory", embedding_fn=fn)
        # embedding_fn is wrapped; indexing/search work.
        engine.index_documents([Document(content="hello")])
        assert engine.count() == 1
        results = engine.search("hello", top_k=1)
        assert len(results) == 1

    def test_count_zero_initially(self) -> None:
        engine = VectorEngine(
            store_type="memory",
            embedding_provider=DeterministicEmbeddingProvider(),
        )
        assert engine.count() == 0

    def test_delete_documents(self) -> None:
        provider = DeterministicEmbeddingProvider()
        engine = VectorEngine(store_type="memory", embedding_provider=provider)
        docs = [Document(content=f"doc {i}") for i in range(3)]
        ids = engine.index_documents(docs)
        assert engine.count() == 3
        engine.delete_documents(ids[:2])
        assert engine.count() == 1

    def test_hybrid_search_fuses_keyword_and_vector(self) -> None:
        provider = DeterministicEmbeddingProvider()
        engine = VectorEngine(store_type="memory", embedding_provider=provider)
        engine.index_documents(
            [
                Document(content="python tutorial for beginners"),
                Document(content="java tutorial advanced"),
                Document(content="python data analysis"),
            ]
        )
        results = engine.hybrid_search(
            query="python", keywords=["python", "tutorial"], top_k=3
        )
        assert len(results) <= 3
        # Documents containing both keywords should rank first.
        top = results[0].document.content.lower()
        assert "python" in top and "tutorial" in top

    def test_hybrid_search_without_keywords_degenerates_to_vector(self) -> None:
        provider = DeterministicEmbeddingProvider()
        engine = VectorEngine(store_type="memory", embedding_provider=provider)
        engine.index_documents([Document(content="python"), Document(content="java")])
        results = engine.hybrid_search(query="python", keywords=None, top_k=2)
        assert len(results) <= 2

    def test_auto_falls_back_to_memory_when_no_extras(self) -> None:
        # When neither chromadb nor faiss is installed, auto -> InMemory.
        if HAS_CHROMA or HAS_FAISS:
            pytest.skip("auto selection only assertable when no vector extras installed")
        engine = VectorEngine(store_type="auto")
        assert isinstance(engine.store, InMemoryVectorStore)

    def test_chroma_unavailable_raises(self) -> None:
        if HAS_CHROMA:
            pytest.skip("chromadb installed; cannot test the unavailable path")
        engine = VectorEngine(store_type="chroma")
        with pytest.raises(RuntimeError, match="ChromaDB not installed"):
            _ = engine.store

    def test_faiss_unavailable_raises(self) -> None:
        if HAS_FAISS:
            pytest.skip("faiss installed; cannot test the unavailable path")
        engine = VectorEngine(store_type="faiss")
        with pytest.raises(RuntimeError, match="FAISS not installed"):
            _ = engine.store

    def test_auto_prefers_chroma_when_available(self) -> None:
        if not HAS_CHROMA:
            pytest.skip("chromadb not installed")
        engine = VectorEngine(store_type="auto")
        assert isinstance(engine.store, ChromaVectorStore)

    def test_auto_uses_faiss_when_chroma_missing(self) -> None:
        if HAS_CHROMA or not HAS_FAISS:
            pytest.skip("requires faiss without chromadb")
        provider = DeterministicEmbeddingProvider(dimension=64)
        engine = VectorEngine(
            store_type="auto", embedding_provider=provider
        )
        assert isinstance(engine.store, FAISSVectorStore)


# ---------------------------------------------------------------------------
# FAISS integration (real faiss + numpy; honest skip otherwise)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@requires_extra("faiss", HAS_FAISS)
class TestFAISSVectorStoreIntegration:
    def _build(self, dim: int = 32) -> tuple[FAISSVectorStore, DeterministicEmbeddingProvider]:
        provider = DeterministicEmbeddingProvider(dimension=dim)
        store = FAISSVectorStore(dimension=dim)
        return store, provider

    def test_add_search_count(self) -> None:
        store, provider = self._build()
        docs = [Document(content=f"doc number {i}") for i in range(5)]
        store.add(docs, provider.embed([d.content for d in docs]))
        assert store.count() == 5
        qv = provider.embed("doc number 0")[0]
        results = store.search(qv, top_k=3)
        assert len(results) == 3

    def test_delete_consistency_c3(self) -> None:
        # C3: after delete, count() and search() must not return deleted docs.
        store, provider = self._build()
        docs = [Document(content=f"unique doc {i}", doc_id=f"id-{i}") for i in range(4)]
        store.add(docs, provider.embed([d.content for d in docs]))
        assert store.count() == 4
        store.delete(["id-0", "id-1"])
        # count reflects live docs (not index.ntotal).
        assert store.count() == 2
        qv = provider.embed("unique doc")[0]
        results = store.search(qv, top_k=10)
        # No deleted doc appears.
        returned_ids = {r.document.doc_id for r in results}
        assert "id-0" not in returned_ids
        assert "id-1" not in returned_ids
        assert returned_ids <= {"id-2", "id-3"}

    def test_dimension_mismatch_raises(self) -> None:
        store, provider = self._build(dim=8)
        with pytest.raises(ValueError, match="dimension"):
            store.add(
                [Document(content="x")],
                [[0.1] * 16],  # wrong dim
            )

    def test_search_empty(self) -> None:
        store, _ = self._build()
        assert store.search([0.1] * 32, top_k=5) == []

    def test_count_uninitialized(self) -> None:
        store, _ = self._build()
        # Before init, count is 0 (no documents).
        store2 = FAISSVectorStore(dimension=8)
        assert store2.count() == 0


# ---------------------------------------------------------------------------
# Chroma integration (real chromadb ephemeral client; honest skip otherwise)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@requires_extra("chromadb", HAS_CHROMA)
class TestChromaVectorStoreIntegration:
    def _build(self) -> tuple[ChromaVectorStore, DeterministicEmbeddingProvider]:
        provider = DeterministicEmbeddingProvider(dimension=16)
        store = ChromaVectorStore(collection_name="vertai_test")
        return store, provider

    def test_add_search_count_delete(self) -> None:
        store, provider = self._build()
        docs = [Document(content=f"chroma doc {i}") for i in range(3)]
        store.add(docs, provider.embed([d.content for d in docs]))
        assert store.count() == 3
        qv = provider.embed("chroma doc 0")[0]
        results = store.search(qv, top_k=2)
        assert len(results) <= 2
        store.delete([docs[0].doc_id])
        assert store.count() == 2
