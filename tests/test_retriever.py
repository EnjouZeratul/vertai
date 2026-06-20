"""Tests for the Retriever abstraction (S3).

Uses a deterministic (non-semantic) EmbeddingProvider and the real
InMemoryVectorStore. Real assertions on retrieval mechanics, ordering, top_k,
and the rerank/transform-query extension points.
"""

from __future__ import annotations

import asyncio

import pytest

from vertai.core.embedding import EmbeddingProvider
from vertai.core.retriever import Retriever, VectorRetriever
from vertai.core.vector import Document, InMemoryVectorStore, SearchResult

from tests._helpers import DeterministicEmbeddingProvider


def _index_store(
    provider: EmbeddingProvider, docs: list[Document]
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    embeddings = provider.embed([d.content for d in docs])
    store.add(docs, embeddings)
    return store


class TestRetrieverABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            Retriever()  # type: ignore[abstract]


class TestVectorRetriever:
    def test_retrieve_returns_results(self) -> None:
        provider = DeterministicEmbeddingProvider()
        docs = [
            Document(content="python programming language"),
            Document(content="machine learning basics"),
            Document(content="cooking recipes for dinner"),
        ]
        store = _index_store(provider, docs)
        retriever = VectorRetriever(provider, store)
        results = retriever.retrieve("python", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_retrieve_top_k_limits_results(self) -> None:
        provider = DeterministicEmbeddingProvider()
        docs = [Document(content=f"document number {i}") for i in range(10)]
        store = _index_store(provider, docs)
        retriever = VectorRetriever(provider, store)
        results = retriever.retrieve("document", top_k=3)
        assert len(results) == 3

    def test_retrieve_empty_store(self) -> None:
        provider = DeterministicEmbeddingProvider()
        store = InMemoryVectorStore()
        retriever = VectorRetriever(provider, store)
        assert retriever.retrieve("anything", top_k=5) == []

    def test_results_ordered_by_score_desc(self) -> None:
        provider = DeterministicEmbeddingProvider()
        docs = [
            Document(content="alpha alpha alpha"),
            Document(content="alpha"),
            Document(content="completely different content"),
        ]
        store = _index_store(provider, docs)
        retriever = VectorRetriever(provider, store)
        results = retriever.retrieve("alpha alpha alpha", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_aretrieve_matches_retrieve(self) -> None:
        provider = DeterministicEmbeddingProvider()
        docs = [Document(content="some document text")]
        store = _index_store(provider, docs)
        retriever = VectorRetriever(provider, store)
        sync = retriever.retrieve("some document", top_k=1)
        async_ = asyncio.run(retriever.aretrieve("some document", top_k=1))
        assert len(async_) == len(sync)
        assert async_[0].document.doc_id == sync[0].document.doc_id

    def test_transform_query_hook(self) -> None:
        provider = DeterministicEmbeddingProvider()

        class QueryRewritingRetriever(VectorRetriever):
            def _transform_query(self, query: str) -> str:
                return "document" if "doc" in query.lower() else query

        docs = [Document(content="document content here")]
        store = _index_store(provider, docs)
        retriever = QueryRewritingRetriever(provider, store)
        results = retriever.retrieve("find me a doc", top_k=1)
        assert len(results) == 1

    def test_rerank_hook_reorders(self) -> None:
        provider = DeterministicEmbeddingProvider()

        class ReversingRetriever(VectorRetriever):
            def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
                return list(reversed(results))

        docs = [Document(content=f"doc {i}") for i in range(5)]
        store = _index_store(provider, docs)
        retriever = ReversingRetriever(provider, store)
        raw = VectorRetriever(provider, store).retrieve("doc", top_k=5)
        reranked = retriever.retrieve("doc", top_k=5)
        # Rerank reverses, so the first of raw should be the last of reranked.
        assert raw[0].document.doc_id == reranked[-1].document.doc_id
