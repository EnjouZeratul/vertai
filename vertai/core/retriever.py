"""Retriever abstraction (S3).

Composes an :class:`~vertai.core.embedding.EmbeddingProvider` with a
:class:`~vertai.core.vector.VectorStore`. ``docs/ARCHITECTURE.md`` 3.4 defines
the contract. Reranking / query transformation hook in via subclassing
(:meth:`VectorRetriever._rerank`, :meth:`VectorRetriever._transform_query`);
1.0 provides the extension point, advanced retrievers (HyDE / multi-query /
reranker models) land in 1.x.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vertai.core.embedding import EmbeddingProvider
from vertai.core.vector import SearchResult, VectorStore

__all__ = ["Retriever", "VectorRetriever"]


class Retriever(ABC):
    """Retriever abstraction. Returns ranked :class:`SearchResult` for a query."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Retrieve the top-``k`` results for ``query``."""

    @abstractmethod
    async def aretrieve(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Async retrieve."""


class VectorRetriever(Retriever):
    """Combine an :class:`EmbeddingProvider` and a :class:`VectorStore`.

    The query is embedded by the provider and the resulting vector is handed to
    the store's :meth:`VectorStore.search`. Subclasses override
    :meth:`_transform_query` / :meth:`_rerank` to add query rewriting or
    reranking (1.0 extension point; concrete advanced retrievers arrive in 1.x).

    The embedding provider is also exposed via :attr:`embedding` so callers that
    index documents (e.g. :class:`~vertai.scenarios.knowledge_qa.KnowledgeQA`)
    can reuse the same provider for indexing and querying.
    """

    def __init__(self, embedding: EmbeddingProvider, store: VectorStore) -> None:
        self.embedding = embedding
        self.store = store

    def retrieve(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        transformed = self._transform_query(query)
        query_vector = self.embedding.embed(transformed)[0]
        results = self.store.search(query_vector, top_k=top_k)
        return self._rerank(transformed, results)[:top_k]

    async def aretrieve(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        transformed = self._transform_query(query)
        vectors = await self.embedding.aembed(transformed)
        results = self.store.search(vectors[0], top_k=top_k)
        return self._rerank(transformed, results)[:top_k]

    def _transform_query(self, query: str) -> str:
        """Hook for query rewriting. Default: identity."""
        return query

    def _rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Hook for reranking. Default: preserve store ordering."""
        return results
