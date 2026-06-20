"""Embedding provider abstraction (S3).

Defines the :class:`EmbeddingProvider` ABC per ``docs/ARCHITECTURE.md`` 3.2.
Providers are independent of :class:`~vertai.core.vector.VectorStore`: the store
holds vectors, the provider turns text into vectors. ``VectorEngine`` and
:class:`~vertai.core.retriever.Retriever` compose the two.

A real local default is :class:`LocalSentenceTransformerProvider` (requires the
``vertai[embeddings]`` extra). There is **no** hash-random fallback: when no
provider is configured, :class:`~vertai.core.vector.VectorEngine` raises
explicitly instead of silently producing non-semantic vectors (fixes C2).

:class:`FunctionEmbeddingProvider` wraps a single-text callable for backward
compatibility with the ``embedding_fn`` parameter; it does not add semantic
similarity by itself.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

__all__ = [
    "EmbeddingProvider",
    "FunctionEmbeddingProvider",
    "LocalSentenceTransformerProvider",
]


class EmbeddingProvider(ABC):
    """Embedding provider abstraction.

    Implementations turn text into dense vectors. ``embed`` / ``aembed`` accept
    a single string or a list of strings and always return
    ``list[list[float]]`` (one vector per input text), so callers do not need to
    special-case the single-text path. ``dimension`` exposes the vector width so
    a :class:`~vertai.core.vector.VectorStore` can initialize itself.
    """

    @abstractmethod
    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """Embed a single text or a batch. Returns one vector per input text."""

    @abstractmethod
    async def aembed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """Async embed. Real providers offload blocking work to a thread."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""


class FunctionEmbeddingProvider(EmbeddingProvider):
    """Wrap a single-text callable ``str -> list[float]`` as an
    :class:`EmbeddingProvider`.

    Kept for backward compatibility with the ``embedding_fn`` parameter of
    :class:`~vertai.core.vector.VectorEngine`. The callable should be
    deterministic for a given text (non-deterministic callables produce
    inconsistent retrieval). Batch calls loop over the callable.

    Note: this provider does **not** add semantic similarity by itself; supply a
    real embedding model (:class:`LocalSentenceTransformerProvider` or a cloud
    provider) for semantic search.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]],
        *,
        dimension: int | None = None,
    ) -> None:
        self._fn = embedding_fn
        self._dimension = dimension
        self._probed_dimension: int | None = dimension

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        if self._probed_dimension is not None:
            return self._probed_dimension
        raise RuntimeError(
            "FunctionEmbeddingProvider.dimension is unknown. Pass dimension= "
            "to the constructor or call embed() first."
        )

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        vectors = [list(self._fn(t)) for t in items]
        if self._probed_dimension is None and vectors:
            self._probed_dimension = len(vectors[0])
        return vectors

    async def aembed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        # An arbitrary callable has no native async API; reuse the sync path.
        return self.embed(texts)


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Real local embedding provider backed by ``sentence-transformers``.

    Requires the ``vertai[embeddings]`` extra. The model is loaded lazily on
    first use (loading is synchronous and may download weights on the first
    run). ``encode_kwargs`` are forwarded to ``SentenceTransformer.encode``.

    Example:
        from vertai.core.embedding import LocalSentenceTransformerProvider
        from vertai.core.vector import VectorEngine

        provider = LocalSentenceTransformerProvider("BAAI/bge-small-zh-v1.5")
        engine = VectorEngine(embedding_provider=provider)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs or {}
        self._model: Any = None
        self._dimension: int | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "LocalSentenceTransformerProvider requires the "
                    "'vertai[embeddings]' extra (sentence-transformers). "
                    "Install with: pip install 'vertai[embeddings]'"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
            self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._dimension

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        self._ensure_model()
        items = [texts] if isinstance(texts, str) else list(texts)
        vectors = self._model.encode(items, **self.encode_kwargs)
        return [list(v) for v in vectors]

    async def aembed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        # sentence-transformers is synchronous; offload the (potentially
        # blocking) encode to a thread so the event loop is not stalled.
        return await asyncio.to_thread(self.embed, texts)
