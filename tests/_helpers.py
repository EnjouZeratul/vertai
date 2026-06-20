"""Shared test helpers for the S3 test suite.

Importable as ``from tests._helpers import ...`` (the project root is placed on
``sys.path`` by ``conftest.py``). Provides:

- :class:`DeterministicEmbeddingProvider` — a deterministic, non-random
  embedding provider for unit tests that exercise retrieval/vector logic
  without the ``vertai[embeddings]`` extra. Disclosed as non-semantic: it
  hashes character n-grams into a fixed-width vector. Real semantic search
  requires :class:`LocalSentenceTransformerProvider` (``vertai[embeddings]``).
- Honest availability flags (``HAS_CHROMA`` / ``HAS_FAISS`` /
  ``HAS_SENTENCE_TRANSFORMERS``) and :func:`requires_extra` for integration
  tests that skip honestly when an optional extra is absent.
"""

from __future__ import annotations

import hashlib
from typing import Union

import pytest

from vertai.core.embedding import EmbeddingProvider


def _optional_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


#: Honest availability flags for optional extras.
HAS_CHROMA = _optional_import("chromadb")
HAS_FAISS = _optional_import("faiss") and _optional_import("numpy")
HAS_SENTENCE_TRANSFORMERS = _optional_import("sentence_transformers")


def requires_extra(name: str, available: bool) -> pytest.MarkDecorator:
    """Return a skip marker for an optional extra, honestly skipping when
    ``available`` is False."""
    reason = (
        f"{name} not installed; install the relevant extra to run this "
        "integration test"
    )
    return pytest.mark.skipif(not available, reason=reason)


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic, non-random embedding provider for unit tests.

    Maps character unigrams + bigrams of the text into a fixed-width vector via
    md5 hashing, then L2-normalizes. Deterministic (same input -> same vector)
    and provides some locality for shared n-grams. Disclosed as non-semantic:
    it exists to test retrieval mechanics, ranking, and store consistency, not
    embedding quality. Real semantic search requires
    :class:`LocalSentenceTransformerProvider` (``vertai[embeddings]``).
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        items = [texts] if isinstance(texts, str) else list(texts)
        return [self._embed_one(t) for t in items]

    async def aembed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        return self.embed(texts)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        if not text:
            return vec
        lowered = text.lower()
        tokens = [lowered] + [lowered[i:i + 2] for i in range(len(lowered))]
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dimension] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]
