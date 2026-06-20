"""Tests for the EmbeddingProvider abstraction (S3).

Testing strategy (per ROADMAP test table):
- ``FunctionEmbeddingProvider`` / dimension probing -> real assertions (pure logic).
- ``LocalSentenceTransformerProvider`` -> ``@integration``, honestly skipped when
  ``vertai[embeddings]`` is not installed; when installed, runs against the real
  sentence-transformers library (no mocks).
"""

from __future__ import annotations

import asyncio

import pytest

from vertai.core.embedding import (
    EmbeddingProvider,
    FunctionEmbeddingProvider,
    LocalSentenceTransformerProvider,
)

from tests._helpers import HAS_SENTENCE_TRANSFORMERS, requires_extra


class TestEmbeddingProviderABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]


class TestFunctionEmbeddingProvider:
    def test_single_text_returns_one_vector(self) -> None:
        def fn(text: str) -> list[float]:
            return [float(len(text)), 1.0]

        provider = FunctionEmbeddingProvider(fn, dimension=2)
        vectors = provider.embed("hello")
        assert vectors == [[5.0, 1.0]]

    def test_batch_returns_one_vector_per_text(self) -> None:
        def fn(text: str) -> list[float]:
            return [float(len(text))]

        provider = FunctionEmbeddingProvider(fn, dimension=1)
        vectors = provider.embed(["a", "bb", "ccc"])
        assert vectors == [[1.0], [2.0], [3.0]]

    def test_dimension_from_constructor(self) -> None:
        provider = FunctionEmbeddingProvider(lambda t: [1.0, 2.0, 3.0], dimension=3)
        assert provider.dimension == 3

    def test_dimension_probed_after_embed(self) -> None:
        provider = FunctionEmbeddingProvider(lambda t: [0.0] * 5)
        # Before any embed, dimension is unknown -> raises honestly.
        with pytest.raises(RuntimeError, match="unknown"):
            _ = provider.dimension
        provider.embed("probe")
        assert provider.dimension == 5

    def test_deterministic_for_same_text(self) -> None:
        def fn(text: str) -> list[float]:
            return [float(ord(c)) for c in text]

        provider = FunctionEmbeddingProvider(fn, dimension=3)
        assert provider.embed("abc") == provider.embed("abc")

    def test_aembed_returns_same_as_embed(self) -> None:
        provider = FunctionEmbeddingProvider(lambda t: [1.0, 2.0], dimension=2)
        sync_result = provider.embed("x")
        async_result = asyncio.run(provider.aembed("x"))
        assert sync_result == async_result

    def test_embed_is_an_embedding_provider(self) -> None:
        provider = FunctionEmbeddingProvider(lambda t: [1.0], dimension=1)
        assert isinstance(provider, EmbeddingProvider)


@pytest.mark.integration
@requires_extra("sentence-transformers", HAS_SENTENCE_TRANSFORMERS)
class TestLocalSentenceTransformerProviderIntegration:
    """Real integration with sentence-transformers. Honestly skipped when the
    ``vertai[embeddings]`` extra is not installed."""

    def test_embed_returns_correct_dimension(self) -> None:
        provider = LocalSentenceTransformerProvider(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        vectors = provider.embed("hello world")
        assert len(vectors) == 1
        assert len(vectors[0]) == provider.dimension
        assert provider.dimension > 0

    def test_batch_embed(self) -> None:
        provider = LocalSentenceTransformerProvider(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        vectors = provider.embed(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) == provider.dimension for v in vectors)

    def test_aembed_real(self) -> None:
        provider = LocalSentenceTransformerProvider(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        vectors = asyncio.run(provider.aembed("hello"))
        assert len(vectors) == 1
        assert len(vectors[0]) == provider.dimension


class TestLocalSentenceTransformerProviderMissingExtra:
    """Verifies the helpful RuntimeError when the ``vertai[embeddings]`` extra is
    absent. Runs unconditionally (the absence is simulated via ``sys.modules``
    so it works whether or not the extra is installed)."""

    def test_dimension_raises_when_extra_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        # Make ``from sentence_transformers import ...`` raise ImportError.
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        provider = LocalSentenceTransformerProvider()
        with pytest.raises(RuntimeError, match=r"vertai\[embeddings\]"):
            _ = provider.dimension

    def test_embed_raises_when_extra_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        provider = LocalSentenceTransformerProvider()
        with pytest.raises(RuntimeError, match=r"vertai\[embeddings\]"):
            provider.embed("hello")
