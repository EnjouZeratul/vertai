"""Test configuration and shared fixtures.

Registers the ``integration`` marker and exposes a deterministic embedding
provider fixture. Shared helpers (the provider class, availability flags, skip
markers) live in :mod:`tests._helpers` so test modules can import them
directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to path so `vertai` and `tests._helpers` are importable.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests._helpers import (  # noqa: E402
    DeterministicEmbeddingProvider,
    HAS_CHROMA,
    HAS_FAISS,
    HAS_SENTENCE_TRANSFORMERS,
    requires_extra,
)

__all__ = [
    "DeterministicEmbeddingProvider",
    "HAS_CHROMA",
    "HAS_FAISS",
    "HAS_SENTENCE_TRANSFORMERS",
    "requires_extra",
]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end tests requiring optional extras "
        "(chromadb/faiss/sentence-transformers) or real services; "
        "skipped honestly when the dependency is unavailable.",
    )


@pytest.fixture
def deterministic_embedding() -> DeterministicEmbeddingProvider:
    """A deterministic embedding provider for retrieval/vector unit tests."""
    return DeterministicEmbeddingProvider(dimension=64)
