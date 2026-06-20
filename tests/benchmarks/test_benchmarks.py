"""Performance benchmarks for critical paths.

These are not regression tests — they establish a performance baseline for the
hot paths (tool schema generation, vector search, agent loop overhead). Run with:

    pytest tests/benchmarks/ -v -s

They assert soft upper bounds so a 10x regression fails loudly, but normal
machine variance does not flake CI.
"""

from __future__ import annotations

import time

import pytest

from vertai import Document, VectorEngine
from vertai.core.embedding import FunctionEmbeddingProvider
from vertai.core.tool import tool, ToolRegistry


def _timed(fn, *, iterations: int) -> float:
    """Return median wall-clock seconds per call over `iterations` runs."""
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


# --- Tool schema generation -------------------------------------------------

def test_tool_schema_generation_is_fast() -> None:
    """@tool schema generation should be sub-millisecond (inspect + pydantic)."""

    @tool
    def lookup(customer_id: str, limit: int = 10) -> str:
        """Look up a customer.

        Args:
            customer_id: The customer identifier.
            limit: Max results.
        """
        return customer_id

    per_call = _timed(lambda: lookup.parameters, iterations=200)
    # Soft bound: schema build must be well under 5ms even on a slow CI runner.
    assert per_call < 0.005, f"tool schema generation too slow: {per_call*1000:.2f}ms/call"


# --- Vector search ----------------------------------------------------------

def test_inmemory_vector_search_baseline() -> None:
    """InMemoryVectorStore cosine search over 1k docs must be sub-50ms (top-5)."""

    def deterministic_embed(text: str) -> list[float]:
        vec = [0.0] * 64
        for ch in text:
            vec[ord(ch) % 64] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    engine = VectorEngine(embedding_provider=FunctionEmbeddingProvider(deterministic_embed, dimension=64))
    engine.index_documents([Document(doc_id=f"d{i}", content=f"document number {i}") for i in range(1000)])

    per_call = _timed(lambda: engine.search("document", top_k=5), iterations=50)
    assert per_call < 0.05, f"in-memory search too slow: {per_call*1000:.2f}ms/call"


# --- ToolRegistry dispatch --------------------------------------------------

def test_registry_spec_generation_is_fast() -> None:
    """ToolRegistry.to_specs() over 20 tools must be sub-millisecond."""

    @tool
    def t0(x: int) -> str:
        """t0."""
        return str(x)

    registry = ToolRegistry()
    for i in range(20):

        def make(idx: int):
            @tool(name=f"t{idx}")
            def _t(x: int) -> str:
                f"""t{idx}."""
                return str(x)
            return _t

        registry.register(make(i))

    per_call = _timed(lambda: registry.to_specs(), iterations=100)
    assert per_call < 0.005, f"registry to_specs too slow: {per_call*1000:.2f}ms/call"


if __name__ == "__main__":
    # Allow running directly for a quick baseline readout.
    pytest.main([__file__, "-v", "-s"])
