# Migration Guide

This documents the breaking changes from the pre-alpha `vertai` (<=0.1.3) to the
refactored API (0.2.0+). If you have code written against the early releases, update
as follows.

## TL;DR

The early API was a flat toolkit (`LLMEngine` doing everything, random default
vectors, 0 async, fake model names). It has been refactored into explicit core
abstractions (`LLMProvider`, `EmbeddingProvider`, `Retriever`, `TextSplitter`,
`Tool`, `Agent`, `Callbacks`). `LLMEngine` remains as a thin facade for basic use.

## Breaking changes

### 1. Vector search no longer silently uses random vectors

**Before** (0.1.x): `VectorEngine()` with no embedding → random vectors, "search"
returned noise.

**After**: `VectorEngine()` with no embedding → `RuntimeError`. You must provide an
embedding.

```python
# After
from vertai.core.embedding import LocalSentenceTransformerProvider  # needs vertai[embeddings]
engine = VectorEngine(embedding_provider=LocalSentenceTransformerProvider("BAAI/bge-small-zh-v1.5"))

# Or a simple (non-semantic) function for wiring/testing:
from vertai.core.embedding import FunctionEmbeddingProvider
engine = VectorEngine(embedding_provider=FunctionEmbeddingProvider(my_fn, dimension=384))
```

### 2. DeepSeek model name

**Before**: examples used `model="deepseek-v4-flash"` (does not exist).

**After**: use the real name `model="deepseek-chat"`.

### 3. `chat()` accepts `ChatMessage` (dict still coerced)

**Before**: `chat(messages)` where messages could be `list[dict]`.

**After**: preferred `list[ChatMessage]`; `list[dict]` is still coerced for
back-compat. Prefer `ChatMessage(role="user", content="...")`.

### 4. `KnowledgeQA.ask()` no longer crashes

**Before**: `KnowledgeQA().ask(...)` raised `TypeError` on the default LLM path
(`LLMEngine(model="local")` was an invalid call).

**After**: inject a provider, or it raises a clear error instead of crashing.

```python
qa = KnowledgeQA(embedding_provider=provider, llm_provider=create_provider(LLMConfig(...)))
```

### 5. Dashboard moved to optional extra

**Before**: `from vertai import Dashboard`.

**After**: `pip install vertai[viz]` then `from vertai.viz.dashboard import Dashboard`.

### 6. Async APIs added (additive, non-breaking)

New: `provider.agenerate()`, `provider.astream()`, `retriever.aretrieve()`,
`agent.arun()`, `embedding.aembed()`. Sync APIs unchanged.

### 7. New agent capabilities (additive)

`Tool` / `@tool` / `ToolRegistry` / `Agent` / `Callbacks` are new in 0.2.0+. The SDK
is now a proper agent SDK, not just a RAG toolkit.

## Environment variables

`AI_SDK_*` → `VERTAI_*` (renamed in 0.1.2). Cache path moved from
`.cache/ai_sdk/models` to `.cache/vertai/models`.
