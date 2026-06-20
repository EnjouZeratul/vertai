# Deployment Guide

> Production deployment guidance for VertAI. Status: **Alpha** — core abstractions
> are implemented and tested, but this is **not yet certified production-ready**
> (see `docs/ROADMAP.md` S12 for the remaining 1.0 criteria).

## Installation

```bash
# Core (httpx + pydantic only)
pip install vertai

# With offline semantic search
pip install vertai[embeddings]

# With document parsing (PDF/Word/Excel/PPT)
pip install vertai[doc-parser]

# With dashboard visualization
pip install vertai[viz]

# Full production setup
pip install vertai[production]
```

## Choosing a backend

### LLM

- **Local / offline**: run [Ollama](https://ollama.ai) and use `LLMConfig()` defaults
  (`provider=OLLAMA`, `localhost:11434`). No API key, no network.
- **Cloud**: `LLMConfig(provider=..., base_url=..., api_key=...)`. API key via
  `VERTAI_API_KEY` env var (preferred) or explicit `api_key=`. Supported providers:
  Ollama, Anthropic, DeepSeek (Anthropic-compatible endpoint), OpenAI.

### Embeddings (for RAG / vector search)

- **Offline**: `LocalSentenceTransformerProvider("BAAI/bge-small-zh-v1.5")` (requires
  `vertai[embeddings]`). First load downloads weights; subsequent runs are offline.
- **Custom**: `FunctionEmbeddingProvider(your_fn, dimension=N)` for any callable.
- ⚠️ `VectorEngine` **raises** when no `EmbeddingProvider` is configured — it does not
  silently fall back to random vectors. This is intentional (random vectors have no
  semantic meaning).

## Secrets

- Never hardcode API keys. Use environment variables (`VERTAI_API_KEY`,
  `ANTHROPIC_API_KEY`) or your platform's secret manager.
- The SDK never logs API keys or full request bodies at INFO level.

## Security notes

- **Prompt injection**: `KnowledgeQA` sanitizes retrieved context (English + Chinese
  injection patterns are detected and redacted). This is defense-in-depth, **not** a
  guarantee — treat untrusted documents as untrusted.
- **Path traversal**: `SessionMemory` whitelists `session_id` to `^[a-zA-Z0-9_-]+$`.
  Do not pass untrusted user input as file paths.
- **Symlinks**: `KnowledgeQA.load_directory` does not follow symlinks (Python 3.10-3.12).
- **Tool execution**: the built-in `calculator` uses a safe AST evaluator (no `eval`).
  `file_read`/`file_write` enforce a `base_dir` boundary. Custom tools you register are
  your responsibility — validate inputs.

## Observability

- Use `Callbacks` (`LoggingCallback`, `TokenCountCallback`) to observe agent loops.
- For production tracing (OpenTelemetry / Langfuse-style), wrap your `LLMProvider` or
  implement `Callback` (full OTel integration is a 1.x goal).

## Performance

- In-memory vector search: ~sub-50ms for 1k documents, top-5 (see
  `tests/benchmarks/`).
- All LLM/embedding I/O is async-capable (`agenerate`/`aembed`/`aretrieve`/`arun`).
  Use async in servers for concurrency.

## What is NOT yet production-hardened

- No retry/backoff/circuit-breaker on LLM calls (planned 1.x).
- No response caching (planned 1.x).
- No multi-agent / human-in-the-loop (planned 1.x).
- API surface may still change before 1.0 (SemVer: 0.x allows breaking changes).
