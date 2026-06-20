# Changelog

All notable changes to VertAI are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — Architecture refactor (0.2.0+)

This is a large architectural refactor from "RAG toolkit" to "agent SDK". See
`docs/MIGRATION.md` for upgrade steps and `docs/ARCHITECTURE.md` for the new design.

### Added — Core agent abstractions (S2-S5)
- `LLMProvider` ABC + 4 adapters (Ollama / Anthropic / DeepSeek / OpenAI) with real
  async (`agenerate`/`astream`/`achat`) and native tool calling. `create_provider()`.
- `Tool` ABC + `@tool` decorator (inspect + docstring + pydantic schema, Field
  constraints, per-tool timeout, failure_error_function) + `ToolRegistry` — aligns with
  the OpenAI Agents SDK `@function_tool` standard.
- Built-in tools: `calculator` (safe AST, no `eval`), `file_read`/`file_write`
  (path-traversal-safe), `http_request`, `web_search`.
- `Agent` with a real tool-calling loop, `max_iterations` guard, `AgentResult`, and
  `Callbacks` (Protocol) for observability (`LoggingCallback`, `TokenCountCallback`).
- `EmbeddingProvider` ABC (`LocalSentenceTransformerProvider`, `FunctionEmbeddingProvider`),
  `Retriever` ABC (`VectorRetriever`), `TextSplitter` ABC (`RecursiveTextSplitter`,
  `FixedLengthSplitter`).

### Fixed — Critical bugs (C1-C5)
- `KnowledgeQA.ask()` no longer raises `TypeError` on the default LLM path.
- `VectorEngine` no longer silently uses `hash()`-seeded random vectors; it raises when
  no `EmbeddingProvider` is configured.
- `FAISSVectorStore.delete` now removes from the index; `count`/`search` are consistent.
- `ModelProvider.OPENAI` now uses the real OpenAI protocol (`/chat/completions` + Bearer).
- `DocGen` PDF type contract corrected (`generate` returns `str | bytes`, `save` branches).

### Security
- Indirect prompt-injection: `_sanitize_context` now detects and redacts injection
  (English + Chinese), instead of being cosmetic.
- `SessionMemory.session_id` whitelisted to `^[a-zA-Z0-9_-]+$` (path traversal).
- `KnowledgeQA.load_directory` no longer follows symlinks (Python 3.10-3.12).
- Atomic session persistence (`tmp + os.replace`); corrupt-file handling.

### Changed
- Real tokenizer estimation (tiktoken when available, else CJK-aware heuristic) replacing
  the `len//4` approximation that badly underestimated Chinese.
- `WorkflowContext` is now thread-safe (RLock) for parallel steps; `WorkflowConfig` /
  `ParallelConfig` timeouts are now actually enforced (were dead config).
- `Workflow` step/branch/parallel/loop now return `self` (fluent/chained API).
- `StructuredOutput` `"string"` type generalized (no longer hard-coded Chinese-name
  extraction); LLM-mode now validates against the schema with retries.
- Dashboard moved out of the core import into the optional `vertai[viz]` extra.
- `LocalModelManager` constructor is now side-effect-free (no `mkdir` on construction);
  fake/placeholder model URLs removed; `HF_ENDPOINT` mirror support actually works.
- Type safety: `mypy --strict` is now clean across the whole package (was 66 errors).

### Tooling
- GitHub Actions CI (ruff + mypy --strict + tests + coverage floor + build, Python
  3.10/3.11/3.12) and a tag-triggered Release workflow.
- Performance benchmarks for critical paths (`tests/benchmarks/`).
- `pdoc`-generated API reference.

### Removed
- Fake/placeholder data: `deepseek-v4-flash` model name (→ `deepseek-chat`), the
  fabricated "code reduction" comparison table, the "~5MB" / "fully offline" /
  "vertical-domain SDK" marketing claims that were not backed by the code.
- Coverage-chasing test classes (`TestRemainingCoverage`, `TestVectorStoreAbstract`,
  the line-number-driven `Test*Exceptions` in workflow) and tests that masked bugs via
  `except Exception: pass`.

### Known limitations (honestly declared, see ROADMAP "1.x")
No LLM response cache, no retry/backoff/circuit-breaker, no multi-agent / handoffs,
no human-in-the-loop, no multimodal input. These are tracked as 1.x post-1.0 work.

---

## [Unreleased]

### Architecture
- Defined target architecture and core abstraction contracts in `docs/ARCHITECTURE.md`
  (LLMProvider, EmbeddingProvider, VectorStore, Retriever, TextSplitter, Tool, Agent,
  Callbacks, Memory). Contracts first; implementation follows in S2-S9.

### Changed
- Project metadata honesty pass: version is now a single source
  (`dynamic = ["version"]` reads `vertai/__init__.py`); `Development Status` downgraded
  from `3 - Alpha` to `2 - Pre-Alpha` (reflects actual maturity); removed the
  `Typing :: Typed` classifier (mypy --strict is not yet clean; will be re-added in S10).
- README carries an "early development, not production-ready" warning.

### Known limitations (carried forward, honestly declared)
- `KnowledgeQA.ask()` raises `TypeError` on the default LLM path
  (`LLMEngine(model=...)` invalid signature). Fixed in S2/S3.
- Default `VectorEngine` uses `hash()`-seeded random vectors (not semantic); cross-process
  non-reproducible. Fixed in S3.
- `FAISSVectorStore.delete` does not remove from the index; `count`/`search` inconsistent.
  Fixed in S3.
- `ModelProvider.OPENAI` routes to the Anthropic protocol path and is broken against the
  real OpenAI API. Fixed in S2.
- 0 async APIs despite README claiming "streaming". Fixed in S2/S5.
- mypy --strict reports 66 errors; ruff reports 23. Cleaned up across S2-S9.
- Test suite: 94% line coverage is misleading — real I/O paths (LLM, model download,
  vector stores, embeddings) have ~0% effective coverage; several tests mask bugs with
  `except Exception: pass`. Fixed across S2-S9.

---

## [0.1.3] - 2026-06-07

### Notes
- Published to PyPI and GitHub.

### Known defects at release (declared honestly, in hindsight)
- See "Known limitations" above — these were present in 0.1.3 and surfaced by the
  post-release audit.
- Documentation overstated capabilities: "core ~5MB" (actual wheel ~60KB), "fully
  offline" (LLM requires external Ollama; default vectors are random), "vertical-domain
  SDK" (no vertical-specific code), fake model name `deepseek-v4-flash` (does not exist),
  fabricated code-reduction comparison table. Corrected in the unreleased honesty pass.

---

## [0.1.2] - 2026-06-07

### Changed
- Renamed module directory `ai_sdk` -> `vertai` so `import vertai` matches the package
  name after installation.
- Environment variables renamed `AI_SDK_*` -> `VERTAI_*`; cache path
  `.cache/ai_sdk/models` -> `.cache/vertai/models`.

### Known defects at release
- Same Critical-class bugs as 0.1.3 (declared above) were present.

---

## [0.1.1] - 2026-06-07

### Changed
- Fixed import path (`from vertai import ...`) to match the published package name.

### Known defects at release
- Same Critical-class bugs as 0.1.3 were present.

---

## [0.1.0] - 2026-06-07

### Added
- Initial release: Workflow, Dashboard, DocGen, DocParser (Markdown), SessionMemory,
  VectorEngine, StructuredOutput, LLMEngine, KnowledgeQA, Reviewer, LocalModelManager.

### Known defects at release
- Critical-class bugs (declared above) were present from this initial release.
- Test suite masked `KnowledgeQA.ask()` crash with `except Exception: pass`.
