"""Knowledge-base Q&A scenario (RAG) — S3 refactor.

Depends on the core abstractions :class:`~vertai.core.retriever.Retriever`,
:class:`~vertai.core.provider.LLMProvider`, and
:class:`~vertai.core.text_splitter.TextSplitter` rather than reaching into
concrete stores / the legacy ``LLMEngine`` single-prompt API. This fixes:

- **C1**: ``_get_llm`` used ``LLMEngine(model="local")`` which raised
  ``TypeError`` (``LLMEngine`` has no ``model`` kwarg); ``ask()`` then crashed.
  ``_generate_answer`` also passed a :class:`GenerateResult` where a ``str`` was
  expected. Both are fixed: the provider is obtained via
  :func:`~vertai.core.provider.create_provider` (or injection) and
  ``provider.generate([ChatMessage(...)])`` is used with ``result.content``.
- **C2**: there is no hash-random embedding fallback; an
  :class:`~vertai.core.embedding.EmbeddingProvider` must be injected.
- Indirect prompt injection via poisoned documents is neutralized in
  :func:`_sanitize_context` (English + Chinese patterns), and symlinked files /
  directories are refused by :class:`DocumentLoader` (3.10–3.12 safe).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from vertai.core.embedding import EmbeddingProvider
from vertai.core.llm import LLMEngine
from vertai.core.provider import (
    ChatMessage,
    LLMConfig,
    LLMProvider,
    create_provider,
)
from vertai.core.retriever import Retriever, VectorRetriever
from vertai.core.text_splitter import RecursiveTextSplitter, TextSplitter
from vertai.core.vector import (
    Document,
    SearchResult,
    VectorConfig,
    VectorEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """A cited source for an answer."""

    content: str
    source: str
    page: int | None = None
    paragraph: int | None = None
    relevance_score: float = 0.0


@dataclass
class AnswerResult:
    """Q&A result: the question, the generated answer, sources, confidence."""

    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# Prompt-injection guardrails. ``_sanitize_input`` rejects these in the user
# question; ``_sanitize_context`` redacts them in retrieved documents (indirect
# injection via poisoned docs). Patterns cover English and Chinese.
_MAX_QUESTION_LENGTH = 1000
_MAX_CONTEXT_LENGTH_SANITY = 10000

_DANGEROUS_PATTERNS = [
    # English
    r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now\s+a\b",
    r"new\s+instructions?\s*:",
    r"disregard\s+(all|previous|prior)",
    r"override\s+(previous|default|system)",
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    # Chinese
    r"忽略(之前|上面|前面|以上|上述)(的)?(指令|提示|规则|设定)",
    r"忘记(之前|上面|前面|所有)(的)?(指令|提示|内容)",
    r"你现在(扮演|是)(一个)?",
    r"请(忽略|忘记|无视)(之前|上面|前面|所有|上述)",
    r"无视(之前|上述|以上)(的)?(指令|规则)",
    r"覆盖(之前|原有|系统)(的)?(指令|设定|规则)",
    r"扮演(以下|这个|一个)?角色",
    r"输出(你的|系统)(提示|指令|规则)",
]
_DANGEROUS_RE = re.compile("|".join(_DANGEROUS_PATTERNS), re.IGNORECASE)
_REDACTION = "[redacted-injection]"


def _sanitize_input(text: str) -> str:
    """Sanitize user input; reject prompt-injection patterns and overlong input.

    Raises ``ValueError`` on injection patterns or length violation. Control
    characters are stripped.
    """
    if len(text) > _MAX_QUESTION_LENGTH:
        raise ValueError(
            f"input too long; maximum {_MAX_QUESTION_LENGTH} characters"
        )
    if _DANGEROUS_RE.search(text):
        raise ValueError("input contains disallowed content")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()


def _sanitize_context(text: str) -> str:
    """Sanitize retrieved context; redact indirect-injection patterns.

    Retrieved documents are not user-authored, but a poisoned document can still
    attempt prompt injection (indirect injection). Dangerous patterns are
    redacted (not rejected, since the surrounding text may be legitimate).
    Length is bounded; control characters are stripped.
    """
    if len(text) > _MAX_CONTEXT_LENGTH_SANITY:
        text = text[:_MAX_CONTEXT_LENGTH_SANITY]
    text = _DANGEROUS_RE.sub(_REDACTION, text)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned


def _get_env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value:
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _get_env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value:
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _get_env_list(key: str, default: list[str]) -> list[str]:
    value = os.environ.get(key)
    if value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


@dataclass
class KnowledgeQAConfig:
    """Knowledge-base Q&A configuration.

    Environment overrides:
        VERTAI_CHUNK_SIZE, VERTAI_CHUNK_OVERLAP, VERTAI_TOP_K,
        VERTAI_MIN_CONFIDENCE, VERTAI_MAX_CONTEXT_LENGTH,
        VERTAI_SUPPORTED_FORMATS (comma-separated).
    """

    chunk_size: int = field(default_factory=lambda: _get_env_int("VERTAI_CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _get_env_int("VERTAI_CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _get_env_int("VERTAI_TOP_K", 5))
    min_confidence: float = field(default_factory=lambda: _get_env_float("VERTAI_MIN_CONFIDENCE", 0.3))
    max_context_length: int = field(default_factory=lambda: _get_env_int("VERTAI_MAX_CONTEXT_LENGTH", 4000))
    supported_formats: list[str] = field(default_factory=lambda: _get_env_list("VERTAI_SUPPORTED_FORMATS", ["txt", "md", "json"]))


class DocumentLoader:
    """Document loader + chunker.

    Text files are split with an injectable :class:`TextSplitter` (default
    :class:`RecursiveTextSplitter`), replacing the previous hard-coded paragraph
    chunking. Symlinked files and directories are refused (``os.walk`` with
    ``followlinks=False`` plus explicit symlink checks — safe on 3.10–3.12).
    """

    def __init__(
        self,
        config: KnowledgeQAConfig | None = None,
        text_splitter: TextSplitter | None = None,
    ) -> None:
        self.config = config or KnowledgeQAConfig()
        self.text_splitter = text_splitter or RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Load all supported documents under ``directory``.

        Symlinked files and directories are skipped (defense against symlink
        poisoning / path traversal).
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"directory not found: {directory}")

        supported = {
            f".{ext.lower().lstrip('.')}" for ext in self.config.supported_formats
        }
        documents: list[Document] = []
        for root, dirs, files in os.walk(directory, followlinks=False):
            # Refuse to descend into symlinked directories.
            dirs[:] = [
                d for d in dirs if not (Path(root) / d).is_symlink()
            ]
            for name in files:
                file_path = Path(root) / name
                # Refuse symlinked files.
                if file_path.is_symlink():
                    logger.warning("skipping symlinked file: %s", file_path)
                    continue
                if file_path.suffix.lower() not in supported:
                    continue
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                    logger.info("loaded: %s", file_path)
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    logger.warning("failed to load %s: %s", file_path, e)
        return documents

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single file. Refuses symlinks."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        if file_path.is_symlink():
            raise ValueError(f"refusing to follow symlink: {file_path}")

        suffix = file_path.suffix.lower().lstrip(".")
        content = file_path.read_text(encoding="utf-8")
        if suffix == "json":
            return self._parse_json(content, str(file_path))
        chunks = self.text_splitter.split(content)
        return [
            Document(
                content=chunk,
                metadata={"source": str(file_path), "chunk_index": i},
            )
            for i, chunk in enumerate(chunks)
        ]

    def _parse_json(self, content: str, source: str) -> list[Document]:
        """Parse a JSON file (list of {content/text, ...} or a single object)."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed: {e}") from e

        documents: list[Document] = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = item.get("content") or item.get("text") or str(item)
                    metadata = {
                        k: v for k, v in item.items() if k not in ("content", "text")
                    }
                    metadata["source"] = source
                    metadata["index"] = i
                    documents.append(Document(content=text, metadata=metadata))
        elif isinstance(data, dict):
            text = data.get("content") or data.get("text") or str(data)
            metadata = {
                k: v for k, v in data.items() if k not in ("content", "text")
            }
            metadata["source"] = source
            documents.append(Document(content=text, metadata=metadata))
        return documents


class KnowledgeQA:
    """Knowledge-base Q&A (RAG).

    Retrieval goes through a :class:`Retriever` abstraction (default
    :class:`VectorRetriever` over a :class:`VectorEngine`); answer generation
    goes through an :class:`LLMProvider`. Both are injectable so the scenario
    depends on core abstractions, not concrete adapters.

    Example:
        from vertai.core.embedding import LocalSentenceTransformerProvider
        from vertai.core.provider import LLMConfig, create_provider

        qa = KnowledgeQA(
            docs_path="./docs",
            embedding_provider=LocalSentenceTransformerProvider(),
            provider=create_provider(LLMConfig()),
        )
        result = qa.ask("what is the reimbursement process?")
        print(result.answer)
    """

    def __init__(
        self,
        docs_path: str | Path | None = None,
        config: KnowledgeQAConfig | None = None,
        vector_config: VectorConfig | None = None,
        llm: LLMEngine | None = None,
        provider: LLMProvider | None = None,
        llm_config: LLMConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        text_splitter: TextSplitter | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.config = config or KnowledgeQAConfig()
        self.vector_config = vector_config or VectorConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            top_k=self.config.top_k,
        )
        self._vector_engine = VectorEngine(
            config=self.vector_config,
            store_type="memory",
            embedding_provider=embedding_provider,
            embedding_fn=embedding_fn,
        )
        self._provider: LLMProvider | None = provider or (
            llm.provider if isinstance(llm, LLMEngine) else None
        )
        self._llm_engine: LLMEngine | None = llm
        self._llm_config = llm_config
        self._retriever = retriever
        self._loader = DocumentLoader(self.config, text_splitter)
        self._indexed = False
        self._documents: list[Document] = []

        if docs_path:
            self.index(docs_path)

    # -- indexing ---------------------------------------------------------

    def index(self, docs_path: str | Path) -> int:
        """Index a document directory. Returns the number of chunks indexed."""
        docs_path = Path(docs_path)
        documents = self._loader.load_directory(docs_path)
        self._documents = documents
        if documents:
            self._vector_engine.index_documents(documents)
            self._indexed = True
            logger.info("indexed %d document chunks", len(documents))
        return len(documents)

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents directly. Returns the number added."""
        self._documents.extend(documents)
        self._vector_engine.index_documents(documents)
        self._indexed = True
        return len(documents)

    # -- retrieval --------------------------------------------------------

    def _get_retriever(self) -> Retriever:
        if self._retriever is not None:
            return self._retriever
        embedding = self._vector_engine.get_embedding()
        return VectorRetriever(embedding, self._vector_engine.store)

    # -- generation -------------------------------------------------------

    def _get_provider(self) -> LLMProvider:
        """Return the LLM provider (fixes C1: no ``LLMEngine(model=...)``)."""
        if self._provider is not None:
            return self._provider
        if self._llm_engine is not None:
            return self._llm_engine.provider
        config = self._llm_config or LLMConfig()
        return create_provider(config)

    def ask(self, question: str) -> AnswerResult:
        """Ask a question against the indexed knowledge base.

        Raises ``ValueError`` if the question is empty or contains disallowed
        content. Raises ``RuntimeError`` (from the vector engine) if no
        embedding provider is configured (C2: no silent random fallback).
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        question = _sanitize_input(question)

        if not self._indexed:
            return AnswerResult(
                question=question,
                answer="knowledge base has no indexed documents; add documents first.",
                confidence=0.0,
            )

        search_results = self._get_retriever().retrieve(
            question, top_k=self.config.top_k
        )
        if not search_results:
            return AnswerResult(
                question=question,
                answer="sorry, no relevant content was found.",
                confidence=0.0,
            )

        relevant = [
            r for r in search_results if r.score >= self.config.min_confidence
        ]
        if not relevant:
            return AnswerResult(
                question=question,
                answer="sorry, no sufficiently relevant content was found.",
                confidence=0.0,
            )

        context = self._build_context(relevant)
        answer = self._generate_answer(question, context)
        confidence = self._calculate_confidence(relevant)
        sources = self._build_sources(relevant)
        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                "context_length": len(context),
                "num_sources": len(sources),
            },
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        context_parts: list[str] = []
        total = 0
        for result in results:
            content = result.document.content
            if total + len(content) <= self.config.max_context_length:
                context_parts.append(content)
                total += len(content)
            else:
                remaining = self.config.max_context_length - total
                if remaining > 100:
                    context_parts.append(content[:remaining] + "...")
                break
        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer from the question + sanitized context.

        Uses the LLMProvider abstraction (``generate([ChatMessage])``) and reads
        ``result.content`` (fixes the GenerateResult/str mismatch that crashed
        ``ask()``).
        """
        provider = self._get_provider()
        safe_context = _sanitize_context(context)
        prompt = (
            "Answer the question based on the reference materials below. "
            "If the materials do not contain the answer, say so explicitly.\n\n"
            "---reference start---\n"
            f"{safe_context}\n"
            "---reference end---\n\n"
            "---question start---\n"
            f"{question}\n"
            "---question end---\n\n"
            "Give an accurate, concise answer based only on the reference "
            "materials, and cite the source:"
        )
        result = provider.generate([ChatMessage(role="user", content=prompt)])
        return self._parse_answer(result.content)

    def _parse_answer(self, response: str) -> str:
        """Parse an answer; unwrap a ``{"answer": ...}`` JSON envelope if present."""
        if response.startswith("{") and response.endswith("}"):
            try:
                data = json.loads(response)
                if isinstance(data, dict) and "answer" in data:
                    return str(data["answer"])
            except json.JSONDecodeError:
                pass
        return response.strip()

    def _calculate_confidence(self, results: list[SearchResult]) -> float:
        if not results:
            return 0.0
        scores = [r.score for r in results[:3]]
        avg = sum(scores) / len(scores)
        source_factor = min(len(results) / 3.0, 1.0)
        return round(avg * source_factor, 2)

    def _build_sources(self, results: list[SearchResult]) -> list[SourceReference]:
        sources: list[SourceReference] = []
        seen: set[str] = set()
        for result in results:
            doc = result.document
            source_key = f"{doc.metadata.get('source', 'unknown')}:{doc.doc_id}"
            if source_key in seen:
                continue
            seen.add(source_key)
            content = doc.content
            truncated = content[:200] + "..." if len(content) > 200 else content
            sources.append(
                SourceReference(
                    content=truncated,
                    source=str(doc.metadata.get("source", "unknown")),
                    page=doc.metadata.get("page") if isinstance(doc.metadata.get("page"), int) else None,
                    paragraph=doc.metadata.get("paragraph") if isinstance(doc.metadata.get("paragraph"), int) else None,
                    relevance_score=round(result.score, 3),
                )
            )
        return sources

    # -- maintenance ------------------------------------------------------

    def count_documents(self) -> int:
        """Number of indexed live documents."""
        return self._vector_engine.count()

    def clear(self) -> None:
        """Clear the knowledge base."""
        if self._documents:
            doc_ids = [doc.doc_id for doc in self._documents]
            self._vector_engine.delete_documents(doc_ids)
        self._documents = []
        self._indexed = False
