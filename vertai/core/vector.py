"""Vector store abstraction (S3 refactor).

Defines the :class:`VectorStore` ABC and three backends (``InMemory`` /
``Chroma`` / ``FAISS``) plus the :class:`VectorEngine` facade. Embeddings are
computed by an external :class:`~vertai.core.embedding.EmbeddingProvider` and
passed into :meth:`VectorStore.add`; the store never embeds text itself. This
separation fixes C2: when no provider is configured, :class:`VectorEngine`
raises explicitly instead of silently producing hash-random vectors, and
fixes C3: :meth:`FAISSVectorStore.delete` removes documents so ``count`` and
``search`` stay consistent.

The ``auto`` backend selection honors the documented priority
``ChromaDB > FAISS > InMemory`` (previously FAISS was never selected).
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vertai.core.embedding import (
    EmbeddingProvider,
    FunctionEmbeddingProvider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Document",
    "SearchResult",
    "VectorConfig",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "VectorEngine",
    # Backward-compat alias (deprecated; prefer FunctionEmbeddingProvider).
    "CustomEmbedding",
]


@dataclass
class Document:
    """Document stored in a :class:`VectorStore`.

    ``doc_id`` defaults to an md5 prefix of ``content`` so identical content
    deduplicates naturally.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class SearchResult:
    """A single retrieval result: the matched document plus similarity score."""

    document: Document
    score: float
    distance: float = 0.0


@dataclass
class VectorConfig:
    """Configuration for :class:`VectorEngine` and its backends."""

    collection_name: str = "default"
    persist_directory: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5


# Backward-compat alias. The old ``CustomEmbedding`` wrapped a callable; the new
# abstraction is :class:`EmbeddingProvider` and :class:`FunctionEmbeddingProvider`
# is the callable adapter. Kept so legacy imports keep resolving.
CustomEmbedding = FunctionEmbeddingProvider


class VectorStore(ABC):
    """Vector store abstraction. Stores documents + precomputed embeddings."""

    @abstractmethod
    def add(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        """Add ``documents`` with their precomputed ``embeddings``.

        Embeddings are computed externally by an
        :class:`~vertai.core.embedding.EmbeddingProvider`; the store never
        embeds text itself. The two lists must have equal length.
        """

    @abstractmethod
    def search(
        self, query_embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        """Search by a precomputed ``query_embedding`` vector."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by id. Real deletion: ``count`` and ``search`` must
        stay consistent afterwards (fixes C3)."""

    @abstractmethod
    def count(self) -> int:
        """Number of live documents."""


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity.

    Suitable for tests and small datasets. No external dependencies.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length "
                f"({len(documents)} vs {len(embeddings)})"
            )
        for doc, vec in zip(documents, embeddings):
            self._documents[doc.doc_id] = doc
            self._vectors[doc.doc_id] = list(vec)

    def search(
        self, query_embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        if not self._documents:
            return []
        scored: list[tuple[str, float, float]] = []
        for doc_id, doc in self._documents.items():
            score = self._cosine_similarity(query_embedding, self._vectors[doc_id])
            scored.append((doc_id, score, 1.0 - score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                document=self._documents[doc_id],
                score=score,
                distance=distance,
            )
            for doc_id, score, distance in scored[:top_k]
        ]

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
            self._vectors.pop(doc_id, None)

    def count(self) -> int:
        return len(self._documents)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a_sq += x * x
            norm_b_sq += y * y
        if norm_a_sq == 0.0 or norm_b_sq == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store. Supports persistence.

    Requires the ``chromadb`` package. Embeddings are supplied to
    :meth:`add` (the store does not embed text).
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

    @staticmethod
    def is_available() -> bool:
        """Check whether ChromaDB is importable."""
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return False
        return True

    def _init_chroma(self) -> None:
        if self._initialized:
            return
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "ChromaDB not installed. Install with: pip install chromadb"
            ) from e

        if self.persist_directory:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
        )
        self._initialized = True
        logger.info("ChromaDB initialized: %s", self.collection_name)

    def add(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        self._init_chroma()
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length "
                f"({len(documents)} vs {len(embeddings)})"
            )
        self._collection.add(
            ids=[doc.doc_id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
        )

    def search(
        self, query_embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        self._init_chroma()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_results: list[SearchResult] = []
        if results and results.get("ids"):
            ids_row = results["ids"][0]
            docs_row = results["documents"][0] if results.get("documents") else []
            metas_row = (
                results["metadatas"][0] if results.get("metadatas") else []
            )
            dists_row = (
                results["distances"][0] if results.get("distances") else []
            )
            for i, doc_id in enumerate(ids_row):
                content = docs_row[i] if i < len(docs_row) else ""
                metadata = metas_row[i] if i < len(metas_row) else {}
                distance = float(dists_row[i]) if i < len(dists_row) else 0.0
                score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0
                search_results.append(
                    SearchResult(
                        document=Document(
                            doc_id=doc_id, content=content, metadata=metadata
                        ),
                        score=score,
                        distance=distance,
                    )
                )
        return search_results

    def delete(self, ids: list[str]) -> None:
        self._init_chroma()
        if not ids:
            return
        self._collection.delete(ids=ids)

    def count(self) -> int:
        self._init_chroma()
        return int(self._collection.count())


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store (``IndexFlatIP``, inner product on normalized
    vectors).

    Requires the ``faiss-cpu`` (or ``faiss-gpu``) and ``numpy`` packages.

    FAISS ``IndexFlatIP`` does not support deletion; deleted documents are
    removed from the live document map and filtered out of search results, and
    :meth:`count` reports the live count (not ``index.ntotal``). This keeps
    ``count`` and ``search`` consistent after deletion (fixes C3). Stale vectors
    remain in the index and are compacted only by rebuilding — acceptable for
    1.0; a ``compact`` method can arrive in 1.x.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._index: Any = None
        self._documents: dict[int, Document] = {}
        self._id_to_idx: dict[str, int] = {}
        self._id_counter = 0
        self._initialized = False
        self._np: Any = None

    @staticmethod
    def is_available() -> bool:
        """Check whether both ``faiss`` and ``numpy`` are importable."""
        try:
            import faiss  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            return False
        return True

    def _init_faiss(self) -> None:
        if self._initialized:
            return
        try:
            import faiss
            import numpy as _np
        except ImportError as e:
            raise RuntimeError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            ) from e
        self._np = _np
        self._index = faiss.IndexFlatIP(self.dimension)
        self._initialized = True
        logger.info("FAISS initialized with dimension %d", self.dimension)

    def add(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        self._init_faiss()
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length "
                f"({len(documents)} vs {len(embeddings)})"
            )
        if not embeddings:
            return
        dim = len(embeddings[0])
        if dim != self.dimension:
            raise ValueError(
                f"embedding dimension {dim} does not match store dimension "
                f"{self.dimension}"
            )
        vectors = self._np.array(embeddings, dtype=self._np.float32)
        self._index.add(vectors)
        for doc in documents:
            idx = self._id_counter
            self._documents[idx] = doc
            self._id_to_idx[doc.doc_id] = idx
            self._id_counter += 1

    def search(
        self, query_embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]:
        self._init_faiss()
        if not self._documents:
            return []
        query_vector = self._np.array([query_embedding], dtype=self._np.float32)
        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_vector, k)
        results: list[SearchResult] = []
        scores_row = scores[0]
        indices_row = indices[0]
        for i, idx in enumerate(indices_row):
            idx_int = int(idx)
            # C3 fix: filter out deleted / stale indices.
            if idx_int in self._documents:
                doc = self._documents[idx_int]
                score = float(scores_row[i])
                results.append(
                    SearchResult(
                        document=doc, score=score, distance=1.0 - score
                    )
                )
        return results

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            idx = self._id_to_idx.pop(doc_id, None)
            if idx is not None:
                self._documents.pop(idx, None)
        # NOTE: stale vectors remain in the FAISS index but are filtered out of
        # search results above; count() reports only live documents.

    def count(self) -> int:
        # C3 fix: live documents, not index.ntotal (which still counts deleted).
        return len(self._documents)


class VectorEngine:
    """Facade composing an :class:`EmbeddingProvider` and a :class:`VectorStore`.

    Selects a backend by ``store_type`` (``memory`` / ``chroma`` / ``faiss`` /
    ``auto``). ``auto`` honors the documented priority ChromaDB > FAISS >
    InMemory (previously FAISS was never selected).

    C2 fix: there is no hash-random fallback. If no embedding provider is
    configured, :meth:`index_documents` and :meth:`search` raise explicitly
    rather than silently producing non-semantic vectors. Construct the engine
    freely (e.g. ``VectorEngine(store_type="memory")``) and inject a provider
    via ``embedding_provider=`` (preferred) or the legacy ``embedding_fn=``
    (wrapped in :class:`FunctionEmbeddingProvider`).

    Example:
        from vertai.core.embedding import LocalSentenceTransformerProvider
        from vertai.core.vector import VectorEngine, Document

        provider = LocalSentenceTransformerProvider()
        engine = VectorEngine(embedding_provider=provider, store_type="memory")
        engine.index_documents([Document(content="...")])
        results = engine.search("query")
    """

    def __init__(
        self,
        config: VectorConfig | None = None,
        store_type: str = "auto",
        embedding_provider: EmbeddingProvider | None = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        self.config = config or VectorConfig()
        if embedding_provider is not None:
            self._embedding: EmbeddingProvider | None = embedding_provider
        elif embedding_fn is not None:
            self._embedding = FunctionEmbeddingProvider(embedding_fn)
        else:
            # No default hash-random provider (C2 fix).
            self._embedding = None
        self._store: VectorStore | None = None
        self._store_type = store_type

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._create_store()
        return self._store

    @property
    def embedding(self) -> EmbeddingProvider | None:
        """The configured embedding provider, or ``None`` if none set."""
        return self._embedding

    def get_embedding(self) -> EmbeddingProvider:
        """Return the embedding provider or raise (C2 fix: no silent random)."""
        if self._embedding is None:
            raise RuntimeError(
                "No EmbeddingProvider configured. VectorEngine no longer falls "
                "back to random vectors (they produced non-semantic results). "
                "Inject one via VectorEngine(embedding_provider=...) or "
                "embedding_fn=..., or install 'vertai[embeddings]' and use "
                "LocalSentenceTransformerProvider."
            )
        return self._embedding

    def _create_store(self) -> VectorStore:
        store_type = self._store_type
        if store_type == "memory":
            return InMemoryVectorStore()
        if store_type == "chroma":
            if not ChromaVectorStore.is_available():
                raise RuntimeError(
                    "ChromaDB not installed. Install with: pip install chromadb"
                )
            return ChromaVectorStore(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
            )
        if store_type == "faiss":
            if not FAISSVectorStore.is_available():
                raise RuntimeError(
                    "FAISS not installed. Install with: pip install faiss-cpu"
                )
            return FAISSVectorStore(dimension=self._infer_dimension())

        # auto: ChromaDB > FAISS > InMemory (honest priority).
        if ChromaVectorStore.is_available():
            logger.info("auto: using ChromaDB as vector store")
            return ChromaVectorStore(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
            )
        if FAISSVectorStore.is_available():
            logger.info("auto: using FAISS as vector store")
            return FAISSVectorStore(dimension=self._infer_dimension())
        logger.info("auto: using in-memory vector store")
        return InMemoryVectorStore()

    def _infer_dimension(self) -> int:
        if self._embedding is not None:
            return self._embedding.dimension
        return 384

    def index_documents(self, documents: list[Document]) -> list[str]:
        """Embed (via the provider) and store ``documents``. Returns their ids."""
        if not documents:
            return []
        provider = self.get_embedding()
        embeddings = provider.embed([doc.content for doc in documents])
        self.store.add(documents, embeddings)
        return [doc.doc_id for doc in documents]

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Embed ``query`` and search the store."""
        provider = self.get_embedding()
        query_vector = provider.embed(query)[0]
        return self.store.search(query_vector, top_k=top_k or self.config.top_k)

    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id."""
        self.store.delete(ids)

    def count(self) -> int:
        """Live document count."""
        return self.store.count()

    def hybrid_search(
        self,
        query: str,
        keywords: list[str] | None = None,
        top_k: int | None = None,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        """Hybrid retrieval: vector similarity + keyword overlap fusion.

        ``alpha`` is the vector-score weight (``1 - alpha`` is the keyword
        weight). Without ``keywords`` this degenerates to plain vector search.
        """
        top_k = top_k or self.config.top_k
        vector_results = self.search(query, top_k=top_k * 2)
        if not keywords:
            return vector_results[:top_k]
        scored = self._compute_keyword_scores(vector_results, keywords, alpha)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _compute_keyword_scores(
        self,
        results: list[SearchResult],
        keywords: list[str],
        alpha: float,
    ) -> list[SearchResult]:
        keywords_lower = [kw.lower() for kw in keywords]
        scored: list[SearchResult] = []
        for result in results:
            content_lower = result.document.content.lower()
            keyword_score = sum(
                1 for kw in keywords_lower if kw in content_lower
            ) / len(keywords)
            final_score = alpha * result.score + (1.0 - alpha) * keyword_score
            scored.append(
                SearchResult(
                    document=result.document,
                    score=final_score,
                    distance=result.distance,
                )
            )
        return scored
