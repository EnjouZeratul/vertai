"""AI Agent SDK - 向量引擎模块

支持本地向量存储和检索，优先使用 ChromaDB，备选 FAISS。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

# numpy 是可选依赖，用于 FAISS 向量操作
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """文档数据结构"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class SearchResult:
    """检索结果"""

    document: Document
    score: float
    distance: float = 0.0


@dataclass
class VectorConfig:
    """向量引擎配置"""

    embedding_model: str = "local"
    collection_name: str = "default"
    persist_directory: str | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5


class EmbeddingEngine:
    """嵌入向量引擎

    支持本地模型和远程模型。
    默认使用模拟嵌入，生产环境可替换为真实模型。
    """

    def __init__(self, model: str = "local", dimension: int = 384):
        self.model = model
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """生成文本嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        if self.model == "local":
            return self._local_embed(text)
        return self._local_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        return [self.embed(text) for text in texts]

    def _local_embed(self, text: str) -> list[float]:
        """本地嵌入（模拟实现）

        WARNING: 此方法仅用于测试和开发，不适用于生产环境。
        随机种子基于文本哈希生成，相同文本产生相同向量，
        但语义相似性不保证向量相似。

        生产环境应替换为:
        - sentence-transformers
        - text-embedding-ada-002
        - 本地 Ollama 模型
        """
        random.seed(hash(text) % (2**32))
        vector = [random.gauss(0, 1) for _ in range(self.dimension)]
        magnitude = sum(v * v for v in vector) ** 0.5
        return [v / magnitude for v in vector]


class CustomEmbedding:
    """自定义嵌入函数包装器

    用于接入云端 API（如 OpenAI Embeddings、DeepSeek 等）。

    示例:
        from vertai import LLMEngine, LLMConfig, ModelProvider

        llm = LLMEngine(LLMConfig(
            provider=ModelProvider.OPENAI,
            api_key="sk-xxx",
        ))

        embedding = CustomEmbedding(llm.embeddings)
        vector = embedding.embed("Hello World")
    """

    def __init__(self, embedding_fn: Callable[[str], list[float]]):
        self._embedding_fn = embedding_fn

    def embed(self, text: str) -> list[float]:
        """生成嵌入向量"""
        return self._embedding_fn(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量"""
        return [self.embed(text) for text in texts]


class VectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add(self, documents: list[Document]) -> list[str]:
        """添加文档

        Args:
            documents: 文档列表

        Returns:
            文档ID列表
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> bool:
        """删除文档

        Args:
            doc_ids: 文档ID列表

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """获取文档数量"""
        pass


class InMemoryVectorStore(VectorStore):
    """内存向量存储

    简单实现，适合测试和小规模数据。
    使用余弦相似度进行检索。
    """

    def __init__(self, embedding: EmbeddingEngine | None = None):
        self.embedding = embedding or EmbeddingEngine()
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, documents: list[Document]) -> list[str]:
        doc_ids = []
        for doc in documents:
            self._documents[doc.doc_id] = doc
            self._vectors[doc.doc_id] = self.embedding.embed(doc.content)
            doc_ids.append(doc.doc_id)
        return doc_ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._documents:
            return []

        query_vector = self.embedding.embed(query)
        scores = []

        for doc_id, doc in self._documents.items():
            doc_vector = self._vectors[doc_id]
            score = self._cosine_similarity(query_vector, doc_vector)
            distance = 1 - score
            scores.append((doc_id, score, distance))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score, distance in scores[:top_k]:
            results.append(SearchResult(
                document=self._documents[doc_id],
                score=score,
                distance=distance,
            ))
        return results

    def delete(self, doc_ids: list[str]) -> bool:
        for doc_id in doc_ids:
            self._documents.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
        return True

    def count(self) -> int:
        return len(self._documents)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储

    生产级本地向量数据库，支持持久化。
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
        embedding: EmbeddingEngine | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding = embedding or EmbeddingEngine()
        self._client = None
        self._collection = None
        self._initialized = False

    @staticmethod
    def is_available() -> bool:
        """Check if ChromaDB is available."""
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False

    def _init_chroma(self):
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                )
            else:
                self._client = chromadb.EphemeralClient()

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
            )
            self._initialized = True
            logger.info(f"ChromaDB initialized: {self.collection_name}")

        except ImportError:
            raise RuntimeError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

    def add(self, documents: list[Document]) -> list[str]:
        self._init_chroma()

        if not documents:
            return []

        ids = [doc.doc_id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedding.embed_batch(contents)

        self._collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self._init_chroma()

        query_embedding = self.embedding.embed(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    doc_id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1 / (1 + distance)
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                ))

        return search_results

    def delete(self, doc_ids: list[str]) -> bool:
        self._init_chroma()

        if not doc_ids:
            return True

        self._collection.delete(ids=doc_ids)
        return True

    def count(self) -> int:
        self._init_chroma()
        return self._collection.count()


class FAISSVectorStore(VectorStore):
    """FAISS 向量存储

    Meta 开源的高效向量检索库，适合大规模数据。
    """

    def __init__(
        self,
        embedding: EmbeddingEngine | None = None,
        dimension: int = 384,
    ):
        self.embedding = embedding or EmbeddingEngine(dimension=dimension)
        self.dimension = dimension
        self._index = None
        self._documents: dict[int, Document] = {}
        self._id_counter = 0
        self._initialized = False
        self._np = None

    def _init_faiss(self):
        if self._initialized:
            return

        try:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
            self._np = np
            self._initialized = True
            logger.info(f"FAISS initialized with dimension {self.dimension}")

        except ImportError:
            raise RuntimeError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

    def add(self, documents: list[Document]) -> list[str]:
        self._init_faiss()

        doc_ids = []
        vectors = []

        for doc in documents:
            vector = self.embedding.embed(doc.content)
            vectors.append(vector)

            idx = self._id_counter
            self._documents[idx] = doc
            doc_ids.append(doc.doc_id)
            self._id_counter += 1

        vectors_np = self._np.array(vectors, dtype=self._np.float32)
        self._index.add(vectors_np)

        return doc_ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self._init_faiss()

        if self._index.ntotal == 0:
            return []

        query_vector = self._np.array(
            [self.embedding.embed(query)],
            dtype=self._np.float32,
        )

        scores, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self._documents):
                doc = self._documents[idx]
                score = float(scores[0][i])
                distance = 1 - score
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    distance=distance,
                ))

        return results

    def delete(self, doc_ids: list[str]) -> bool:
        idxs_to_remove = [
            idx for idx, doc in self._documents.items()
            if doc.doc_id in doc_ids
        ]
        for idx in idxs_to_remove:
            del self._documents[idx]
        return True

    def count(self) -> int:
        return self._index.ntotal if self._initialized else 0


class VectorEngine:
    """向量引擎

    统一的向量存储接口，自动选择最佳后端。
    优先级: ChromaDB > FAISS > InMemory

    支持自定义嵌入函数，可接入云端 API：

    示例:
        # 使用云端嵌入 API
        from vertai import LLMEngine, LLMConfig, ModelProvider

        llm = LLMEngine(LLMConfig(
            provider=ModelProvider.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="sk-xxx",
        ))

        # 方式1: 传入嵌入函数
        engine = VectorEngine(embedding_fn=llm.embeddings)

        # 方式2: 使用本地模拟（默认，仅测试用）
        engine = VectorEngine(store_type="memory")
    """

    def __init__(
        self,
        config: VectorConfig | None = None,
        store_type: str = "auto",
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        self.config = config or VectorConfig()

        # 支持自定义嵌入函数（如云端 API）
        if embedding_fn is not None:
            self.embedding = CustomEmbedding(embedding_fn)
        else:
            # 默认使用本地模拟嵌入（仅测试用）
            self.embedding = EmbeddingEngine(model=self.config.embedding_model)

        self._store: VectorStore | None = None
        self._store_type = store_type

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._create_store()
        return self._store

    def _create_store(self) -> VectorStore:
        if self._store_type == "memory":
            return InMemoryVectorStore(embedding=self.embedding)

        if self._store_type == "chroma":
            if not ChromaVectorStore.is_available():
                raise RuntimeError(
                    "ChromaDB not installed. Install with: pip install chromadb"
                )
            return ChromaVectorStore(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
                embedding=self.embedding,
            )

        if self._store_type == "faiss":
            return FAISSVectorStore(embedding=self.embedding)

        # auto: try ChromaDB first, fallback to memory
        if ChromaVectorStore.is_available():
            logger.info("Using ChromaDB as vector store")
            return ChromaVectorStore(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
                embedding=self.embedding,
            )
        else:
            logger.info("ChromaDB unavailable, using in-memory store")
            return InMemoryVectorStore(embedding=self.embedding)

    def index_documents(self, documents: list[Document]) -> list[str]:
        """索引文档

        Args:
            documents: 文档列表

        Returns:
            文档ID列表
        """
        return self.store.add(documents)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        return self.store.search(query, top_k or self.config.top_k)

    def delete_documents(self, doc_ids: list[str]) -> bool:
        """删除文档

        Args:
            doc_ids: 文档ID列表

        Returns:
            是否成功
        """
        return self.store.delete(doc_ids)

    def count(self) -> int:
        """获取文档数量"""
        return self.store.count()

    def hybrid_search(
        self,
        query: str,
        keywords: list[str] | None = None,
        top_k: int | None = None,
        alpha: float = 0.7,
    ) -> list[SearchResult]:
        """混合检索（向量 + 关键词）

        Args:
            query: 查询文本
            keywords: 关键词列表
            top_k: 返回数量
            alpha: 向量检索权重 (0-1)

        Returns:
            混合排序结果
        """
        top_k = top_k or self.config.top_k
        vector_results = self.search(query, top_k=top_k * 2)

        if not keywords:
            return vector_results[:top_k]

        scored_results = self._compute_keyword_scores(vector_results, keywords, alpha)
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def _compute_keyword_scores(
        self,
        results: list[SearchResult],
        keywords: list[str],
        alpha: float,
    ) -> list[SearchResult]:
        """计算关键词匹配分数并融合向量分数

        Args:
            results: 向量检索结果
            keywords: 关键词列表
            alpha: 向量分数权重

        Returns:
            融合分数后的结果列表
        """
        scored_results = []
        keywords_lower = [kw.lower() for kw in keywords]

        for result in results:
            content_lower = result.document.content.lower()
            keyword_score = sum(
                1 for kw in keywords_lower if kw in content_lower
            ) / len(keywords)

            final_score = alpha * result.score + (1 - alpha) * keyword_score
            scored_results.append(SearchResult(
                document=result.document,
                score=final_score,
                distance=result.distance,
            ))

        return scored_results
