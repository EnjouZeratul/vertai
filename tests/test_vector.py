"""向量引擎模块测试 - 全量覆盖"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import importlib
import sys

from vertai.core.vector import (
    Document,
    EmbeddingEngine,
    InMemoryVectorStore,
    VectorConfig,
    VectorEngine,
    SearchResult,
)

# Import and check actual availability
from vertai.core.vector import ChromaVectorStore, FAISSVectorStore

# Use is_available() to check actual ChromaDB installation
CHROMA_AVAILABLE = ChromaVectorStore.is_available()

# FAISS availability depends on numpy AND faiss package
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import faiss
    FAISS_PKG_AVAILABLE = True
except ImportError:
    FAISS_PKG_AVAILABLE = False

FAISS_AVAILABLE = NUMPY_AVAILABLE and FAISS_PKG_AVAILABLE


class TestDocument:
    """Document 测试 - 全量覆盖"""

    def test_create_document(self):
        """测试创建文档"""
        doc = Document(content="测试内容", metadata={"source": "test"})
        assert doc.content == "测试内容"
        assert doc.metadata["source"] == "test"
        assert len(doc.doc_id) == 12

    def test_document_auto_id(self):
        """测试自动生成ID"""
        doc1 = Document(content="相同内容")
        doc2 = Document(content="相同内容")
        assert doc1.doc_id == doc2.doc_id

    def test_document_custom_id(self):
        """测试自定义ID"""
        doc = Document(content="内容", doc_id="custom_id")
        assert doc.doc_id == "custom_id"

    def test_document_empty_metadata(self):
        """测试空元数据"""
        doc = Document(content="内容")
        assert doc.metadata == {}


class TestEmbeddingEngine:
    """EmbeddingEngine 测试 - 全量覆盖"""

    def test_embed_returns_correct_dimension(self):
        """测试返回正确维度"""
        engine = EmbeddingEngine(dimension=384)
        vector = engine.embed("测试文本")
        assert len(vector) == 384

    def test_embed_normalization(self):
        """测试向量归一化"""
        engine = EmbeddingEngine()
        vector = engine.embed("测试文本")
        magnitude = sum(v * v for v in vector) ** 0.5
        assert abs(magnitude - 1.0) < 0.0001

    def test_embed_deterministic(self):
        """测试确定性嵌入"""
        engine = EmbeddingEngine()
        v1 = engine.embed("相同文本")
        v2 = engine.embed("相同文本")
        assert v1 == v2

    def test_embed_batch(self):
        """测试批量嵌入"""
        engine = EmbeddingEngine()
        vectors = engine.embed_batch(["文本1", "文本2", "文本3"])
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 384

    def test_embed_custom_dimension(self):
        """测试自定义维度"""
        engine = EmbeddingEngine(dimension=128)
        vector = engine.embed("测试")
        assert len(vector) == 128

    def test_embed_custom_model(self):
        """测试自定义模型名称"""
        engine = EmbeddingEngine(model="custom-model")
        # custom model 仍然使用 local 实现
        vector = engine.embed("测试")
        assert len(vector) == 384


class TestInMemoryVectorStore:
    """InMemoryVectorStore 测试 - 全量覆盖"""

    def test_add_documents(self):
        """测试添加文档"""
        store = InMemoryVectorStore()
        docs = [
            Document(content="文档1"),
            Document(content="文档2"),
        ]
        ids = store.add(docs)
        assert len(ids) == 2
        assert store.count() == 2

    def test_search_returns_results(self):
        """测试搜索返回结果"""
        store = InMemoryVectorStore()
        store.add([
            Document(content="Python是一种编程语言"),
            Document(content="JavaScript用于网页开发"),
            Document(content="机器学习是人工智能的分支"),
        ])

        results = store.search("编程语言", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r.document, Document) for r in results)

    def test_search_empty_store(self):
        """测试空存储搜索"""
        store = InMemoryVectorStore()
        results = store.search("查询", top_k=5)
        assert results == []

    def test_delete_documents(self):
        """测试删除文档"""
        store = InMemoryVectorStore()
        docs = [Document(content=f"文档{i}") for i in range(3)]
        ids = store.add(docs)

        assert store.count() == 3
        store.delete(ids[:2])
        assert store.count() == 1

    def test_delete_nonexistent_documents(self):
        """测试删除不存在的文档"""
        store = InMemoryVectorStore()
        store.add([Document(content="文档1")])
        # 删除不存在的ID不应报错
        store.delete(["nonexistent_id"])
        assert store.count() == 1

    def test_search_ranking(self):
        """测试搜索排序"""
        store = InMemoryVectorStore()
        store.add([
            Document(content="Python编程"),
            Document(content="美食烹饪"),
            Document(content="Python数据分析"),
        ])

        results = store.search("Python", top_k=3)
        assert any("Python" in r.document.content for r in results)

    def test_cosine_similarity_zero_vector(self):
        """测试零向量相似度"""
        # 测试 _cosine_similarity 处理零向量
        result = InMemoryVectorStore._cosine_similarity([0, 0, 0], [1, 2, 3])
        assert result == 0.0

    def test_search_with_distance(self):
        """测试搜索返回距离"""
        store = InMemoryVectorStore()
        store.add([Document(content="测试文档")])
        results = store.search("测试")
        assert len(results) == 1
        assert hasattr(results[0], 'distance')
        assert hasattr(results[0], 'score')

    def test_count_empty(self):
        """测试空存储计数"""
        store = InMemoryVectorStore()
        assert store.count() == 0


class TestVectorEngine:
    """VectorEngine 测试 - 全量覆盖"""

    def test_create_with_memory_store(self):
        """测试创建内存存储"""
        engine = VectorEngine(store_type="memory")
        assert engine.store is not None

    def test_index_and_search(self):
        """测试索引和搜索"""
        engine = VectorEngine(store_type="memory")
        docs = [
            Document(content="机器学习教程"),
            Document(content="深度学习框架"),
        ]

        ids = engine.index_documents(docs)
        assert len(ids) == 2

        results = engine.search("学习")
        assert len(results) > 0

    def test_count(self):
        """测试计数"""
        engine = VectorEngine(store_type="memory")
        assert engine.count() == 0

        engine.index_documents([
            Document(content="文档1"),
            Document(content="文档2"),
        ])
        assert engine.count() == 2

    def test_hybrid_search(self):
        """测试混合搜索"""
        engine = VectorEngine(store_type="memory")
        engine.index_documents([
            Document(content="Python编程入门教程"),
            Document(content="Java开发实战"),
            Document(content="Python数据分析"),
        ])

        results = engine.hybrid_search(
            query="Python",
            keywords=["Python", "教程"],
            top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_search_without_keywords(self):
        """测试无关键词的混合搜索"""
        engine = VectorEngine(store_type="memory")
        engine.index_documents([
            Document(content="Python编程"),
            Document(content="Java开发"),
        ])

        # 无关键词时应该退化为普通向量搜索
        results = engine.hybrid_search(
            query="Python",
            keywords=None,
            top_k=2,
        )
        assert len(results) <= 2

    def test_hybrid_search_with_alpha(self):
        """测试带alpha权重的混合搜索"""
        engine = VectorEngine(store_type="memory")
        engine.index_documents([
            Document(content="Python编程教程"),
        ])

        results = engine.hybrid_search(
            query="Python",
            keywords=["Python"],
            top_k=1,
            alpha=0.5,
        )
        assert len(results) == 1

    def test_config(self):
        """测试配置"""
        config = VectorConfig(
            embedding_model="local",
            collection_name="test",
            top_k=10,
        )
        engine = VectorEngine(config=config, store_type="memory")
        assert engine.config.top_k == 10

    def test_delete_documents(self):
        """测试删除文档"""
        engine = VectorEngine(store_type="memory")
        docs = [Document(content=f"文档{i}") for i in range(3)]
        ids = engine.index_documents(docs)

        assert engine.count() == 3
        engine.delete_documents(ids[:2])
        assert engine.count() == 1

    def test_auto_store_type_fallback(self):
        """测试auto模式自动降级"""
        # 当ChromaDB不可用时，应自动使用InMemory
        engine = VectorEngine(store_type="auto")
        assert isinstance(engine.store, InMemoryVectorStore)

    def test_chroma_store_type_unavailable(self):
        """测试chroma模式不可用时抛出异常"""
        if not CHROMA_AVAILABLE:
            engine = VectorEngine(store_type="chroma")
            with pytest.raises(RuntimeError, match="ChromaDB not installed"):
                _ = engine.store  # Trigger lazy loading

    def test_chroma_store_type_available(self):
        """测试chroma模式可用时创建ChromaVectorStore"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            engine = vector.VectorEngine(store_type="chroma")
            assert isinstance(engine.store, vector.ChromaVectorStore)

    def test_faiss_store_type(self):
        """测试faiss模式创建FAISSVectorStore"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            engine = vector.VectorEngine(store_type="faiss")
            assert isinstance(engine.store, vector.FAISSVectorStore)

    def test_auto_store_type_uses_chroma_when_available(self):
        """测试auto模式在ChromaDB可用时使用ChromaDB"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            engine = vector.VectorEngine(store_type="auto")
            assert isinstance(engine.store, vector.ChromaVectorStore)


class TestChromaVectorStore:
    """ChromaVectorStore 测试 - 全量覆盖（使用 mock）"""

    def test_chroma_is_available_true(self):
        """测试ChromaDB可用性检查 - 已安装"""
        mock_chromadb = MagicMock()
        with patch.dict('sys.modules', {'chromadb': mock_chromadb}):
            # 需要重新导入模块以应用 mock
            from vertai.core import vector
            importlib.reload(vector)
            assert vector.ChromaVectorStore.is_available() == True

    def test_chroma_is_available_false(self):
        """测试ChromaDB可用性检查 - 未安装"""
        with patch.dict('sys.modules', {'chromadb': None}):
            from vertai.core import vector
            importlib.reload(vector)
            assert vector.ChromaVectorStore.is_available() == False

    def test_chroma_init(self):
        """测试ChromaDB存储初始化"""
        mock_chromadb = MagicMock()
        with patch.dict('sys.modules', {'chromadb': mock_chromadb}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(
                collection_name="test_collection",
                persist_directory="/tmp/test",
            )
            assert store.collection_name == "test_collection"
            assert store.persist_directory == "/tmp/test"
            assert store._initialized == False

    def test_chroma_init_with_ephemeral_client(self):
        """测试ChromaDB EphemeralClient 初始化"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_ephemeral")
            store._init_chroma()

            mock_chromadb.EphemeralClient.assert_called_once()
            mock_client.get_or_create_collection.assert_called_once_with(name="test_ephemeral")
            assert store._initialized == True

    def test_chroma_init_with_persistent_client(self):
        """测试ChromaDB PersistentClient 初始化"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(
                collection_name="test_persistent",
                persist_directory="/tmp/chroma",
            )
            store._init_chroma()

            mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/chroma")
            assert store._initialized == True

    def test_chroma_init_already_initialized(self):
        """测试ChromaDB重复初始化"""
        mock_chromadb = MagicMock()
        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test")
            store._initialized = True
            store._init_chroma()
            # 不会再次调用 client
            mock_chromadb.EphemeralClient.assert_not_called()

    def test_chroma_init_import_error(self):
        """测试ChromaDB未安装时抛出异常"""
        with patch.dict('sys.modules', {'chromadb': None, 'chromadb.config': None}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test")
            with pytest.raises(RuntimeError, match="ChromaDB not installed"):
                store._init_chroma()

    def test_chroma_add_documents(self):
        """测试ChromaDB添加文档"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_add")
            docs = [
                vector.Document(content="文档1", metadata={"key": "value"}),
                vector.Document(content="文档2"),
            ]
            ids = store.add(docs)

            assert len(ids) == 2
            mock_collection.add.assert_called_once()

    def test_chroma_add_empty_documents(self):
        """测试ChromaDB添加空文档列表"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_empty")
            ids = store.add([])
            assert ids == []

    def test_chroma_search(self):
        """测试ChromaDB搜索"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        # 模拟搜索结果
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['内容1', '内容2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]],
        }

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_search")
            results = store.search("测试查询", top_k=2)

            assert len(results) == 2
            assert results[0].document.doc_id == 'doc1'
            assert results[0].document.content == '内容1'
            assert results[0].document.metadata == {'key': 'value1'}

    def test_chroma_search_empty_results(self):
        """测试ChromaDB搜索无结果"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_collection.query.return_value = {'ids': [[]]}

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_search_empty")
            results = store.search("测试", top_k=5)
            assert results == []

    def test_chroma_search_no_metadata(self):
        """测试ChromaDB搜索结果无元数据"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        # 当 metadatas 或 distances 为 None 时，代码使用默认值
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['内容1']],
            'metadatas': None,
            'distances': None,
        }

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_no_meta")
            results = store.search("测试", top_k=1)

            assert len(results) == 1
            assert results[0].document.metadata == {}
            assert results[0].distance == 0.0

    def test_chroma_delete(self):
        """测试ChromaDB删除文档"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_delete")
            result = store.delete(["doc1", "doc2"])

            assert result == True
            mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])

    def test_chroma_delete_empty_list(self):
        """测试ChromaDB删除空列表"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_delete_empty")
            result = store.delete([])

            assert result == True
            mock_collection.delete.assert_not_called()

    def test_chroma_count(self):
        """测试ChromaDB计数"""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.EphemeralClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 5

        with patch.dict('sys.modules', {'chromadb': mock_chromadb, 'chromadb.config': MagicMock()}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.ChromaVectorStore(collection_name="test_count")
            count = store.count()

            assert count == 5


class TestFAISSVectorStore:
    """FAISSVectorStore 测试 - 全量覆盖（使用 mock）"""

    def test_faiss_init(self):
        """测试FAISS存储初始化"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=128)

            assert store.dimension == 128
            assert store._initialized == False
            assert store._index is None

    def test_faiss_init_with_custom_embedding(self):
        """测试FAISS使用自定义嵌入引擎"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            embedding = vector.EmbeddingEngine(dimension=256)
            store = vector.FAISSVectorStore(embedding=embedding, dimension=256)
            assert store.embedding.dimension == 256

    def test_faiss_init_faiss(self):
        """测试FAISS初始化"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            store._init_faiss()

            mock_faiss.IndexFlatIP.assert_called_once_with(384)
            assert store._initialized == True
            assert store._index == mock_index

    def test_faiss_init_already_initialized(self):
        """测试FAISS重复初始化"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            store._initialized = True
            store._init_faiss()

            mock_faiss.IndexFlatIP.assert_not_called()

    def test_faiss_init_import_error(self):
        """测试FAISS未安装时抛出异常"""
        with patch.dict('sys.modules', {'faiss': None}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            with pytest.raises(RuntimeError, match="FAISS not installed"):
                store._init_faiss()

    def test_faiss_add_documents(self):
        """测试FAISS添加文档"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        # 模拟 numpy 数组
        mock_array = MagicMock()
        mock_numpy.array.return_value = mock_array
        mock_numpy.float32 = 'float32'

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)

            docs = [
                vector.Document(content="文档1"),
                vector.Document(content="文档2"),
            ]
            ids = store.add(docs)

            assert len(ids) == 2
            mock_index.add.assert_called_once()

    def test_faiss_search(self):
        """测试FAISS搜索"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        # 模拟搜索结果
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            [[0.9, 0.8]],  # scores
            [[0, 1]],      # indices
        )

        # 模拟 numpy 数组
        mock_array = MagicMock()
        mock_numpy.array.return_value = mock_array
        mock_numpy.float32 = 'float32'

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)

            # 调用 _init_faiss 来正确初始化 _np
            store._init_faiss()

            # 预先添加文档到 _documents
            store._documents = {
                0: vector.Document(content="文档1", doc_id="doc1"),
                1: vector.Document(content="文档2", doc_id="doc2"),
            }

            results = store.search("查询", top_k=2)

            assert len(results) == 2
            assert results[0].score == 0.9

    def test_faiss_search_empty(self):
        """测试FAISS空存储搜索"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_index.ntotal = 0

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            # 调用 _init_faiss 来正确初始化
            store._init_faiss()

            results = store.search("测试", top_k=5)
            assert results == []

    def test_faiss_search_with_index_limit(self):
        """测试FAISS搜索结果数量限制"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        mock_index.ntotal = 1
        mock_index.search.return_value = (
            [[0.9]],
            [[0]],
        )

        mock_array = MagicMock()
        mock_numpy.array.return_value = mock_array
        mock_numpy.float32 = 'float32'

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)

            # 调用 _init_faiss 来正确初始化 _np
            store._init_faiss()

            store._documents = {
                0: vector.Document(content="文档1", doc_id="doc1"),
            }

            results = store.search("查询", top_k=5)  # 请求5个，但只有1个

            # min(top_k, ntotal) 应该是 1
            assert len(results) == 1

    def test_faiss_search_invalid_index(self):
        """测试FAISS搜索结果索引越界"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        # 返回一个越界索引
        mock_index.ntotal = 1
        mock_index.search.return_value = (
            [[0.9]],
            [[99]],  # 越界索引
        )

        mock_array = MagicMock()
        mock_numpy.array.return_value = mock_array
        mock_numpy.float32 = 'float32'

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)

            # 调用 _init_faiss 来正确初始化 _np
            store._init_faiss()

            store._documents = {
                0: vector.Document(content="文档1", doc_id="doc1"),
            }

            results = store.search("查询", top_k=1)
            # 索引越界时不应该返回结果
            assert len(results) == 0

    def test_faiss_delete(self):
        """测试FAISS删除文档"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)

            # 添加文档
            doc1 = vector.Document(content="文档1", doc_id="doc1")
            doc2 = vector.Document(content="文档2", doc_id="doc2")
            doc3 = vector.Document(content="文档3", doc_id="doc3")
            store._documents = {0: doc1, 1: doc2, 2: doc3}

            result = store.delete(["doc1", "doc2"])
            assert result == True
            assert len(store._documents) == 1

    def test_faiss_count_initialized(self):
        """测试FAISS计数 - 已初始化"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_index.ntotal = 10

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            store._initialized = True
            store._index = mock_index

            assert store.count() == 10

    def test_faiss_count_not_initialized(self):
        """测试FAISS计数 - 未初始化"""
        mock_faiss = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict('sys.modules', {'faiss': mock_faiss, 'numpy': mock_numpy}):
            from vertai.core import vector
            importlib.reload(vector)
            store = vector.FAISSVectorStore(dimension=384)
            # 未初始化时返回 0
            assert store.count() == 0


class TestSearchResult:
    """SearchResult 测试"""

    def test_create_search_result(self):
        """测试创建搜索结果"""
        doc = Document(content="测试")
        result = SearchResult(document=doc, score=0.9, distance=0.1)
        assert result.document == doc
        assert result.score == 0.9
        assert result.distance == 0.1


class TestVectorConfig:
    """VectorConfig 测试 - 全量覆盖"""

    def test_default_config(self):
        """测试默认配置"""
        config = VectorConfig()
        assert config.embedding_model == "local"
        assert config.chunk_size == 512
        assert config.top_k == 5

    def test_custom_config(self):
        """测试自定义配置"""
        config = VectorConfig(
            embedding_model="custom",
            chunk_size=1024,
            top_k=10,
            collection_name="custom_collection",
            persist_directory="/tmp/vectors",
            chunk_overlap=100,
        )
        assert config.embedding_model == "custom"
        assert config.chunk_size == 1024
        assert config.top_k == 10
        assert config.collection_name == "custom_collection"
        assert config.persist_directory == "/tmp/vectors"
        assert config.chunk_overlap == 100


class TestVectorStoreAbstract:
    """VectorStore 抽象类测试"""

    def test_vector_store_abstract_methods(self):
        """测试 VectorStore 抽象方法"""
        from vertai.core.vector import VectorStore

        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            VectorStore()

        # 创建具体实现来覆盖抽象方法的 pass 语句
        class ConcreteVectorStore(VectorStore):
            def add(self, documents):
                # 调用父类的 pass 语句（行 131）
                super().add(documents)
                return []

            def search(self, query, top_k=5):
                # 调用父类的 pass 语句（行 144）
                super().search(query, top_k)
                return []

            def delete(self, doc_ids):
                # 调用父类的 pass 语句（行 156）
                super().delete(doc_ids)
                return True

            def count(self):
                # 调用父类的 pass 语句（行 161）
                super().count()
                return 0

        store = ConcreteVectorStore()
        # 测试抽象方法可以调用父类的 pass 实现
        assert store.add([]) == []
        assert store.search("test") == []
        assert store.delete([]) == True
        assert store.count() == 0


class TestNumpyNotAvailable:
    """numpy 不可用时的测试"""

    def test_numpy_not_available_branch(self):
        """测试 numpy 不可用时 NUMPY_AVAILABLE = False 分支"""
        # 使用 patch 来模拟 numpy 不可用
        with patch.dict('sys.modules', {'numpy': None}):
            # 强制重新加载模块以触发 ImportError 分支
            import vertai.core.vector as vector_mod
            importlib.reload(vector_mod)

            # 验证 NUMPY_AVAILABLE 为 False
            assert vector_mod.NUMPY_AVAILABLE == False
            # 验证 np 为 None
            assert vector_mod.np is None

            # 测试 FAISSVectorStore 在 numpy 不可用时的行为
            store = vector_mod.FAISSVectorStore(dimension=384)
            with pytest.raises(RuntimeError, match="FAISS not installed"):
                store._init_faiss()

            # 重新加载模块恢复状态
            from vertai.core import vector
            importlib.reload(vector)


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_document_list(self):
        """测试空文档列表"""
        store = InMemoryVectorStore()
        ids = store.add([])
        assert ids == []

    def test_single_character_content(self):
        """测试单字符内容"""
        engine = EmbeddingEngine()
        vector = engine.embed("a")
        assert len(vector) == 384

    def test_very_long_content(self):
        """测试超长内容"""
        long_text = "测试" * 10000
        engine = EmbeddingEngine()
        vector = engine.embed(long_text)
        assert len(vector) == 384

    def test_special_characters(self):
        """测试特殊字符"""
        engine = EmbeddingEngine()
        vector = engine.embed("!@#$%^&*()")
        assert len(vector) == 384

    def test_unicode_content(self):
        """测试Unicode内容"""
        engine = EmbeddingEngine()
        vector = engine.embed("你好世界🎉🌍")
        assert len(vector) == 384
