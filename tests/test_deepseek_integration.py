"""DeepSeek API real integration tests (S2).

Executes against the real DeepSeek Anthropic-compatible API when
``VERTAI_API_KEY`` is set. Without the key the integration tests are skipped
(not silently faked). Run with:

    VERTAI_API_KEY=sk-xxx pytest tests/test_deepseek_integration.py -v

Note: the real DeepSeek chat model is ``deepseek-chat`` (the previous
``deepseek-v4-flash`` was a fabricated name).
"""

import os
import sys
import pytest
import httpx

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from vertai.core.llm import (
    LLMConfig,
    LLMEngine,
    ModelProvider,
    ChatMessage,
    GenerateResult,
)
from vertai.core.vector import VectorEngine, Document
from vertai.output.structured import StructuredOutput


TEST_API_KEY = os.environ.get("VERTAI_API_KEY")
TEST_BASE_URL = "https://api.deepseek.com/anthropic"
TEST_MODEL = "deepseek-chat"


requires_api_key = pytest.mark.skipif(
    not TEST_API_KEY,
    reason="Set VERTAI_API_KEY to run real DeepSeek integration tests",
)


@pytest.fixture
def deepseek_config():
    """DeepSeek 配置"""
    return LLMConfig(
        provider=ModelProvider.DEEPSEEK,
        base_url=TEST_BASE_URL,
        api_key=TEST_API_KEY,
        model=TEST_MODEL,
        max_tokens=300,  # 增加以确保有足够的 text 输出空间
    )


@pytest.fixture
def deepseek_engine(deepseek_config):
    """DeepSeek 引擎"""
    return LLMEngine(deepseek_config)


class TestDeepSeekIntegration:
    """DeepSeek API 真实集成测试"""

    @requires_api_key
    def test_generate_simple(self, deepseek_engine):
        """测试简单生成"""
        result = deepseek_engine.generate("你好，请用一句话介绍自己。")

        assert isinstance(result, GenerateResult)
        assert result.content
        assert len(result.content) > 0
        assert result.model == TEST_MODEL
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        print(f"\n生成结果: {result.content[:100]}...")
        print(f"Token 使用: 输入={result.prompt_tokens}, 输出={result.completion_tokens}")

    @requires_api_key
    def test_generate_with_system_prompt(self, deepseek_engine):
        """测试带系统提示词的生成"""
        result = deepseek_engine.generate(
            "你是谁？",
            system_prompt="你是一个友好的助手，请用简短的中文回答。",
        )

        assert result.content
        print(f"\n带系统提示词结果: {result.content[:100]}...")

    @requires_api_key
    def test_generate_with_parameters(self, deepseek_engine):
        """测试带参数的生成"""
        # DeepSeek 模型会先输出 thinking，需要足够的空间输出 text
        result = deepseek_engine.generate(
            "讲一个笑话",
            temperature=0.5,
            max_tokens=200,  # 增加以确保有 text 输出
        )

        # 检查结果 - 可能有 text 或只有 thinking
        assert result.model == TEST_MODEL
        print(f"\n带参数结果: {result.content[:100] if result.content else '(只有thinking)'}...")
        if result.metadata.get("thinking"):
            print(f"思考过程: {result.metadata['thinking'][:100]}...")

    @requires_api_key
    def test_stream_simple(self, deepseek_engine):
        """测试流式生成"""
        chunks = list(deepseek_engine.stream("数到5"))

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
        print(f"\n流式结果: {full_text[:100]}...")

    @requires_api_key
    def test_chat_single(self, deepseek_engine):
        """测试单轮对话"""
        messages = [
            ChatMessage(role="user", content="1+1等于几？"),
        ]

        result = deepseek_engine.chat(messages)

        assert result.content
        print(f"\n单轮对话结果: {result.content}")

    @requires_api_key
    def test_chat_multi_turn(self, deepseek_engine):
        """测试多轮对话"""
        # 更新 max_tokens 以确保有足够空间输出文本
        messages = [
            ChatMessage(role="user", content="我叫小明"),
            ChatMessage(role="assistant", content="你好小明！很高兴认识你。"),
            ChatMessage(role="user", content="我叫什么名字？"),
        ]

        result = deepseek_engine.chat(messages, max_tokens=200)

        # 检查结果 - 可能返回 text 或只有 thinking
        assert result.model == TEST_MODEL
        print(f"\n多轮对话结果: {result.content if result.content else '(只有thinking)'}")
        # 如果有 thinking，验证是否正确理解上下文
        if result.metadata.get("thinking"):
            thinking = result.metadata["thinking"]
            assert "小明" in thinking  # thinking 应该提到小明

    @requires_api_key
    def test_chat_stream(self, deepseek_engine):
        """测试流式对话"""
        messages = [
            ChatMessage(role="user", content="讲一个短笑话"),
        ]

        chunks = list(deepseek_engine.chat_stream(messages))

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0
        print(f"\n流式对话结果: {full_text}")


class TestConfigValidation:
    """配置验证测试"""

    def test_is_anthropic_compatible(self):
        """测试 Anthropic 兼容检测"""
        # Ollama 不兼容
        ollama_config = LLMConfig(provider=ModelProvider.OLLAMA)
        assert not ollama_config.is_anthropic_compatible()

        # DeepSeek 兼容
        deepseek_config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="test-key",
        )
        assert deepseek_config.is_anthropic_compatible()

        # Anthropic 兼容
        anthropic_config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )
        assert anthropic_config.is_anthropic_compatible()

    def test_missing_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch):
        """Missing API key must raise before any request is made."""
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(provider=ModelProvider.DEEPSEEK, api_key=None)
        engine = LLMEngine(config)
        with pytest.raises(RuntimeError, match="API key"):
            engine.generate("test")


class TestStructuredOutputWithLLM:
    """StructuredOutput LLM 模式真实测试"""

    @requires_api_key
    def test_extract_with_llm(self, deepseek_engine):
        """测试 LLM 语义提取"""
        schema = {"name": "string", "amount": "number"}
        output = StructuredOutput(schema, llm=deepseek_engine)

        result = output.extract("张三报销500元")

        assert result.success
        assert result.data["name"] == "张三"
        assert result.data["amount"] == 500
        print(f"\n提取结果: {result.data}")

    @requires_api_key
    def test_extract_complex_with_llm(self, deepseek_engine):
        """测试复杂结构 LLM 提取"""
        schema = {
            "product": "string",
            "price": "number",
            "quantity": "integer",
            "in_stock": "boolean",
        }
        output = StructuredOutput(schema, llm=deepseek_engine)

        result = output.extract("苹果手机，价格5999元，库存还有25台，有货")

        assert result.success
        assert "手机" in result.data.get("product", "")
        assert result.data.get("price") == 5999
        assert result.data.get("quantity") == 25
        assert result.data.get("in_stock") is True
        print(f"\n复杂提取结果: {result.data}")

    @requires_api_key
    def test_extract_enum_with_llm(self, deepseek_engine):
        """测试枚举类型 LLM 提取"""
        schema = {"status": "enum[pending,approved,rejected]", "reason": "string"}
        output = StructuredOutput(schema, llm=deepseek_engine)

        result = output.extract("申请已通过审核，原因是材料齐全")

        assert result.success
        assert result.data.get("status") == "approved"
        print(f"\n枚举提取结果: {result.data}")


class TestVectorEngineWithCustomEmbedding:
    """VectorEngine 自定义嵌入函数真实测试"""

    @requires_api_key
    def test_custom_embedding_with_llm(self, deepseek_engine):
        """测试使用自定义嵌入函数的向量引擎"""
        # 使用自定义嵌入函数（注意：随机向量不具备语义相似性）
        def embedding_fn(text: str) -> list[float]:
            import random
            random.seed(hash(text) % (2**32))
            return [random.gauss(0, 1) for _ in range(384)]

        engine = VectorEngine(store_type="memory", embedding_fn=embedding_fn)

        # 索引文档
        docs = [
            Document(content="Python是一种编程语言"),
            Document(content="机器学习是人工智能的子领域"),
            Document(content="今天天气很好"),
        ]
        ids = engine.index_documents(docs)

        assert len(ids) == 3
        assert engine.count() == 3

        # 搜索（随机向量不具备语义相似性，只验证返回结果）
        results = engine.search("编程语言")

        assert len(results) > 0
        # 验证搜索返回了文档
        assert all(r.document.content in ["Python是一种编程语言", "机器学习是人工智能的子领域", "今天天气很好"] for r in results)
        print(f"\n搜索返回 {len(results)} 个结果")

    @requires_api_key
    def test_custom_embedding_consistency(self, deepseek_engine):
        """测试自定义嵌入函数的一致性"""
        def embedding_fn(text: str) -> list[float]:
            import random
            random.seed(hash(text) % (2**32))
            return [random.gauss(0, 1) for _ in range(128)]

        # 相同文本应产生相同向量
        vec1 = embedding_fn("测试文本")
        vec2 = embedding_fn("测试文本")
        assert vec1 == vec2

        # 不同文本应产生不同向量
        vec3 = embedding_fn("不同文本")
        assert vec1 != vec3
        print("\n嵌入一致性验证通过")

    @requires_api_key
    def test_embedding_batch(self, deepseek_engine):
        """测试批量嵌入"""
        # 创建自定义嵌入函数
        def batch_embedding_fn(text: str) -> list[float]:
            import random
            random.seed(hash(text) % (2**32))
            return [random.gauss(0, 1) for _ in range(128)]

        from vertai.core.vector import CustomEmbedding
        embedding = CustomEmbedding(batch_embedding_fn)

        # 测试单个嵌入 (embed always returns list[list[float]])
        vec = embedding.embed("测试文本")
        assert len(vec) == 1
        assert len(vec[0]) == 128

        # 测试批量嵌入 (embed accepts a list of texts)
        texts = ["文本1", "文本2", "文本3"]
        vectors = embedding.embed(texts)
        assert len(vectors) == 3
        assert all(len(v) == 128 for v in vectors)
        print(f"\n批量嵌入: {len(vectors)} 个向量，每个 {len(vectors[0])} 维")


class TestLLMEmbeddings:
    """LLMEngine 嵌入 API 真实测试"""

    @requires_api_key
    def test_embeddings_single(self, deepseek_engine):
        """Single-text embeddings. DeepSeek does not expose an embeddings
        endpoint, so a 404 is a real (expected) response -> skip; any other
        error surfaces as a failure rather than being swallowed."""
        try:
            embeddings = deepseek_engine.embeddings("Hello World")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                pytest.skip("DeepSeek does not expose an embeddings API (404)")
            raise
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert all(isinstance(x, float) for x in embeddings[0])
        assert len(embeddings[0]) > 0
        print(f"\n嵌入向量维度: {len(embeddings[0])}")

    @requires_api_key
    def test_embeddings_batch(self, deepseek_engine):
        """Batch embeddings. Same 404-skip semantics as the single test."""
        texts = ["Python", "JavaScript", "Go"]
        try:
            embeddings = deepseek_engine.embeddings(texts)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                pytest.skip("DeepSeek does not expose an embeddings API (404)")
            raise
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(e) > 0 for e in embeddings)
        print(f"\n批量嵌入: {len(embeddings)} 个向量")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
