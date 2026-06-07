"""LLM 引擎单元测试"""

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from vertai.core.llm import (
    ChatMessage,
    GenerateResult,
    LLMConfig,
    LLMEngine,
    ModelInfo,
    ModelProvider,
    ModelStatus,
    OllamaDetector,
)


class TestLLMConfig:
    """LLMConfig 测试"""

    def test_default_config(self) -> None:
        """测试默认配置"""
        config = LLMConfig()
        assert config.model == "llama3.2"
        assert config.provider == ModelProvider.OLLAMA
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 120.0

    def test_custom_config(self) -> None:
        """测试自定义配置"""
        config = LLMConfig(
            model="mistral",
            temperature=0.5,
            max_tokens=1024,
            system_prompt="You are a helpful assistant.",
        )
        assert config.model == "mistral"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.system_prompt == "You are a helpful assistant."

    def test_temperature_bounds(self) -> None:
        """测试温度边界"""
        # 有效值
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

        # 无效值
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_bounds(self) -> None:
        """测试 max_tokens 边界"""
        config = LLMConfig(max_tokens=1)
        assert config.max_tokens == 1

        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)


class TestChatMessage:
    """ChatMessage 测试"""

    def test_create_message(self) -> None:
        """测试创建消息"""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_frozen(self) -> None:
        """测试消息不可变"""
        msg = ChatMessage(role="user", content="Hello")
        with pytest.raises(Exception):
            msg.content = "Modified"  # type: ignore


class TestGenerateResult:
    """GenerateResult 测试"""

    def test_create_result(self) -> None:
        """测试创建结果"""
        result = GenerateResult(
            content="Hello, world!",
            model="llama3.2",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert result.content == "Hello, world!"
        assert result.model == "llama3.2"
        assert result.total_tokens == 30

    def test_default_values(self) -> None:
        """测试默认值"""
        result = GenerateResult(content="Test", model="test")
        assert result.finish_reason == "stop"
        assert result.metadata == {}


class TestModelInfo:
    """ModelInfo 测试"""

    def test_create_model_info(self) -> None:
        """测试创建模型信息"""
        info = ModelInfo(
            name="llama3.2",
            provider=ModelProvider.OLLAMA,
            status=ModelStatus.AVAILABLE,
        )
        assert info.name == "llama3.2"
        assert info.provider == ModelProvider.OLLAMA
        assert info.status == ModelStatus.AVAILABLE


class TestOllamaDetector:
    """OllamaDetector 测试"""

    def test_is_running_success(self) -> None:
        """测试服务运行检测 - 成功"""
        with patch.object(httpx.Client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            detector = OllamaDetector()
            assert detector.is_running() is True

    def test_is_running_failure(self) -> None:
        """测试服务运行检测 - 失败"""
        with patch.object(httpx.Client, "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")

            detector = OllamaDetector()
            assert detector.is_running() is False

    def test_is_running_timeout(self) -> None:
        """测试服务运行检测 - 超时"""
        with patch.object(httpx.Client, "get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Timeout")

            detector = OllamaDetector()
            assert detector.is_running() is False

    def test_list_models(self) -> None:
        """测试获取模型列表"""
        mock_response_data = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "size": 2048,
                    "modified_at": "2024-01-01",
                    "details": {
                        "parameter_size": "3B",
                        "quantization_level": "Q4_0",
                    },
                },
                {
                    "name": "mistral:latest",
                    "size": 4096,
                    "modified_at": "2024-01-02",
                    "details": {},
                },
            ]
        }

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "get") as mock_get:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_get.return_value = mock_response

                detector = OllamaDetector()
                models = detector.list_models()

                assert len(models) == 2
                assert models[0].name == "llama3.2:latest"
                assert models[0].parameters == "3B"
                assert models[0].quantization == "Q4_0"

    def test_list_models_not_running(self) -> None:
        """测试服务未运行时获取模型列表"""
        with patch.object(OllamaDetector, "is_running", return_value=False):
            detector = OllamaDetector()
            models = detector.list_models()
            assert models == []

    def test_list_models_http_error(self) -> None:
        """测试获取模型列表 - HTTP 错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "get") as mock_get:
                mock_get.side_effect = httpx.HTTPError("HTTP error")

                detector = OllamaDetector()
                models = detector.list_models()
                assert models == []

    def test_list_models_json_decode_error(self) -> None:
        """测试获取模型列表 - JSON 解码错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "get") as mock_get:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
                mock_get.return_value = mock_response

                detector = OllamaDetector()
                models = detector.list_models()
                assert models == []

    def test_get_model_info_found(self) -> None:
        """测试获取模型信息 - 找到"""
        mock_models = [
            ModelInfo(name="llama3.2:latest", provider=ModelProvider.OLLAMA),
            ModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA),
        ]

        with patch.object(OllamaDetector, "list_models", return_value=mock_models):
            detector = OllamaDetector()
            info = detector.get_model_info("llama3.2")
            assert info is not None
            assert info.name == "llama3.2:latest"

    def test_get_model_info_not_found(self) -> None:
        """测试获取模型信息 - 未找到"""
        mock_models = [
            ModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA),
        ]

        with patch.object(OllamaDetector, "list_models", return_value=mock_models):
            detector = OllamaDetector()
            info = detector.get_model_info("llama3.2")
            assert info is None

    def test_pull_model_non_stream_success(self) -> None:
        """测试拉取模型 - 非流式成功"""
        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            detector = OllamaDetector()
            result = detector.pull_model("llama3.2", stream=False)
            assert result is True

    def test_pull_model_non_stream_failure(self) -> None:
        """测试拉取模型 - 非流式失败"""
        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            detector = OllamaDetector()
            result = detector.pull_model("llama3.2", stream=False)
            assert result is False

    def test_pull_model_non_stream_http_error(self) -> None:
        """测试拉取模型 - HTTP 错误"""
        with patch.object(httpx.Client, "post") as mock_post:
            mock_post.side_effect = httpx.HTTPError("Connection error")

            detector = OllamaDetector()
            result = detector.pull_model("llama3.2", stream=False)
            assert result is False

    def test_pull_model_stream(self) -> None:
        """测试拉取模型 - 流式"""
        mock_stream_data = [
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "complete"},
        ]

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            json.dumps(d) for d in mock_stream_data
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            detector = OllamaDetector()
            result = list(detector.pull_model("llama3.2", stream=True))
            assert len(result) == 3
            assert result[0]["status"] == "pulling manifest"
            assert result[1]["completed"] == 50

    def test_pull_model_stream_json_decode_error(self) -> None:
        """测试拉取模型 - 流式 JSON 解码错误"""
        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            '{"status": "pulling"}',
            'invalid json line',
            '{"status": "complete"}',
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            detector = OllamaDetector()
            result = list(detector.pull_model("llama3.2", stream=True))
            # Should skip invalid json lines
            assert len(result) == 2

    def test_ollama_detector_close(self) -> None:
        """测试关闭检测器"""
        with patch.object(httpx.Client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            detector = OllamaDetector()
            detector.is_running()  # Initialize client
            # 验证客户端已初始化
            assert detector._client is not None
            detector.close()
            # 验证客户端已关闭
            assert detector._client is None

    def test_ollama_detector_context_manager(self) -> None:
        """测试检测器上下文管理器"""
        with patch.object(httpx.Client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            with OllamaDetector() as detector:
                assert detector.is_running() is True


class TestLLMEngine:
    """LLMEngine 测试"""

    def test_init_default_config(self) -> None:
        """测试默认配置初始化"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            engine = LLMEngine()
            assert engine.current_model == "llama3.2"

    def test_init_custom_config(self) -> None:
        """测试自定义配置初始化"""
        config = LLMConfig(model="mistral", temperature=0.5)
        with patch.object(OllamaDetector, "is_running", return_value=True):
            engine = LLMEngine(config)
            assert engine.current_model == "mistral"

    def test_init_invalid_model_name(self) -> None:
        """测试初始化时模型名称包含非法字符"""
        # 包含特殊字符的模型名应被拒绝
        with pytest.raises(ValueError, match="非法字符"):
            LLMConfig(model="invalid/model")

        with pytest.raises(ValueError, match="非法字符"):
            LLMConfig(model="model with spaces")

        with pytest.raises(ValueError, match="非法字符"):
            LLMConfig(model="model<script>")
            engine = LLMEngine(config)
            assert engine.current_model == "mistral"

    def test_is_available(self) -> None:
        """测试服务可用性检查"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            engine = LLMEngine()
            assert engine.is_available is True

        with patch.object(OllamaDetector, "is_running", return_value=False):
            engine = LLMEngine()
            assert engine.is_available is False

    def test_list_models(self) -> None:
        """测试列出模型"""
        mock_models = [
            ModelInfo(name="llama3.2:latest", provider=ModelProvider.OLLAMA),
        ]
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(OllamaDetector, "list_models", return_value=mock_models):
                engine = LLMEngine()
                models = engine.list_models()
                assert len(models) == 1
                assert models[0].name == "llama3.2:latest"

    def test_switch_model_success(self) -> None:
        """测试切换模型 - 成功"""
        mock_models = [
            ModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA),
        ]

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(OllamaDetector, "get_model_info") as mock_info:
                mock_info.return_value = ModelInfo(
                    name="mistral:latest", provider=ModelProvider.OLLAMA
                )

                engine = LLMEngine()
                result = engine.switch_model("mistral")
                assert result is True
                assert engine.current_model == "mistral"

    def test_switch_model_not_found(self) -> None:
        """测试切换模型 - 模型不存在"""
        mock_models = [
            ModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA),
        ]

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(OllamaDetector, "get_model_info", return_value=None):
                with patch.object(LLMEngine, "list_models", return_value=mock_models):
                    engine = LLMEngine()
                    with pytest.raises(ValueError, match="不存在"):
                        engine.switch_model("nonexistent")

    def test_switch_model_no_available_models(self) -> None:
        """测试切换模型 - 无可用模型"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(OllamaDetector, "get_model_info", return_value=None):
                with patch.object(LLMEngine, "list_models", return_value=[]):
                    engine = LLMEngine()
                    with pytest.raises(ValueError, match="无可用模型"):
                        engine.switch_model("nonexistent")

    def test_switch_model_service_not_running(self) -> None:
        """测试切换模型 - 服务未运行"""
        with patch.object(OllamaDetector, "is_running", return_value=False):
            engine = LLMEngine()
            with pytest.raises(RuntimeError, match="未运行"):
                engine.switch_model("mistral")

    def test_switch_model_invalid_name(self) -> None:
        """测试切换模型 - 非法模型名称"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            engine = LLMEngine()
            with pytest.raises(ValueError, match="非法字符"):
                engine.switch_model("invalid/model")

    def test_generate_success(self) -> None:
        """测试生成文本 - 成功"""
        mock_response_data = {
            "response": "Hello! How can I help you?",
            "model": "llama3.2",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done": True,
        }

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                engine = LLMEngine()
                result = engine.generate("Hello")

                assert result.content == "Hello! How can I help you?"
                assert result.model == "llama3.2"
                assert result.prompt_tokens == 10
                assert result.completion_tokens == 20

    def test_generate_service_not_available(self) -> None:
        """测试生成文本 - 服务不可用"""
        with patch.object(OllamaDetector, "is_running", return_value=False):
            engine = LLMEngine()
            with pytest.raises(RuntimeError, match="未运行"):
                engine.generate("Hello")

    def test_generate_with_system_prompt_and_seed(self) -> None:
        """测试生成文本 - 带系统提示词和随机种子"""
        mock_response_data = {
            "response": "Response",
            "model": "llama3.2",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "done": True,
        }

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                config = LLMConfig(seed=42, system_prompt="Default system prompt")
                engine = LLMEngine(config)
                result = engine.generate("Hello", system_prompt="Custom system prompt")

                assert result.content == "Response"
                # Verify system_prompt and seed were included in payload
                call_args = mock_post.call_args
                assert call_args[1]["json"]["system"] == "Custom system prompt"
                assert call_args[1]["json"]["options"]["seed"] == 42

    def test_generate_connection_error(self) -> None:
        """测试生成文本 - 连接错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_post.side_effect = httpx.ConnectError("Connection failed")

                engine = LLMEngine()
                with pytest.raises(RuntimeError, match="无法连接"):
                    engine.generate("Hello")

    def test_generate_http_status_error(self) -> None:
        """测试生成文本 - HTTP 状态错误"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=mock_response
        )

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post", return_value=mock_response):
                engine = LLMEngine()
                with pytest.raises(RuntimeError, match="生成请求失败"):
                    engine.generate("Hello")

    def test_stream_success(self) -> None:
        """测试流式生成"""
        mock_stream_data = [
            {"response": "Hello"},
            {"response": "!"},
            {"response": " How"},
            {"response": " are"},
            {"response": " you?"},
        ]

        # 创建支持上下文管理器的 mock
        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            json.dumps(d) for d in mock_stream_data
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream", return_value=mock_stream):
                engine = LLMEngine()
                chunks = list(engine.stream("Hello"))
                assert chunks == ["Hello", "!", " How", " are", " you?"]

    def test_stream_connection_error(self) -> None:
        """测试流式生成 - 连接错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream") as mock_stream:
                mock_stream.side_effect = httpx.ConnectError("Connection failed")

                engine = LLMEngine()
                with pytest.raises(RuntimeError, match="无法连接"):
                    list(engine.stream("Hello"))

    def test_stream_json_decode_error(self) -> None:
        """测试流式生成 - JSON 解码错误"""
        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            '{"response": "valid"}',
            'invalid json',
            '{"response": " valid2"}',
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream", return_value=mock_stream):
                engine = LLMEngine()
                chunks = list(engine.stream("Hello"))
                # Should skip invalid json lines
                assert chunks == ["valid", " valid2"]

    def test_chat_success(self) -> None:
        """测试多轮对话"""
        mock_response_data = {
            "message": {"content": "I'm doing well, thanks!"},
            "model": "llama3.2",
            "prompt_eval_count": 20,
            "eval_count": 10,
            "done": True,
        }

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                engine = LLMEngine()
                messages = [
                    ChatMessage(role="user", content="Hello"),
                    ChatMessage(role="assistant", content="Hi there!"),
                    ChatMessage(role="user", content="How are you?"),
                ]
                result = engine.chat(messages)

                assert result.content == "I'm doing well, thanks!"

    def test_chat_http_status_error(self) -> None:
        """测试多轮对话 - HTTP 状态错误"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=Mock(), response=mock_response
        )

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post", return_value=mock_response):
                engine = LLMEngine()
                messages = [ChatMessage(role="user", content="Hello")]
                with pytest.raises(RuntimeError, match="对话请求失败"):
                    engine.chat(messages)

    def test_chat_connection_error(self) -> None:
        """测试多轮对话 - 连接错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_post.side_effect = httpx.ConnectError("Connection failed")

                engine = LLMEngine()
                messages = [ChatMessage(role="user", content="Hello")]
                with pytest.raises(RuntimeError, match="无法连接"):
                    engine.chat(messages)

    def test_chat_with_seed(self) -> None:
        """测试多轮对话 - 带随机种子"""
        mock_response_data = {
            "message": {"content": "Response"},
            "model": "llama3.2",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "done": True,
        }

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                config = LLMConfig(model="llama3.2", seed=42)
                engine = LLMEngine(config)
                messages = [ChatMessage(role="user", content="Hello")]
                result = engine.chat(messages)

                assert result.content == "Response"
                # Verify seed was included in payload
                call_args = mock_post.call_args
                assert call_args[1]["json"]["options"]["seed"] == 42

    def test_chat_stream_success(self) -> None:
        """测试流式对话"""
        mock_stream_data = [
            {"message": {"content": "I'm"}},
            {"message": {"content": " doing"}},
            {"message": {"content": " well"}},
            {"message": {"content": "!"}},
        ]

        # 创建支持上下文管理器的 mock
        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            json.dumps(d) for d in mock_stream_data
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream", return_value=mock_stream):
                engine = LLMEngine()
                messages = [
                    ChatMessage(role="user", content="How are you?"),
                ]
                chunks = list(engine.chat_stream(messages))
                assert chunks == ["I'm", " doing", " well", "!"]

    def test_chat_stream_connection_error(self) -> None:
        """测试流式对话 - 连接错误"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream") as mock_stream:
                mock_stream.side_effect = httpx.ConnectError("Connection failed")

                engine = LLMEngine()
                messages = [ChatMessage(role="user", content="Hello")]
                with pytest.raises(RuntimeError, match="无法连接"):
                    list(engine.chat_stream(messages))

    def test_chat_stream_json_decode_error(self) -> None:
        """测试流式对话 - JSON 解码错误"""
        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = [
            '{"message": {"content": "valid"}}',
            'invalid json',
            '{"message": {"content": " valid2"}}',
        ]
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "stream", return_value=mock_stream):
                engine = LLMEngine()
                messages = [ChatMessage(role="user", content="Hello")]
                chunks = list(engine.chat_stream(messages))
                # Should skip invalid json lines
                assert chunks == ["valid", " valid2"]

    def test_embeddings_success(self) -> None:
        """测试获取嵌入向量"""
        mock_response_data = {"embedding": [0.1, 0.2, 0.3]}

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                engine = LLMEngine()
                result = engine.embeddings("test text")

                assert len(result) == 1
                assert result[0] == [0.1, 0.2, 0.3]

    def test_embeddings_multiple_texts(self) -> None:
        """测试获取多个文本的嵌入向量"""
        mock_response_data = {"embedding": [0.1, 0.2, 0.3]}

        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_post.return_value = mock_response

                engine = LLMEngine()
                result = engine.embeddings(["text1", "text2"])

                assert len(result) == 2
                assert result[0] == [0.1, 0.2, 0.3]
                assert result[1] == [0.1, 0.2, 0.3]

    def test_context_manager(self) -> None:
        """测试上下文管理器"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with LLMEngine() as engine:
                assert engine.current_model == "llama3.2"

    def test_close(self) -> None:
        """测试关闭引擎"""
        with patch.object(OllamaDetector, "is_running", return_value=True):
            engine = LLMEngine()
            # Initialize the client by making a request
            with patch.object(httpx.Client, "post") as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = {
                    "response": "test",
                    "model": "llama3.2",
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "done": True,
                }
                mock_post.return_value = mock_response
                engine.generate("test")  # Initialize client
                # 验证客户端已初始化
                assert engine._client is not None
            engine.close()
            # 验证客户端已关闭
            assert engine._client is None


class TestLLMEngineIntegration:
    """LLMEngine 集成测试（需要真实 Ollama 服务）"""

    @pytest.mark.skip(reason="需要运行中的 Ollama 服务")
    def test_real_generate(self) -> None:
        """真实生成测试"""
        engine = LLMEngine()
        if not engine.is_available:
            pytest.skip("Ollama 服务未运行")

        result = engine.generate("Say 'Hello World' and nothing else.")
        assert len(result.content) > 0

    @pytest.mark.skip(reason="需要运行中的 Ollama 服务")
    def test_real_stream(self) -> None:
        """真实流式测试"""
        engine = LLMEngine()
        if not engine.is_available:
            pytest.skip("Ollama 服务未运行")

        chunks = list(engine.stream("Count from 1 to 5."))
        assert len(chunks) > 0

    @pytest.mark.skip(reason="需要运行中的 Ollama 服务")
    def test_real_chat(self) -> None:
        """真实对话测试"""
        engine = LLMEngine()
        if not engine.is_available:
            pytest.skip("Ollama 服务未运行")

        messages = [
            ChatMessage(role="user", content="What is 2+2?"),
        ]
        result = engine.chat(messages)
        assert len(result.content) > 0


class TestAnthropicAPI:
    """Anthropic 兼容 API 测试（使用 Mock）"""

    def test_config_anthropic_compatible(self) -> None:
        """测试 Anthropic 兼容检测"""
        # Ollama 不兼容
        config = LLMConfig(provider=ModelProvider.OLLAMA)
        assert not config.is_anthropic_compatible()

        # DeepSeek 兼容
        config = LLMConfig(provider=ModelProvider.DEEPSEEK, api_key="test")
        assert config.is_anthropic_compatible()

        # Anthropic 兼容
        config = LLMConfig(provider=ModelProvider.ANTHROPIC, api_key="test")
        assert config.is_anthropic_compatible()

    def test_config_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试从环境变量读取 API Key"""
        monkeypatch.setenv("VERTAI_API_KEY", "test-api-key")
        config = LLMConfig(provider=ModelProvider.DEEPSEEK)
        assert config.api_key == "test-api-key"

    def test_config_anthropic_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试从 ANTHROPIC_API_KEY 环境变量读取"""
        # 先清除 VERTAI_API_KEY（优先级更高）
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        config = LLMConfig(provider=ModelProvider.ANTHROPIC)
        assert config.api_key == "anthropic-key"

    def test_generate_anthropic_success(self) -> None:
        """测试 Anthropic API 生成成功"""
        mock_response_data = {
            "id": "msg-123",
            "type": "message",
            "model": "claude-3-sonnet",
            "content": [
                {"type": "text", "text": "Hello!"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            engine = LLMEngine(config)
            result = engine.generate("Hello")

            assert result.content == "Hello!"
            assert result.model == "claude-3-sonnet"
            assert result.prompt_tokens == 10
            assert result.completion_tokens == 5
            assert result.metadata["id"] == "msg-123"

    def test_generate_anthropic_with_system_prompt(self) -> None:
        """测试 Anthropic API 生成带系统提示词"""
        mock_response_data = {
            "id": "msg-123",
            "type": "message",
            "model": "claude-3-sonnet",
            "content": [
                {"type": "text", "text": "Response"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet",
            system_prompt="You are helpful.",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            engine = LLMEngine(config)
            # 使用配置中的 system_prompt
            result = engine.generate("Hello")

            # 验证请求 payload 包含 system 字段
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert "system" in payload
            assert payload["system"] == "You are helpful."
            assert result.content == "Response"

    def test_generate_anthropic_with_thinking(self) -> None:
        """测试 Anthropic API 生成包含 thinking 块"""
        mock_response_data = {
            "id": "msg-123",
            "type": "message",
            "model": "deepseek-v4",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="test-key",
            model="deepseek-v4",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            engine = LLMEngine(config)
            result = engine.generate("What is the answer?")

            assert result.content == "The answer is 42."
            assert result.metadata["thinking"] == "Let me think..."

    def test_generate_anthropic_http_error(self) -> None:
        """测试 Anthropic API HTTP 错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.side_effect = httpx.HTTPStatusError(
                "401", request=Mock(), response=mock_response
            )

            engine = LLMEngine(config)
            with pytest.raises(RuntimeError, match="401"):
                engine.generate("test")

    def test_generate_anthropic_connect_error(self) -> None:
        """测试 Anthropic API 连接错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            engine = LLMEngine(config)
            with pytest.raises(RuntimeError, match="无法连接"):
                engine.generate("test")

    def test_stream_anthropic_success(self) -> None:
        """测试 Anthropic API 流式生成"""
        mock_stream_lines = [
            'event: message_start',
            'data: {"type":"message_start"}',
            'event: content_block_start',
            'data: {"type":"content_block_start","content_block":{"type":"text"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"World"}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            chunks = list(engine.stream("test"))

            assert chunks == ["Hello", "World"]

    def test_stream_anthropic_skips_thinking(self) -> None:
        """测试 Anthropic API 流式跳过 thinking_delta"""
        mock_stream_lines = [
            'event: content_block_start',
            'data: {"type":"content_block_start","content_block":{"type":"thinking"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"..."}}',
            'event: content_block_start',
            'data: {"type":"content_block_start","content_block":{"type":"text"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Answer"}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            chunks = list(engine.stream("test"))

            # Should only have the text_delta, not thinking_delta
            assert chunks == ["Answer"]

    def test_stream_anthropic_connect_error(self) -> None:
        """测试 Anthropic API 流式连接错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.side_effect = httpx.ConnectError("Connection failed")
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            with pytest.raises(RuntimeError, match="无法连接"):
                list(engine.stream("test"))

    def test_chat_anthropic_success(self) -> None:
        """测试 Anthropic API 多轮对话成功"""
        mock_response_data = {
            "id": "msg-456",
            "type": "message",
            "model": "claude-3-sonnet",
            "content": [
                {"type": "text", "text": "Your name is Alice."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = mock_response_data
            mock_post.return_value = mock_response

            engine = LLMEngine(config)
            messages = [
                ChatMessage(role="user", content="My name is Alice"),
                ChatMessage(role="assistant", content="Hello Alice!"),
                ChatMessage(role="user", content="What's my name?"),
            ]
            result = engine.chat(messages)

            assert result.content == "Your name is Alice."
            assert result.model == "claude-3-sonnet"

    def test_chat_anthropic_http_error(self) -> None:
        """测试 Anthropic API 对话 HTTP 错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.side_effect = httpx.HTTPStatusError(
                "500", request=Mock(), response=mock_response
            )

            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="test")]
            with pytest.raises(RuntimeError, match="500"):
                engine.chat(messages)

    def test_chat_stream_anthropic_success(self) -> None:
        """测试 Anthropic API 流式对话成功"""
        mock_stream_lines = [
            'event: message_start',
            'data: {"type":"message_start"}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="Hello")]
            chunks = list(engine.chat_stream(messages))

            assert chunks == ["Hi"]

    def test_missing_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """测试缺少 API Key 时抛出错误"""
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = LLMConfig(provider=ModelProvider.ANTHROPIC)
        engine = LLMEngine(config)

        with pytest.raises(RuntimeError, match="API 密钥"):
            engine.generate("test")

    def test_build_anthropic_payload_with_custom_params(self) -> None:
        """测试构建 Anthropic payload 包含自定义参数"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet",
            temperature=0.5,
            top_p=0.8,
            top_k=10,
        )

        engine = LLMEngine(config)
        messages = [{"role": "user", "content": "test"}]
        payload = engine._build_anthropic_payload(messages, stream=False)

        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.8
        assert payload["top_k"] == 10

    def test_stream_anthropic_json_decode_error(self) -> None:
        """测试 Anthropic API 流式 JSON 解码错误"""
        mock_stream_lines = [
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            'data: invalid json line',  # This will cause JSONDecodeError
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"World"}}',
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            chunks = list(engine.stream("test"))

            # Should skip invalid json line
            assert chunks == ["Hello", "World"]

    def test_chat_anthropic_connect_error(self) -> None:
        """测试 Anthropic API 对话连接错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        with patch.object(httpx.Client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="test")]
            with pytest.raises(RuntimeError, match="无法连接"):
                engine.chat(messages)

    def test_chat_stream_anthropic_json_decode_error(self) -> None:
        """测试 Anthropic API 流式对话 JSON 解码错误"""
        mock_stream_lines = [
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'data: invalid json',  # This will cause JSONDecodeError
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="Hello")]
            chunks = list(engine.chat_stream(messages))

            # Should skip invalid json line
            assert chunks == ["Hi"]

    def test_chat_stream_anthropic_connect_error(self) -> None:
        """测试 Anthropic API 流式对话连接错误"""
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.side_effect = httpx.ConnectError("Connection failed")
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="Hello")]
            with pytest.raises(RuntimeError, match="无法连接"):
                list(engine.chat_stream(messages))

    def test_chat_stream_anthropic_with_content_block_start(self) -> None:
        """测试 Anthropic API 流式对话处理 content_block_start 事件"""
        mock_stream_lines = [
            'event: content_block_start',
            'data: {"type":"content_block_start","content_block":{"type":"text"}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]

        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key",
        )

        mock_stream = MagicMock()
        mock_stream.iter_lines.return_value = mock_stream_lines
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)

        with patch.object(httpx.Client, "stream", return_value=mock_stream):
            engine = LLMEngine(config)
            messages = [ChatMessage(role="user", content="Hello")]
            chunks = list(engine.chat_stream(messages))

            # content_block_start 事件被正确处理（continue）
            assert chunks == ["Hi"]

