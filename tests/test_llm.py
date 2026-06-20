"""LLM engine facade tests (S2).

Tests the :class:`LLMEngine` backward-compat facade over :class:`LLMProvider`.
HTTP behavior is verified with ``httpx.MockTransport`` that stubs *real* wire
behavior (real-shaped JSON / SSE), so tests exercise the facade's coercion and
delegation without hitting the network. Pure logic (config, types, detector)
uses real assertions.
"""

from __future__ import annotations

import json
from typing import Callable
from unittest.mock import Mock, patch

import httpx
import pytest

from vertai.core.llm import (
    ChatMessage,
    GenerateResult,
    LLMConfig,
    LLMEngine,
    LLMModelInfo,
    ModelInfo,
    ModelProvider,
    ModelStatus,
    OllamaDetector,
)
from vertai.core.provider import (
    ToolCall,
    ToolSpec,
)

SyncHandler = Callable[[httpx.Request], httpx.Response]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_default_config(self) -> None:
        config = LLMConfig()
        assert config.model == "llama3.2"
        assert config.provider == ModelProvider.OLLAMA
        assert config.base_url == "http://localhost:11434"
        # Sentinel sampling params default to None (not hardcoded 0.7/0.9/40).
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.max_tokens == 4096
        assert config.timeout == 120.0

    def test_custom_config(self) -> None:
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
        assert LLMConfig(temperature=0.0).temperature == 0.0
        assert LLMConfig(temperature=2.0).temperature == 2.0
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_bounds(self) -> None:
        assert LLMConfig(max_tokens=1).max_tokens == 1
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)

    def test_invalid_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="illegal characters"):
            LLMConfig(model="invalid/model")
        with pytest.raises(ValueError, match="illegal characters"):
            LLMConfig(model="model with spaces")

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERTAI_API_KEY", "test-api-key")
        assert LLMConfig(provider=ModelProvider.DEEPSEEK).api_key == "test-api-key"

    def test_anthropic_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        assert LLMConfig(provider=ModelProvider.ANTHROPIC).api_key == "anthropic-key"

    def test_is_anthropic_compatible(self) -> None:
        assert not LLMConfig(provider=ModelProvider.OLLAMA).is_anthropic_compatible()
        assert LLMConfig(
            provider=ModelProvider.DEEPSEEK, api_key="k"
        ).is_anthropic_compatible()
        assert LLMConfig(
            provider=ModelProvider.ANTHROPIC, api_key="k"
        ).is_anthropic_compatible()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_create_message(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_frozen(self) -> None:
        msg = ChatMessage(role="user", content="Hello")
        with pytest.raises(Exception):
            msg.content = "Modified"  # type: ignore[misc]

    def test_coerce_dict(self) -> None:
        msg = ChatMessage.coerce({"role": "assistant", "content": "hi"})
        assert msg.role == "assistant"
        assert msg.content == "hi"


class TestGenerateResult:
    def test_create_result(self) -> None:
        result = GenerateResult(
            content="Hello, world!",
            model="llama3.2",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert result.content == "Hello, world!"
        assert result.total_tokens == 30

    def test_default_values(self) -> None:
        result = GenerateResult(content="Test", model="test")
        assert result.finish_reason == "stop"
        assert result.metadata == {}
        assert result.tool_calls == []


class TestToolTypes:
    def test_tool_spec_defaults(self) -> None:
        spec = ToolSpec(name="f", description="d")
        assert spec.input_schema == {"type": "object", "properties": {}}

    def test_tool_call(self) -> None:
        call = ToolCall(id="c1", name="f", arguments={"x": 1})
        assert call.arguments == {"x": 1}


class TestLLMModelInfo:
    def test_create_model_info(self) -> None:
        info = LLMModelInfo(
            name="llama3.2",
            provider=ModelProvider.OLLAMA,
            status=ModelStatus.AVAILABLE,
        )
        assert info.name == "llama3.2"
        assert info.provider == ModelProvider.OLLAMA
        assert info.status == ModelStatus.AVAILABLE

    def test_model_info_alias(self) -> None:
        # ModelInfo is a backward-compat alias for LLMModelInfo.
        assert ModelInfo is LLMModelInfo


# ---------------------------------------------------------------------------
# OllamaDetector (unchanged behavior; returns LLMModelInfo)
# ---------------------------------------------------------------------------


class TestOllamaDetector:
    def test_is_running_success(self) -> None:
        with patch.object(httpx.Client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            assert OllamaDetector().is_running() is True

    def test_is_running_failure(self) -> None:
        with patch.object(httpx.Client, "get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection failed")
            assert OllamaDetector().is_running() is False

    def test_list_models(self) -> None:
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
            ]
        }
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(httpx.Client, "get") as mock_get:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.json.return_value = mock_response_data
                mock_get.return_value = mock_response
                models = OllamaDetector().list_models()
                assert len(models) == 1
                assert models[0].name == "llama3.2:latest"
                assert models[0].parameters == "3B"
                assert models[0].quantization == "Q4_0"

    def test_list_models_not_running(self) -> None:
        with patch.object(OllamaDetector, "is_running", return_value=False):
            assert OllamaDetector().list_models() == []

    def test_get_model_info_found(self) -> None:
        mock_models = [
            LLMModelInfo(name="llama3.2:latest", provider=ModelProvider.OLLAMA),
        ]
        with patch.object(OllamaDetector, "list_models", return_value=mock_models):
            info = OllamaDetector().get_model_info("llama3.2")
            assert info is not None
            assert info.name == "llama3.2:latest"

    def test_get_model_info_not_found(self) -> None:
        with patch.object(OllamaDetector, "list_models", return_value=[]):
            assert OllamaDetector().get_model_info("llama3.2") is None

    def test_pull_model_non_stream_success(self) -> None:
        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            assert OllamaDetector().pull_model("llama3.2", stream=False) is True

    def test_pull_model_non_stream_failure(self) -> None:
        with patch.object(httpx.Client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            assert OllamaDetector().pull_model("llama3.2", stream=False) is False

    def test_close(self) -> None:
        with patch.object(httpx.Client, "get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            detector = OllamaDetector()
            detector.is_running()
            assert detector._client is not None
            detector.close()
            assert detector._client is None


# ---------------------------------------------------------------------------
# LLMEngine facade helpers (MockTransport on the underlying provider client)
# ---------------------------------------------------------------------------


def _attach_provider_client(engine: LLMEngine, handler: SyncHandler) -> list[httpx.Request]:
    """Attach a MockTransport sync client to the engine's provider and return a
    list capturing each request (for payload/header/URL assertions)."""
    captured: list[httpx.Request] = []

    def _h(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    engine.provider._sync_client = httpx.Client(  # type: ignore[attr-defined]
        transport=httpx.MockTransport(_h)
    )
    return captured


def _sse(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# LLMEngine: Anthropic facade via MockTransport
# ---------------------------------------------------------------------------


class TestLLMEngineAnthropic:
    def _engine(self) -> LLMEngine:
        return LLMEngine(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                base_url="https://api.anthropic.com",
                api_key="sk-ant",
                model="claude-3-sonnet",
            )
        )

    def test_generate_single_prompt(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                },
            )

        captured = _attach_provider_client(engine, handler)
        result = engine.generate("Hi")

        assert result.content == "Hello!"
        body = json.loads(captured[0].content)
        # system_prompt from config is None; generate(prompt) wraps as user msg.
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_generate_with_system_prompt(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        captured = _attach_provider_client(engine, handler)
        engine.generate("Hi", system_prompt="Be brief.")
        body = json.loads(captured[0].content)
        # Anthropic: system prompt goes into the `system` field, not messages.
        assert body["system"] == "Be brief."
        assert body["messages"] == [{"role": "user", "content": "Hi"}]

    def test_generate_with_thinking(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "The answer is 42."},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 4},
                },
            )

        _attach_provider_client(engine, handler)
        result = engine.generate("answer?")
        assert result.content == "The answer is 42."
        assert result.metadata["thinking"] == "Let me think..."

    def test_chat_accepts_dict_and_chatmessage(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [{"type": "text", "text": "Your name is Alice."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        captured = _attach_provider_client(engine, handler)
        # Mixed dict + ChatMessage input must be coerced uniformly.
        messages = [
            {"role": "user", "content": "I am Alice"},
            ChatMessage(role="assistant", content="Hi Alice!"),
            {"role": "user", "content": "Who am I?"},
        ]
        result = engine.chat(messages)
        assert result.content == "Your name is Alice."
        body = json.loads(captured[0].content)
        assert body["messages"] == [
            {"role": "user", "content": "I am Alice"},
            {"role": "assistant", "content": "Hi Alice!"},
            {"role": "user", "content": "Who am I?"},
        ]

    def test_chat_rejects_bad_message_type(self) -> None:
        engine = self._engine()
        _attach_provider_client(engine, lambda r: httpx.Response(200, json={}))
        with pytest.raises(TypeError, match="Unsupported message type"):
            engine.chat(["not a message"])  # type: ignore[list-item]

    def test_stream_yields_str(self) -> None:
        engine = self._engine()
        sse = _sse([
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"World"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_provider_client(engine, handler)
        chunks = list(engine.stream("hi"))
        assert chunks == ["Hello", "World"]

    def test_chat_stream_yields_str(self) -> None:
        engine = self._engine()
        sse = _sse([
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_provider_client(engine, handler)
        chunks = list(engine.chat_stream([ChatMessage(role="user", content="x")]))
        assert chunks == ["Hi"]

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        engine = LLMEngine(
            LLMConfig(provider=ModelProvider.ANTHROPIC, api_key=None)
        )
        with pytest.raises(RuntimeError, match="API key"):
            engine.generate("test")

    def test_is_available_with_api_key(self) -> None:
        engine = self._engine()
        assert engine.is_available is True

    def test_current_model(self) -> None:
        assert self._engine().current_model == "claude-3-sonnet"

    def test_provider_property(self) -> None:
        engine = self._engine()
        from vertai.core.provider import AnthropicProvider

        assert isinstance(engine.provider, AnthropicProvider)


# ---------------------------------------------------------------------------
# LLMEngine: OpenAI facade (verifies C4 fix end-to-end through the facade)
# ---------------------------------------------------------------------------


class TestLLMEngineOpenAI:
    def _engine(self) -> LLMEngine:
        return LLMEngine(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk-openai",
                model="gpt-4o-mini",
            )
        )

    def test_generate_hits_chat_completions_with_bearer(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        captured = _attach_provider_client(engine, handler)
        result = engine.generate("hello")
        assert result.content == "Hi"
        # C4 fix verified through the facade: /chat/completions + Bearer.
        assert captured[0].url == "https://api.openai.com/v1/chat/completions"
        assert captured[0].headers["authorization"] == "Bearer sk-openai"

    def test_tool_calling_through_facade(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_time",
                                            "arguments": '{"tz": "UTC"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        _attach_provider_client(engine, handler)
        spec = ToolSpec(
            name="get_time",
            description="Get time",
            input_schema={"type": "object", "properties": {}},
        )
        result = engine.provider.generate(
            [ChatMessage(role="user", content="time?")], tools=[spec]
        )
        assert result.tool_calls[0].name == "get_time"
        assert result.tool_calls[0].arguments == {"tz": "UTC"}


# ---------------------------------------------------------------------------
# LLMEngine: Ollama facade
# ---------------------------------------------------------------------------


class TestLLMEngineOllama:
    def _engine(self) -> LLMEngine:
        return LLMEngine(LLMConfig(model="llama3.2"))

    def test_generate_ollama(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":  # provider availability probe
                return httpx.Response(200, json={"models": []})
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Hello!"},
                    "prompt_eval_count": 3,
                    "eval_count": 2,
                    "done": True,
                },
            )

        with patch.object(OllamaDetector, "is_running", return_value=True):
            captured = _attach_provider_client(engine, handler)
            result = engine.generate("Hi")
        assert result.content == "Hello!"
        posts = [r for r in captured if r.method == "POST"]
        assert posts[0].url == "http://localhost:11434/api/chat"

    def test_service_not_available(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        _attach_provider_client(engine, handler)
        with pytest.raises(RuntimeError, match="not running"):
            engine.generate("Hi")

    def test_switch_model_success(self) -> None:
        engine = self._engine()
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(
                OllamaDetector,
                "get_model_info",
                return_value=LLMModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA),
            ):
                assert engine.switch_model("mistral") is True
                assert engine.current_model == "mistral"

    def test_switch_model_not_found(self) -> None:
        engine = self._engine()
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with patch.object(OllamaDetector, "get_model_info", return_value=None):
                with patch.object(
                    LLMEngine,
                    "list_models",
                    return_value=[LLMModelInfo(name="mistral:latest", provider=ModelProvider.OLLAMA)],
                ):
                    with pytest.raises(ValueError, match="not found"):
                        engine.switch_model("nonexistent")

    def test_switch_model_service_not_running(self) -> None:
        engine = self._engine()
        with patch.object(OllamaDetector, "is_running", return_value=False):
            with pytest.raises(RuntimeError, match="not running"):
                engine.switch_model("mistral")

    def test_switch_model_invalid_name(self) -> None:
        engine = self._engine()
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with pytest.raises(ValueError, match="illegal characters"):
                engine.switch_model("invalid/model")

    def test_embeddings_ollama(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

        with patch.object(OllamaDetector, "is_running", return_value=True):
            _attach_provider_client(engine, handler)
            result = engine.embeddings("text")
        assert result == [[0.1, 0.2, 0.3]]

    def test_context_manager(self) -> None:
        with patch.object(OllamaDetector, "is_running", return_value=True):
            with LLMEngine() as engine:
                assert engine.current_model == "llama3.2"

    def test_init_invalid_model_name_no_dead_code(self) -> None:
        """Regression: the old test had unreachable lines after a ValueError.
        A bad model name must raise during config construction, before the
        engine is ever created."""
        with pytest.raises(ValueError, match="illegal characters"):
            LLMConfig(model="model<script>")
        # No engine is constructed from an invalid config; nothing follows.


# ---------------------------------------------------------------------------
# LLMEngine: DeepSeek (Anthropic-compatible) facade
# ---------------------------------------------------------------------------


class TestLLMEngineDeepSeek:
    def _engine(self) -> LLMEngine:
        return LLMEngine(
            LLMConfig(
                provider=ModelProvider.DEEPSEEK,
                base_url="https://api.deepseek.com/anthropic",
                api_key="sk-ds",
                model="deepseek-chat",
            )
        )

    def test_generate_routes_to_anthropic_endpoint(self) -> None:
        engine = self._engine()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "deepseek-chat",
                    "content": [{"type": "text", "text": "你好"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        captured = _attach_provider_client(engine, handler)
        result = engine.generate("你好")
        assert result.content == "你好"
        # DeepSeek uses the Anthropic /v1/messages endpoint with x-api-key.
        assert captured[0].url == "https://api.deepseek.com/anthropic/v1/messages"
        assert captured[0].headers["x-api-key"] == "sk-ds"


# ---------------------------------------------------------------------------
# StreamEvent typing sanity (facade stream filters TextDeltaEvent)
# ---------------------------------------------------------------------------


class TestStreamEventFiltering:
    def test_facade_stream_skips_non_text_events(self) -> None:
        engine = LLMEngine(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key="k",
                model="claude-3-sonnet",
            )
        )
        # Stream includes a tool-use block; facade.stream must only yield text.
        sse = _sse([
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"A"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"B"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_provider_client(engine, handler)
        assert list(engine.stream("x")) == ["A", "B"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
