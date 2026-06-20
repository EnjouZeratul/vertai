"""Tests for the LLMProvider abstraction layer (S2).

Testing strategy (per ROADMAP test table):
- Pure logic (payload construction, SSE/response/tool_use parsing, coercion,
  config env injection, factory routing) -> real assertions, no HTTP.
- HTTP protocol construction (payload/headers/URL) -> httpx.MockTransport that
  stubs *real* wire behavior (returns real-shaped JSON / SSE bytes), so tests
  verify the adapter builds correct requests AND parses real responses.
- OpenAI adapter -> MockTransport verifies the real ``/chat/completions`` URL +
  ``Authorization: Bearer`` header (C4 fix).
- async -> pytest-asyncio driving real ``httpx.AsyncClient`` against MockTransport.
- tool calling end-to-end -> fake responses carrying ``tool_use`` /
  ``tool_calls`` parsed into ``GenerateResult.tool_calls``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from vertai.core.provider import (
    ANTHROPIC_API_VERSION,
    AnthropicProvider,
    ChatMessage,
    DeepSeekProvider,
    DoneEvent,
    LLMConfig,
    LLMProvider,
    ModelProvider,
    OllamaProvider,
    OpenAIProvider,
    TextDeltaEvent,
    ToolSpec,
    ToolUseEvent,
    create_provider,
    _parse_anthropic_response,
    _parse_openai_response,
    _parse_ollama_chat_response,
)


# ---------------------------------------------------------------------------
# Helpers: build a provider with a MockTransport-backed client.
# ---------------------------------------------------------------------------


SyncHandler = Callable[[httpx.Request], httpx.Response]


def _make_sync_client(handler: SyncHandler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _make_async_client(handler: SyncHandler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _attach_sync(provider: LLMProvider, handler: SyncHandler) -> httpx.Request:
    """Attach a MockTransport sync client and return a captured request holder.

    Returns a list-like via a one-element container so the handler can record
    the request; we return the container for assertion.
    """
    captured: list[httpx.Request] = []

    def _h(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    provider._sync_client = _make_sync_client(_h)  # type: ignore[attr-defined]
    return captured  # type: ignore[return-value]


def _attach_async(provider: LLMProvider, handler: SyncHandler) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _h(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    provider._async_client = _make_async_client(_h)  # type: ignore[attr-defined]
    return captured


def _sse(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Pure logic: response parsers
# ---------------------------------------------------------------------------


class TestAnthropicResponseParser:
    def test_text_block(self) -> None:
        data = {
            "id": "msg-1",
            "type": "message",
            "model": "claude-3-sonnet",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _parse_anthropic_response(data, "fallback")
        assert result.content == "Hello!"
        assert result.model == "claude-3-sonnet"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.finish_reason == "end_turn"
        assert result.tool_calls == []

    def test_thinking_block_goes_to_metadata(self) -> None:
        data = {
            "model": "deepseek-chat",
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 7},
        }
        result = _parse_anthropic_response(data, "deepseek-chat")
        assert result.content == "The answer is 42."
        assert result.metadata["thinking"] == "Let me think..."

    def test_tool_use_block_parsed_into_tool_calls(self) -> None:
        data = {
            "model": "claude-3-sonnet",
            "content": [
                {"type": "text", "text": "Calling tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Tokyo", "units": "celsius"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = _parse_anthropic_response(data, "claude-3-sonnet")
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call.id == "toolu_01"
        assert call.name == "get_weather"
        assert call.arguments == {"city": "Tokyo", "units": "celsius"}
        assert result.finish_reason == "tool_use"


class TestOpenAIResponseParser:
    def test_content_and_usage(self) -> None:
        data = {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        }
        result = _parse_openai_response(data, "fallback")
        assert result.content == "Hi there"
        assert result.model == "gpt-4o-mini"
        assert result.total_tokens == 10
        assert result.finish_reason == "stop"

    def test_tool_calls_parsed(self) -> None:
        data = {
            "id": "chatcmpl-2",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "vertai"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = _parse_openai_response(data, "gpt-4o-mini")
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "vertai"}
        assert result.finish_reason == "tool_calls"


class TestOllamaResponseParser:
    def test_content_and_done(self) -> None:
        data = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello!"},
            "prompt_eval_count": 4,
            "eval_count": 2,
            "done": True,
        }
        result = _parse_ollama_chat_response(data, "llama3.2")
        assert result.content == "Hello!"
        assert result.prompt_tokens == 4
        assert result.completion_tokens == 2
        assert result.finish_reason == "stop"

    def test_tool_calls_parsed(self) -> None:
        data = {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": {"expr": "1+1"},
                        }
                    }
                ],
            },
            "done": True,
        }
        result = _parse_ollama_chat_response(data, "llama3.2")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"
        assert result.tool_calls[0].arguments == {"expr": "1+1"}


# ---------------------------------------------------------------------------
# Pure logic: ChatMessage coercion
# ---------------------------------------------------------------------------


class TestChatMessageCoerce:
    def test_passes_through_chatmessage(self) -> None:
        m = ChatMessage(role="user", content="hi")
        assert ChatMessage.coerce(m) is m

    def test_coerces_dict(self) -> None:
        m = ChatMessage.coerce({"role": "assistant", "content": "hello"})
        assert m.role == "assistant"
        assert m.content == "hello"

    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported message type"):
            ChatMessage.coerce("not a message")  # type: ignore[arg-type]

    def test_rejects_missing_keys(self) -> None:
        with pytest.raises(KeyError):
            ChatMessage.coerce({"role": "user"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Pure logic: payload construction
# ---------------------------------------------------------------------------


class TestAnthropicPayload:
    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key="k",
                model="claude-3-sonnet",
            )
        )

    def test_system_split_out_of_messages(self) -> None:
        provider = self._provider()
        messages = [
            ChatMessage(role="system", content="Be brief."),
            ChatMessage(role="user", content="Hi"),
        ]
        payload = provider._build_payload(
            messages, stream=False, tools=None, system_override=None
        )
        assert payload["system"] == "Be brief."
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["stream"] is False

    def test_none_sentinel_params_omitted(self) -> None:
        provider = self._provider()
        messages = [ChatMessage(role="user", content="Hi")]
        payload = provider._build_payload(
            messages, stream=False, tools=None, system_override=None
        )
        # temperature/top_p/top_k default to None -> must NOT appear
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload

    def test_explicit_params_included(self) -> None:
        provider = self._provider()
        messages = [ChatMessage(role="user", content="Hi")]
        payload = provider._build_payload(
            messages,
            stream=False,
            tools=None,
            system_override=None,
            temperature=0.3,
            top_p=0.8,
        )
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.8

    def test_tools_use_anthropic_schema(self) -> None:
        provider = self._provider()
        messages = [ChatMessage(role="user", content="weather?")]
        spec = ToolSpec(
            name="get_weather",
            description="Get weather",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        )
        payload = provider._build_payload(
            messages, stream=False, tools=[spec], system_override=None
        )
        assert payload["tools"] == [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]


class TestOpenAIPayload:
    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk-test",
                model="gpt-4o-mini",
            )
        )

    def test_system_message_merged_inline(self) -> None:
        provider = self._provider()
        messages = [
            ChatMessage(role="system", content="Be nice."),
            ChatMessage(role="user", content="Hi"),
        ]
        payload = provider._build_payload(messages, stream=False, tools=None)
        assert payload["messages"][0] == {"role": "system", "content": "Be nice."}
        assert payload["messages"][1] == {"role": "user", "content": "Hi"}

    def test_tools_use_function_format(self) -> None:
        provider = self._provider()
        messages = [ChatMessage(role="user", content="search")]
        spec = ToolSpec(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {}},
        )
        payload = provider._build_payload(messages, stream=False, tools=[spec])
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def test_none_sentinel_omitted(self) -> None:
        provider = self._provider()
        messages = [ChatMessage(role="user", content="Hi")]
        payload = provider._build_payload(messages, stream=False, tools=None)
        assert "temperature" not in payload
        assert "seed" not in payload


class TestOllamaPayload:
    def _provider(self) -> OllamaProvider:
        return OllamaProvider(LLMConfig(model="llama3.2"))

    def test_system_inserted_as_message(self) -> None:
        provider = self._provider()
        messages = [
            ChatMessage(role="system", content="Be brief."),
            ChatMessage(role="user", content="Hi"),
        ]
        payload = provider._build_payload(messages, stream=False, tools=None)
        assert payload["messages"][0] == {"role": "system", "content": "Be brief."}
        assert payload["options"]["num_predict"] == 4096
        # None-sentinel params omitted from options
        assert "temperature" not in payload["options"]

    def test_seed_included_when_set(self) -> None:
        provider = OllamaProvider(LLMConfig(model="llama3.2", seed=42))
        messages = [ChatMessage(role="user", content="Hi")]
        payload = provider._build_payload(messages, stream=False, tools=None)
        assert payload["options"]["seed"] == 42


# ---------------------------------------------------------------------------
# Pure logic: config env injection + factory
# ---------------------------------------------------------------------------


class TestLLMConfig:
    def test_env_injection_vertai_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERTAI_API_KEY", "vertai-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        config = LLMConfig(provider=ModelProvider.ANTHROPIC)
        assert config.api_key == "vertai-key"

    def test_env_injection_anthropic_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
        config = LLMConfig(provider=ModelProvider.ANTHROPIC)
        assert config.api_key == "anthropic-key"

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VERTAI_API_KEY", "env-key")
        config = LLMConfig(provider=ModelProvider.ANTHROPIC, api_key="explicit")
        assert config.api_key == "explicit"

    def test_none_sentinel_defaults(self) -> None:
        config = LLMConfig()
        assert config.temperature is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.seed is None

    def test_invalid_model_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="illegal characters"):
            LLMConfig(model="bad/model")

    def test_is_anthropic_compatible(self) -> None:
        assert LLMConfig(provider=ModelProvider.ANTHROPIC, api_key="k").is_anthropic_compatible()
        assert LLMConfig(provider=ModelProvider.DEEPSEEK, api_key="k").is_anthropic_compatible()
        assert not LLMConfig(provider=ModelProvider.OLLAMA).is_anthropic_compatible()

    def test_is_openai_compatible(self) -> None:
        assert LLMConfig(provider=ModelProvider.OPENAI, api_key="k").is_openai_compatible()
        assert not LLMConfig(provider=ModelProvider.ANTHROPIC, api_key="k").is_openai_compatible()


class TestCreateProvider:
    def test_routing(self) -> None:
        assert isinstance(
            create_provider(LLMConfig(provider=ModelProvider.OLLAMA)), OllamaProvider
        )
        assert isinstance(
            create_provider(LLMConfig(provider=ModelProvider.OPENAI, api_key="k")),
            OpenAIProvider,
        )
        assert isinstance(
            create_provider(LLMConfig(provider=ModelProvider.ANTHROPIC, api_key="k")),
            AnthropicProvider,
        )
        assert isinstance(
            create_provider(LLMConfig(provider=ModelProvider.DEEPSEEK, api_key="k")),
            DeepSeekProvider,
        )
        assert isinstance(
            create_provider(LLMConfig(provider=ModelProvider.CUSTOM, api_key="k")),
            AnthropicProvider,
        )

    def test_deepseek_is_anthropic_subclass(self) -> None:
        provider = create_provider(LLMConfig(provider=ModelProvider.DEEPSEEK, api_key="k"))
        assert isinstance(provider, AnthropicProvider)
        assert isinstance(provider, DeepSeekProvider)


# ---------------------------------------------------------------------------
# HTTP protocol: MockTransport verifying payload/headers/URL (real behavior)
# ---------------------------------------------------------------------------


class TestAnthropicHTTP:
    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                base_url="https://api.anthropic.com",
                api_key="sk-ant",
                model="claude-3-sonnet",
            )
        )

    def test_generate_hits_v1_messages_with_x_api_key(self) -> None:
        provider = self._provider()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [{"type": "text", "text": "Hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        captured = _attach_sync(provider, handler)
        result = provider.generate([ChatMessage(role="user", content="hello")])

        assert result.content == "Hi"
        request = captured[0]
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "sk-ant"
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        body = json.loads(request.content)
        assert body["model"] == "claude-3-sonnet"
        assert body["messages"] == [{"role": "user", "content": "hello"}]

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VERTAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicProvider(
            LLMConfig(provider=ModelProvider.ANTHROPIC, api_key=None)
        )
        with pytest.raises(RuntimeError, match="API key"):
            provider.generate([ChatMessage(role="user", content="x")])

    def test_http_status_error_wrapped(self) -> None:
        provider = self._provider()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized")

        _attach_sync(provider, handler)
        with pytest.raises(RuntimeError, match="401"):
            provider.generate([ChatMessage(role="user", content="x")])


class TestOpenAIHTTP:
    """Verifies the C4 fix: real /chat/completions + Bearer header."""

    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk-openai",
                model="gpt-4o-mini",
            )
        )

    def test_generate_hits_chat_completions_with_bearer(self) -> None:
        provider = self._provider()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
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

        captured = _attach_sync(provider, handler)
        result = provider.generate([ChatMessage(role="user", content="hello")])

        assert result.content == "Hi"
        request = captured[0]
        # C4 fix: real /chat/completions endpoint (not /v1/messages)
        assert request.url == "https://api.openai.com/v1/chat/completions"
        # C4 fix: Bearer auth (not x-api-key)
        assert request.headers["authorization"] == "Bearer sk-openai"
        assert "x-api-key" not in {k.lower() for k in request.headers.keys()}
        body = json.loads(request.content)
        assert body["model"] == "gpt-4o-mini"

    def test_tool_calling_end_to_end(self) -> None:
        provider = self._provider()

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
                                            "arguments": '{"timezone": "UTC"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )

        captured = _attach_sync(provider, handler)
        spec = ToolSpec(
            name="get_time",
            description="Get current time",
            input_schema={
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        )
        result = provider.generate(
            [ChatMessage(role="user", content="what time is it?")], tools=[spec]
        )

        # Verify the tools payload was built in OpenAI function format
        body = json.loads(captured[0].content)
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "get_time"
        # Verify the tool call was parsed
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_time"
        assert result.tool_calls[0].arguments == {"timezone": "UTC"}
        assert result.finish_reason == "tool_calls"


class TestOllamaHTTP:
    def _provider(self) -> OllamaProvider:
        return OllamaProvider(
            LLMConfig(
                provider=ModelProvider.OLLAMA,
                base_url="http://localhost:11434",
                model="llama3.2",
            )
        )

    def test_generate_hits_api_chat(self) -> None:
        provider = self._provider()

        def handler(request: httpx.Request) -> httpx.Response:
            # Ollama provider first probes /api/tags (GET) for availability.
            if request.method == "GET":
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

        captured = _attach_sync(provider, handler)
        result = provider.generate([ChatMessage(role="user", content="hi")])

        assert result.content == "Hello!"
        # The POST request (second captured) should target /api/chat
        post_requests = [r for r in captured if r.method == "POST"]
        assert len(post_requests) == 1
        assert post_requests[0].url == "http://localhost:11434/api/chat"

    def test_service_not_running_raises(self) -> None:
        provider = self._provider()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        _attach_sync(provider, handler)
        with pytest.raises(RuntimeError, match="not running"):
            provider.generate([ChatMessage(role="user", content="x")])


# ---------------------------------------------------------------------------
# SSE stream parsing (real behavior, MockTransport returns real SSE bytes)
# ---------------------------------------------------------------------------


class TestAnthropicStream:
    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                base_url="https://api.anthropic.com",
                api_key="k",
                model="claude-3-sonnet",
            )
        )

    def test_text_delta_and_done(self) -> None:
        provider = self._provider()
        sse = _sse([
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"World"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))

        deltas = [e for e in events if isinstance(e, TextDeltaEvent)]
        assert [d.text for d in deltas] == ["Hello", "World"]
        assert any(isinstance(e, DoneEvent) for e in events)

    def test_tool_use_accumulated(self) -> None:
        provider = self._provider()
        events_json = [
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "calc"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"x":'}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": "1}"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        sse = _sse([f"data: {json.dumps(e)}" for e in events_json])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].id == "toolu_1"
        assert tool_events[0].name == "calc"
        assert tool_events[0].arguments == '{"x":1}'

    def test_thinking_delta_skipped(self) -> None:
        provider = self._provider()
        sse = _sse([
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))
        deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas == ["Answer"]


class TestOpenAIStream:
    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk",
                model="gpt-4o-mini",
            )
        )

    def test_content_and_done(self) -> None:
        provider = self._provider()
        sse = _sse([
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{"content":" there"}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))
        deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas == ["Hi", " there"]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].finish_reason == "stop"

    def test_tool_call_streaming_accumulated(self) -> None:
        provider = self._provider()
        events_json = [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "function": {"name": "search"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"q":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"x"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        sse = _sse([f"data: {json.dumps(e)}" for e in events_json])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))
        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].name == "search"
        assert json.loads(tool_events[0].arguments) == {"q": "x"}
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert done[0].tool_calls[0].name == "search"
        assert done[0].tool_calls[0].arguments == {"q": "x"}


class TestOllamaStream:
    def _provider(self) -> OllamaProvider:
        return OllamaProvider(
            LLMConfig(provider=ModelProvider.OLLAMA, model="llama3.2")
        )

    def test_content_and_done(self) -> None:
        provider = self._provider()
        # Ollama streams newline-delimited JSON (not SSE).
        lines = [
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            json.dumps({"message": {"content": " there"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

        _attach_sync(provider, handler)
        events = list(provider.stream([ChatMessage(role="user", content="x")]))
        deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas == ["Hi", " there"]
        assert any(isinstance(e, DoneEvent) for e in events)


# ---------------------------------------------------------------------------
# Async (real httpx.AsyncClient via MockTransport, pytest-asyncio)
# ---------------------------------------------------------------------------


class TestAsyncProviders:
    @pytest.mark.asyncio
    async def test_openai_agenerate(self) -> None:
        provider = OpenAIProvider(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk",
                model="gpt-4o-mini",
            )
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "async hi"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        captured = _attach_async(provider, handler)
        result = await provider.agenerate([ChatMessage(role="user", content="x")])
        assert result.content == "async hi"
        assert captured[0].url == "https://api.openai.com/v1/chat/completions"
        assert captured[0].headers["authorization"] == "Bearer sk"
        await provider.aclose()

    @pytest.mark.asyncio
    async def test_anthropic_astream(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                base_url="https://api.anthropic.com",
                api_key="k",
                model="claude-3-sonnet",
            )
        )
        sse = _sse([
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"A"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"B"}}',
            'data: {"type":"message_stop"}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_async(provider, handler)
        events: list[Any] = []
        async for event in provider.astream([ChatMessage(role="user", content="x")]):
            events.append(event)
        deltas = [e.text for e in events if isinstance(e, TextDeltaEvent)]
        assert deltas == ["A", "B"]
        await provider.aclose()

    @pytest.mark.asyncio
    async def test_openai_astream_tool_use(self) -> None:
        provider = OpenAIProvider(
            LLMConfig(
                provider=ModelProvider.OPENAI,
                base_url="https://api.openai.com/v1",
                api_key="sk",
                model="gpt-4o-mini",
            )
        )
        sse = _sse([
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        ])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse)

        _attach_async(provider, handler)
        events: list[Any] = []
        async for event in provider.astream([ChatMessage(role="user", content="x")]):
            events.append(event)
        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert tool_events[0].name == "f"
        assert json.loads(tool_events[0].arguments) == {}
        await provider.aclose()


# ---------------------------------------------------------------------------
# Tool calling end-to-end across providers (fake responses, real parsing)
# ---------------------------------------------------------------------------


class TestToolCallingEndToEnd:
    def test_anthropic_tool_use_roundtrip(self) -> None:
        provider = AnthropicProvider(
            LLMConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key="k",
                model="claude-3-sonnet",
            )
        )
        spec = ToolSpec(
            name="add",
            description="Add numbers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            # Verify the tool spec was sent in Anthropic format
            assert body["tools"] == [
                {
                    "name": "add",
                    "description": "Add numbers",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                    },
                }
            ]
            return httpx.Response(
                200,
                json={
                    "model": "claude-3-sonnet",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_9",
                            "name": "add",
                            "input": {"a": 2, "b": 3},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        _attach_sync(provider, handler)
        result = provider.generate(
            [ChatMessage(role="user", content="add 2 and 3")], tools=[spec]
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "toolu_9"
        assert result.tool_calls[0].arguments == {"a": 2, "b": 3}
