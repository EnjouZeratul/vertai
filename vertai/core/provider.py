"""LLM provider abstraction layer.

Defines the ``LLMProvider`` ABC and concrete adapters for Ollama, Anthropic,
DeepSeek (Anthropic-compatible), and OpenAI. Adapters are async-first
(real ``httpx.AsyncClient``); sync variants use an independent ``httpx.Client``
(not an ``asyncio.run`` wrapper, to avoid nested event-loop issues).

The module is the contract layer defined by ``docs/ARCHITECTURE.md`` section 3.1.
``vertai/core/llm.py`` keeps ``LLMEngine`` as a backward-compatible facade that
delegates to :func:`create_provider`.

Tool calling is supported at the protocol layer: ``generate(tools=...)`` builds
the provider-specific tools payload (Anthropic tools / OpenAI functions /
Ollama tools) and parses ``tool_use`` / ``tool_calls`` into
``GenerateResult.tool_calls``.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Protocol, Union

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Anthropic Messages API version (used by Anthropic + DeepSeek adapters).
ANTHROPIC_API_VERSION = "2023-06-01"

# Model-name safety pattern: letters, digits, dot, underscore, hyphen.
_MODEL_NAME_PATTERN = r"^[a-zA-Z0-9._-]+$"


class ModelProvider(str, Enum):
    """Model provider enum.

    ``CUSTOM`` is kept for backward compatibility; ``create_provider`` falls
    back to the Anthropic-compatible path for it (configurable via base_url).
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Model availability status (used by :class:`LLMModelInfo`)."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class LLMModelInfo:
    """Metadata for a model (renamed from ``ModelInfo`` to avoid clashing with
    ``vertai.local.ModelInfo``; the local side is reconciled in S7).
    """

    name: str
    provider: ModelProvider
    status: ModelStatus = ModelStatus.AVAILABLE
    size: str | None = None
    parameters: str | None = None
    quantization: str | None = None
    modified_at: str | None = None


# ---------------------------------------------------------------------------
# Chat / result / streaming / tool types
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single chat message.

    ``role`` is one of ``system``/``user``/``assistant``/``tool``. Content is
    plain text; assistant messages that contained tool calls are represented
    via the separate :class:`ToolCall` list on :class:`GenerateResult`.
    """

    role: str = Field(description="Message role: system/user/assistant/tool")
    content: str = Field(description="Message content")

    model_config = ConfigDict(frozen=True)

    @classmethod
    def coerce(cls, message: ChatMessage | dict[str, Any]) -> ChatMessage:
        """Coerce a dict ``{"role", "content"}`` or a ``ChatMessage`` into a
        ``ChatMessage``. Raises ``TypeError`` for unsupported types so that
        ``chat()`` never silently misinterprets input.
        """
        if isinstance(message, ChatMessage):
            return message
        if isinstance(message, dict):
            return cls(
                role=str(message["role"]),
                content=str(message["content"]),
            )
        raise TypeError(
            f"Unsupported message type {type(message).__name__}; "
            "expected ChatMessage or dict with 'role' and 'content'."
        )


class ToolSpec(BaseModel):
    """Tool specification passed to a provider for function calling.

    ``input_schema`` is a JSON Schema dict describing the tool's parameters.
    """

    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the tool's input parameters",
    )

    model_config = ConfigDict(frozen=True)


class ToolCall(BaseModel):
    """A single tool invocation returned by the model.

    ``arguments`` is the parsed argument dict; ``raw_arguments`` preserves the
    raw string for debugging when parsing fails.
    """

    id: str = Field(description="Tool-call id assigned by the provider")
    name: str = Field(description="Tool name to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Parsed arguments"
    )

    model_config = ConfigDict(frozen=True)


class GenerateResult(BaseModel):
    """Result of a non-streaming generation."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model name that produced the result")
    prompt_tokens: int = Field(default=0, description="Input token count")
    completion_tokens: int = Field(default=0, description="Output token count")
    total_tokens: int = Field(default=0, description="Total token count")
    finish_reason: str = Field(default="stop", description="Stop reason")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the model"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra provider-specific metadata"
    )

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Streaming events (discriminated via dataclass type, not bare str)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextDeltaEvent:
    """A chunk of generated text."""

    text: str


@dataclass(frozen=True)
class ToolUseEvent:
    """A completed tool-use block emitted at the end of the tool block.

    Streaming providers emit incremental argument fragments internally and
    surface a single :class:`ToolUseEvent` once the block closes, so consumers
    receive one event per tool call with the full argument string.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class DoneEvent:
    """Signals stream completion.

    ``tool_calls`` carries fully-parsed tool calls when the stream contained
    tool-use blocks (so non-incremental consumers can read them from the
    terminal event).
    """

    finish_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)


StreamEvent = Union[TextDeltaEvent, ToolUseEvent, DoneEvent]
"""Union of streaming events emitted by ``stream`` / ``astream``."""


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Configuration for an LLM provider.

    API key is injected from the ``VERTAI_API_KEY`` / ``ANTHROPIC_API_KEY``
    environment variables via ``model_validator(mode="before")`` (not by
    overriding ``__init__``).

    Sentinel sampling parameters (``temperature``/``top_p``/``top_k``) default
    to ``None``: providers only include them in the payload when explicitly
    set, rather than guessing with hardcoded ``0.7``/``0.9``/``40``.

    Example:
        # Default local model (Ollama)
        config = LLMConfig()

        # Anthropic-compatible API (e.g. DeepSeek)
        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="deepseek-chat",
        )

        # Real OpenAI
        config = LLMConfig(
            provider=ModelProvider.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key="sk-xxx",
            model="gpt-4o-mini",
        )
    """

    model: str = Field(default="llama3.2", description="Model name")
    provider: ModelProvider = Field(
        default=ModelProvider.OLLAMA, description="Model provider"
    )
    base_url: str = Field(
        default="http://localhost:11434", description="API base URL"
    )
    api_key: str | None = Field(default=None, description="API key")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(default=4096, ge=1, description="Max output tokens")
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Top-p sampling"
    )
    top_k: int | None = Field(
        default=None, ge=0, description="Top-k sampling"
    )
    repeat_penalty: float | None = Field(
        default=None, ge=1.0, description="Repeat penalty (Ollama)"
    )
    seed: int | None = Field(default=None, description="Random seed")
    system_prompt: str | None = Field(default=None, description="System prompt")
    timeout: float = Field(
        default=120.0, ge=1.0, description="Request timeout in seconds"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _inject_env_api_key(cls, data: Any) -> Any:
        """Inject API key from environment if not explicitly provided.

        Priority: explicit ``api_key`` arg > ``VERTAI_API_KEY`` >
        ``ANTHROPIC_API_KEY``.
        """
        if not isinstance(data, dict):
            return data
        if data.get("api_key") is None:
            env_key = os.environ.get("VERTAI_API_KEY") or os.environ.get(
                "ANTHROPIC_API_KEY"
            )
            if env_key:
                data["api_key"] = env_key
        return data

    @model_validator(mode="after")
    def _validate_model_name(self) -> LLMConfig:
        """Reject model names with characters that could enable injection."""
        if not re.match(_MODEL_NAME_PATTERN, self.model):
            raise ValueError(
                "Model name contains illegal characters. "
                "Only letters, digits, dot (.), underscore (_), and hyphen (-) "
                "are allowed."
            )
        return self

    def is_anthropic_compatible(self) -> bool:
        """Whether the provider speaks the Anthropic Messages API."""
        return self.provider in (
            ModelProvider.ANTHROPIC,
            ModelProvider.DEEPSEEK,
        )

    def is_openai_compatible(self) -> bool:
        """Whether the provider speaks the OpenAI Chat Completions API.

        Note: ``CUSTOM`` is treated as Anthropic-compatible by default; callers
        who want OpenAI semantics must set ``provider=ModelProvider.OPENAI``.
        """
        return self.provider is ModelProvider.OPENAI


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """LLM provider abstraction.

    Each concrete provider (Ollama/Anthropic/DeepSeek/OpenAI) implements sync
    and async generation + streaming with optional tool calling. ``messages``
    are always :class:`ChatMessage`; ``tools`` is a list of :class:`ToolSpec`.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # -- client lifecycle -------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self.config.timeout)
        return self._sync_client

    async def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.timeout
            )
        return self._async_client

    def close(self) -> None:
        """Close the underlying sync HTTP client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> LLMProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- message helpers --------------------------------------------------

    @staticmethod
    def _coerce_messages(
        messages: list[ChatMessage] | list[dict[str, Any]],
    ) -> list[ChatMessage]:
        """Coerce a list of ChatMessage or dict into list[ChatMessage]."""
        return [ChatMessage.coerce(m) for m in messages]

    @staticmethod
    def _split_system(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[ChatMessage]]:
        """Pull leading/system messages into a system string (Anthropic style).

        Returns ``(system_or_None, non_system_messages)``.
        """
        system_parts: list[str] = []
        rest: list[ChatMessage] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                rest.append(m)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, rest

    # -- abstract API -----------------------------------------------------

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult: ...

    @abstractmethod
    def astream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming generation.

        Concrete adapters implement this as ``async def`` (an async generator);
        the ABC declares it without ``async`` so the override is type-compatible
        (see mypy's async-iterator override rules).
        """
        ...


# ---------------------------------------------------------------------------
# Provider-capable config protocol (for dependency injection in S3+)
# ---------------------------------------------------------------------------


class ProviderLike(Protocol):
    """Minimal protocol for objects that can act as a provider.

    Allows scenarios (S3+) to depend on the abstraction rather than a concrete
    adapter. ``LLMProvider`` satisfies this protocol.
    """

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object string; return ``{}`` on failure (best-effort for
    incremental tool-argument fragments that may be partial)."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_anthropic_content(
    blocks: list[dict[str, Any]],
) -> tuple[str, str, list[ToolCall]]:
    """Extract text, thinking, and tool calls from Anthropic content blocks.

    Returns ``(text_content, thinking_content, tool_calls)``.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in blocks:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "thinking":
            thinking_parts.append(str(block.get("thinking", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=str(block.get("id", "")),
                    name=str(block.get("name", "")),
                    arguments=(
                        block.get("input", {})
                        if isinstance(block.get("input"), dict)
                        else _parse_json_object(str(block.get("input", "{}")))
                    ),
                )
            )
    return "".join(text_parts), "".join(thinking_parts), tool_calls


def _parse_anthropic_response(
    data: dict[str, Any], default_model: str
) -> GenerateResult:
    """Parse an Anthropic Messages API response into GenerateResult."""
    text, thinking, tool_calls = _extract_anthropic_content(
        list(data.get("content", []))
    )
    usage = data.get("usage", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    return GenerateResult(
        content=text,
        model=str(data.get("model", default_model)),
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        finish_reason=str(data.get("stop_reason", "stop")),
        tool_calls=tool_calls,
        metadata={
            "id": data.get("id"),
            "type": data.get("type"),
            "thinking": thinking,
        },
    )


def _parse_openai_response(
    data: dict[str, Any], default_model: str
) -> GenerateResult:
    """Parse an OpenAI Chat Completions response into GenerateResult."""
    choices = data.get("choices", [])
    content = ""
    tool_calls: list[ToolCall] = []
    finish_reason = "stop"
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        content = str(message.get("content") or "")
        finish_reason = str(choice.get("finish_reason", "stop"))
        for i, tc in enumerate(message.get("tool_calls", []) or []):
            fn = tc.get("function", {})
            args_raw = str(fn.get("arguments", "{}"))
            tool_calls.append(
                ToolCall(
                    id=str(tc.get("id", f"call_{i}")),
                    name=str(fn.get("name", "")),
                    arguments=_parse_json_object(args_raw),
                )
            )
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return GenerateResult(
        content=content,
        model=str(data.get("model", default_model)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(usage.get("total_tokens", prompt_tokens + completion_tokens)),
        finish_reason=finish_reason,
        tool_calls=tool_calls,
        metadata={"id": data.get("id")},
    )


def _parse_ollama_chat_response(
    data: dict[str, Any], default_model: str
) -> GenerateResult:
    """Parse an Ollama ``/api/chat`` response into GenerateResult."""
    message = data.get("message", {})
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(message.get("tool_calls", []) or []):
        fn = tc.get("function", {})
        tool_calls.append(
            ToolCall(
                id=str(tc.get("id", f"call_{i}")),
                name=str(fn.get("name", "")),
                arguments=(
                    fn.get("arguments", {})
                    if isinstance(fn.get("arguments"), dict)
                    else _parse_json_object(str(fn.get("arguments", "{}")))
                ),
            )
        )
    prompt_eval = int(data.get("prompt_eval_count", 0))
    eval_count = int(data.get("eval_count", 0))
    return GenerateResult(
        content=str(message.get("content", "")),
        model=str(data.get("model", default_model)),
        prompt_tokens=prompt_eval,
        completion_tokens=eval_count,
        total_tokens=prompt_eval + eval_count,
        finish_reason="stop" if data.get("done") else "length",
        tool_calls=tool_calls,
        metadata={"done": data.get("done")},
    )


# ---------------------------------------------------------------------------
# SSE line iteration helpers
# ---------------------------------------------------------------------------


def _iter_sse_lines_sync(
    response: httpx.Response,
) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON payloads from ``data: ...`` SSE lines (sync)."""
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            continue


async def _iter_sse_lines_async(
    response: httpx.Response,
) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed JSON payloads from ``data: ...`` SSE lines (async)."""
    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# Anthropic adapter (also base for DeepSeek)
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Adapter for the Anthropic Messages API (``/v1/messages``).

    DeepSeek's Anthropic-compatible endpoint reuses this adapter via
    :class:`DeepSeekProvider`.
    """

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key or "",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/v1/messages"

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
        tools: list[ToolSpec] | None,
        system_override: str | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = self.config
        system, rest = self._split_system(messages)
        if system_override is not None:
            system = system_override
        elif system is None and config.system_prompt is not None:
            system = config.system_prompt

        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": int(kwargs.get("max_tokens", config.max_tokens)),
            "messages": [{"role": m.role, "content": m.content} for m in rest],
            "stream": stream,
        }
        if system:
            payload["system"] = system

        temperature = kwargs.get("temperature", config.temperature)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", config.top_p)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        top_k = kwargs.get("top_k", config.top_k)
        if top_k is not None:
            payload["top_k"] = int(top_k)

        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]
        return payload

    def _ensure_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError(
                "Anthropic-compatible API requires an API key. Provide it via "
                "the VERTAI_API_KEY / ANTHROPIC_API_KEY environment variable or "
                "LLMConfig(api_key=...)."
            )

    # -- sync -------------------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self._ensure_api_key()
        coerced = self._coerce_messages(messages)
        payload = self._build_payload(
            coerced, stream=False, tools=tools, system_override=None, **kwargs
        )
        try:
            response = self._get_sync_client().post(
                self._endpoint(), json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to API service ({self.config.base_url}). "
                "Check the network and base_url configuration."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Anthropic API request failed: {e.response.status_code}\n"
                f"Response: {e.response.text}"
            ) from e
        return _parse_anthropic_response(response.json(), self.config.model)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self._ensure_api_key()
        coerced = self._coerce_messages(messages)
        payload = self._build_payload(
            coerced, stream=True, tools=tools, system_override=None, **kwargs
        )
        try:
            with self._get_sync_client().stream(
                "POST",
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            ) as response:
                yield from self._iter_anthropic_stream(response)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to API service ({self.config.base_url}). "
                "Check the network and base_url configuration."
            ) from e

    @staticmethod
    def _iter_anthropic_stream(
        response: httpx.Response,
    ) -> Iterator[StreamEvent]:
        """Parse an Anthropic SSE stream into StreamEvent values."""
        # Accumulate tool-use blocks (id/name set on start, args accumulate).
        tool_blocks: dict[int, dict[str, str]] = {}
        for data in _iter_sse_lines_sync(response):
            event_type = str(data.get("type", ""))
            if event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    index = int(data.get("index", 0))
                    tool_blocks[index] = {
                        "id": str(block.get("id", "")),
                        "name": str(block.get("name", "")),
                        "args": "",
                    }
            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    yield TextDeltaEvent(text=str(delta.get("text", "")))
                elif delta_type == "input_json_delta":
                    index = int(data.get("index", 0))
                    if index in tool_blocks:
                        tool_blocks[index]["args"] += str(
                            delta.get("partial_json", "")
                        )
            elif event_type == "content_block_stop":
                index = int(data.get("index", 0))
                block = tool_blocks.pop(index, None)
                if block is not None:
                    yield ToolUseEvent(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["args"],
                    )
            elif event_type == "message_stop":
                yield DoneEvent(finish_reason="stop")
                break
            elif event_type == "message_delta":
                delta = data.get("delta", {})
                if delta.get("stop_reason"):
                    # message_stop follows; record reason for terminal event.
                    pass

    # -- async ------------------------------------------------------------

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self._ensure_api_key()
        coerced = self._coerce_messages(messages)
        payload = self._build_payload(
            coerced, stream=False, tools=tools, system_override=None, **kwargs
        )
        client = await self._get_async_client()
        try:
            response = await client.post(
                self._endpoint(), json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to API service ({self.config.base_url}). "
                "Check the network and base_url configuration."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Anthropic API request failed: {e.response.status_code}\n"
                f"Response: {e.response.text}"
            ) from e
        return _parse_anthropic_response(response.json(), self.config.model)

    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self._ensure_api_key()
        coerced = self._coerce_messages(messages)
        payload = self._build_payload(
            coerced, stream=True, tools=tools, system_override=None, **kwargs
        )
        client = await self._get_async_client()
        try:
            async with client.stream(
                "POST",
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            ) as response:
                async for event in self._iter_anthropic_stream_async(response):
                    yield event
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to API service ({self.config.base_url}). "
                "Check the network and base_url configuration."
            ) from e

    @staticmethod
    async def _iter_anthropic_stream_async(
        response: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        tool_blocks: dict[int, dict[str, str]] = {}
        async for data in _iter_sse_lines_async(response):
            event_type = str(data.get("type", ""))
            if event_type == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "tool_use":
                    index = int(data.get("index", 0))
                    tool_blocks[index] = {
                        "id": str(block.get("id", "")),
                        "name": str(block.get("name", "")),
                        "args": "",
                    }
            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type", "")
                if delta_type == "text_delta":
                    yield TextDeltaEvent(text=str(delta.get("text", "")))
                elif delta_type == "input_json_delta":
                    index = int(data.get("index", 0))
                    if index in tool_blocks:
                        tool_blocks[index]["args"] += str(
                            delta.get("partial_json", "")
                        )
            elif event_type == "content_block_stop":
                index = int(data.get("index", 0))
                block = tool_blocks.pop(index, None)
                if block is not None:
                    yield ToolUseEvent(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["args"],
                    )
            elif event_type == "message_stop":
                yield DoneEvent(finish_reason="stop")
                break


class DeepSeekProvider(AnthropicProvider):
    """DeepSeek adapter. DeepSeek exposes an Anthropic-compatible Messages API,
    so this is a thin specialization of :class:`AnthropicProvider`. The
    reasoning ("thinking") content blocks are preserved in
    ``GenerateResult.metadata['thinking']``.
    """


# ---------------------------------------------------------------------------
# OpenAI adapter (fixes C4: real /v1/chat/completions + Bearer)
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """Adapter for the real OpenAI Chat Completions API.

    Uses ``/chat/completions`` under the configured ``base_url`` and sends the
    API key as a ``Authorization: Bearer ...`` header (the previous code routed
    OpenAI through the Anthropic path with ``x-api-key``, which is incorrect).
    """

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key or ''}",
        }

    def _endpoint(self) -> str:
        # base_url is expected to end with /v1; append /chat/completions.
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
        tools: list[ToolSpec] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = self.config
        # OpenAI uses a single messages list; system messages stay inline.
        msg_list: list[dict[str, str]] = []
        system_text = config.system_prompt
        for m in self._coerce_messages(messages):
            if m.role == "system":
                # Merge any inline system messages with the configured one.
                system_text = (
                    f"{system_text}\n\n{m.content}"
                    if system_text
                    else m.content
                )
            else:
                msg_list.append({"role": m.role, "content": m.content})
        if system_text:
            msg_list.insert(0, {"role": "system", "content": system_text})

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": msg_list,
            "stream": stream,
            "max_tokens": int(kwargs.get("max_tokens", config.max_tokens)),
        }
        temperature = kwargs.get("temperature", config.temperature)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", config.top_p)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        seed = kwargs.get("seed", config.seed)
        if seed is not None:
            payload["seed"] = int(seed)

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        return payload

    def _ensure_api_key(self) -> None:
        if not self.config.api_key:
            raise RuntimeError(
                "OpenAI API requires an API key. Provide it via the "
                "VERTAI_API_KEY environment variable or LLMConfig(api_key=...)."
            )

    # -- sync -------------------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self._ensure_api_key()
        payload = self._build_payload(
            self._coerce_messages(messages), stream=False, tools=tools, **kwargs
        )
        try:
            response = self._get_sync_client().post(
                self._endpoint(), json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenAI API ({self.config.base_url})."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenAI API request failed: {e.response.status_code}\n"
                f"Response: {e.response.text}"
            ) from e
        return _parse_openai_response(response.json(), self.config.model)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self._ensure_api_key()
        payload = self._build_payload(
            self._coerce_messages(messages), stream=True, tools=tools, **kwargs
        )
        try:
            with self._get_sync_client().stream(
                "POST",
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            ) as response:
                yield from self._iter_openai_stream(response)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenAI API ({self.config.base_url})."
            ) from e

    @staticmethod
    def _iter_openai_stream(
        response: httpx.Response,
    ) -> Iterator[StreamEvent]:
        """Parse an OpenAI SSE stream into StreamEvent values."""
        tool_acc: dict[int, dict[str, str]] = {}
        tool_order: list[int] = []
        finish_reason = "stop"
        for data in _iter_sse_lines_sync(response):
            if data.get("[DONE]"):  # not standard but defensive
                break
            choices = data.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str) and delta["content"]:
                yield TextDeltaEvent(text=delta["content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = int(tc.get("index", 0))
                if idx not in tool_acc:
                    tool_acc[idx] = {
                        "id": str(tc.get("id", "")),
                        "name": "",
                        "args": "",
                    }
                    tool_order.append(idx)
                if tc.get("id"):
                    tool_acc[idx]["id"] = str(tc["id"])
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_acc[idx]["name"] = str(fn["name"])
                if fn.get("arguments"):
                    tool_acc[idx]["args"] += str(fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
        for idx in tool_order:
            block = tool_acc[idx]
            yield ToolUseEvent(
                id=block["id"], name=block["name"], arguments=block["args"]
            )
        yield DoneEvent(
            finish_reason=finish_reason,
            tool_calls=[
                ToolCall(
                    id=tool_acc[idx]["id"],
                    name=tool_acc[idx]["name"],
                    arguments=_parse_json_object(tool_acc[idx]["args"]),
                )
                for idx in tool_order
            ],
        )

    # -- async ------------------------------------------------------------

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self._ensure_api_key()
        payload = self._build_payload(
            self._coerce_messages(messages), stream=False, tools=tools, **kwargs
        )
        client = await self._get_async_client()
        try:
            response = await client.post(
                self._endpoint(), json=payload, headers=self._headers()
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenAI API ({self.config.base_url})."
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenAI API request failed: {e.response.status_code}\n"
                f"Response: {e.response.text}"
            ) from e
        return _parse_openai_response(response.json(), self.config.model)

    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self._ensure_api_key()
        payload = self._build_payload(
            self._coerce_messages(messages), stream=True, tools=tools, **kwargs
        )
        client = await self._get_async_client()
        try:
            async with client.stream(
                "POST",
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            ) as response:
                async for event in self._iter_openai_stream_async(response):
                    yield event
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to OpenAI API ({self.config.base_url})."
            ) from e

    @staticmethod
    async def _iter_openai_stream_async(
        response: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        tool_acc: dict[int, dict[str, str]] = {}
        tool_order: list[int] = []
        finish_reason = "stop"
        async for data in _iter_sse_lines_async(response):
            choices = data.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str) and delta["content"]:
                yield TextDeltaEvent(text=delta["content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = int(tc.get("index", 0))
                if idx not in tool_acc:
                    tool_acc[idx] = {
                        "id": str(tc.get("id", "")),
                        "name": "",
                        "args": "",
                    }
                    tool_order.append(idx)
                if tc.get("id"):
                    tool_acc[idx]["id"] = str(tc["id"])
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_acc[idx]["name"] = str(fn["name"])
                if fn.get("arguments"):
                    tool_acc[idx]["args"] += str(fn["arguments"])
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
        for idx in tool_order:
            block = tool_acc[idx]
            yield ToolUseEvent(
                id=block["id"], name=block["name"], arguments=block["args"]
            )
        yield DoneEvent(
            finish_reason=finish_reason,
            tool_calls=[
                ToolCall(
                    id=tool_acc[idx]["id"],
                    name=tool_acc[idx]["name"],
                    arguments=_parse_json_object(tool_acc[idx]["args"]),
                )
                for idx in tool_order
            ],
        )


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    """Adapter for the Ollama ``/api/chat`` API."""

    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/chat"

    def _build_payload(
        self,
        messages: list[ChatMessage],
        *,
        stream: bool,
        tools: list[ToolSpec] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = self.config
        system, rest = self._split_system(messages)
        if system is None and config.system_prompt is not None:
            system = config.system_prompt
        msg_list: list[dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in rest
        ]
        if system:
            msg_list.insert(0, {"role": "system", "content": system})

        options: dict[str, Any] = {
            "num_predict": int(kwargs.get("max_tokens", config.max_tokens)),
        }
        temperature = kwargs.get("temperature", config.temperature)
        if temperature is not None:
            options["temperature"] = float(temperature)
        top_p = kwargs.get("top_p", config.top_p)
        if top_p is not None:
            options["top_p"] = float(top_p)
        top_k = kwargs.get("top_k", config.top_k)
        if top_k is not None:
            options["top_k"] = int(top_k)
        repeat_penalty = kwargs.get("repeat_penalty", config.repeat_penalty)
        if repeat_penalty is not None:
            options["repeat_penalty"] = float(repeat_penalty)
        seed = kwargs.get("seed", config.seed)
        if seed is not None:
            options["seed"] = int(seed)

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": msg_list,
            "stream": stream,
            "options": options,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        return payload

    def _ensure_available(self, running: bool) -> None:
        if not running:
            raise RuntimeError(
                "Ollama service is not running.\n\n"
                "Steps:\n"
                "  1. Install Ollama: https://ollama.ai\n"
                "  2. Start the service: ollama serve\n"
                "  3. Pull a model: ollama pull <model>\n"
                "Check the base_url configuration."
            )

    def _is_running(self) -> bool:
        try:
            response = self._get_sync_client().get(
                f"{self.config.base_url.rstrip('/')}/api/tags"
            )
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # -- sync -------------------------------------------------------------

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        self._ensure_available(self._is_running())
        payload = self._build_payload(
            self._coerce_messages(messages), stream=False, tools=tools, **kwargs
        )
        try:
            response = self._get_sync_client().post(self._endpoint(), json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama ({self.config.base_url}). "
                "Ensure Ollama is running: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama request failed: {e.response.status_code} - "
                f"{e.response.text}"
            ) from e
        return _parse_ollama_chat_response(response.json(), self.config.model)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self._ensure_available(self._is_running())
        payload = self._build_payload(
            self._coerce_messages(messages), stream=True, tools=tools, **kwargs
        )
        try:
            with self._get_sync_client().stream(
                "POST", self._endpoint(), json=payload
            ) as response:
                yield from self._iter_ollama_stream(response)
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama ({self.config.base_url}). "
                "Ensure Ollama is running: ollama serve"
            ) from e

    @staticmethod
    def _iter_ollama_stream(
        response: httpx.Response,
    ) -> Iterator[StreamEvent]:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message", {})
            if isinstance(message.get("content"), str) and message["content"]:
                yield TextDeltaEvent(text=message["content"])
            if data.get("done"):
                yield DoneEvent(
                    finish_reason="stop" if data.get("done") else "length"
                )
                break

    # -- async ------------------------------------------------------------

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        # Reuse a sync availability probe (cheap GET); async path still does
        # the real generation via AsyncClient.
        self._ensure_available(self._is_running())
        payload = self._build_payload(
            self._coerce_messages(messages), stream=False, tools=tools, **kwargs
        )
        client = await self._get_async_client()
        try:
            response = await client.post(self._endpoint(), json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama ({self.config.base_url}). "
                "Ensure Ollama is running: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama request failed: {e.response.status_code} - "
                f"{e.response.text}"
            ) from e
        return _parse_ollama_chat_response(response.json(), self.config.model)

    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        self._ensure_available(self._is_running())
        payload = self._build_payload(
            self._coerce_messages(messages), stream=True, tools=tools, **kwargs
        )
        client = await self._get_async_client()
        try:
            async with client.stream("POST", self._endpoint(), json=payload) as response:
                async for event in self._iter_ollama_stream_async(response):
                    yield event
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama ({self.config.base_url}). "
                "Ensure Ollama is running: ollama serve"
            ) from e

    @staticmethod
    async def _iter_ollama_stream_async(
        response: httpx.Response,
    ) -> AsyncIterator[StreamEvent]:
        async for line in response.aiter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message", {})
            if isinstance(message.get("content"), str) and message["content"]:
                yield TextDeltaEvent(text=message["content"])
            if data.get("done"):
                yield DoneEvent(
                    finish_reason="stop" if data.get("done") else "length"
                )
                break


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_provider(config: LLMConfig) -> LLMProvider:
    """Route a :class:`LLMConfig` to the appropriate :class:`LLMProvider`.

    Routing:
        - ``OLLAMA`` -> :class:`OllamaProvider`
        - ``OPENAI`` -> :class:`OpenAIProvider` (real ``/chat/completions`` +
          Bearer header; fixes C4)
        - ``ANTHROPIC`` -> :class:`AnthropicProvider`
        - ``DEEPSEEK`` -> :class:`DeepSeekProvider` (Anthropic-compatible)
        - ``CUSTOM`` -> :class:`AnthropicProvider` (default; configurable via
          ``base_url``)
    """
    provider = config.provider
    if provider is ModelProvider.OLLAMA:
        return OllamaProvider(config)
    if provider is ModelProvider.OPENAI:
        return OpenAIProvider(config)
    if provider is ModelProvider.ANTHROPIC:
        return AnthropicProvider(config)
    if provider is ModelProvider.DEEPSEEK:
        return DeepSeekProvider(config)
    if provider is ModelProvider.CUSTOM:
        return AnthropicProvider(config)
    # Unreachable: enum is exhaustive, but keep a defensive fallback.
    raise ValueError(f"Unknown provider: {provider!r}")


__all__ = [
    "ANTHROPIC_API_VERSION",
    "AnthropicProvider",
    "ChatMessage",
    "DeepSeekProvider",
    "DoneEvent",
    "GenerateResult",
    "LLMConfig",
    "LLMModelInfo",
    "LLMProvider",
    "ModelProvider",
    "ModelStatus",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderLike",
    "StreamEvent",
    "TextDeltaEvent",
    "ToolCall",
    "ToolSpec",
    "ToolUseEvent",
    "create_provider",
]
