"""LLM engine facade (backward-compat layer over :mod:`vertai.core.provider`).

The real provider abstraction lives in :mod:`vertai.core.provider`
(``LLMProvider`` ABC + Ollama/Anthropic/DeepSeek/OpenAI adapters). This module
keeps the legacy ``LLMEngine`` API working for existing scenarios and tests by
delegating to :func:`vertai.core.provider.create_provider`.

Legacy entry points preserved:
    - ``LLMEngine(config)`` — wraps a provider selected from ``config.provider``
    - ``generate(prompt, system_prompt=...)`` — single-prompt convenience
    - ``stream(prompt, ...)`` — single-prompt streaming (yields ``str`` for
      backward compat; use the provider's ``stream`` for ``StreamEvent``)
    - ``chat(messages)`` / ``chat_stream(messages)`` — accept ``list[ChatMessage]``
      **or** ``list[dict]`` (coerced uniformly)
    - ``embeddings(text)`` — Ollama embeddings (kept for S3 to refactor)

New code should prefer ``create_provider(config)`` and the
:class:`~vertai.core.provider.LLMProvider` API directly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Union

import httpx

from vertai.core.provider import (
    ANTHROPIC_API_VERSION,
    ChatMessage,
    DoneEvent,
    GenerateResult,
    LLMConfig,
    LLMModelInfo,
    LLMProvider,
    ModelProvider,
    ModelStatus,
    StreamEvent,
    TextDeltaEvent,
    ToolCall,
    ToolSpec,
    ToolUseEvent,
    create_provider,
)

__all__ = [
    "ANTHROPIC_API_VERSION",
    "ChatMessage",
    "GenerateResult",
    "LLMConfig",
    "LLMEngine",
    "LLMModelInfo",
    "LLMProvider",
    "ModelInfo",
    "ModelProvider",
    "ModelStatus",
    "OllamaDetector",
    "StreamEvent",
    "TextDeltaEvent",
    "ToolCall",
    "ToolSpec",
    "ToolUseEvent",
    "DoneEvent",
    "create_provider",
]

# Backward-compat alias. The core side is renamed to ``LLMModelInfo`` to avoid
# clashing with ``vertai.local.ModelInfo`` (the local side is reconciled in S7).
# Kept so legacy imports ``from vertai.core.llm import ModelInfo`` keep working.
ModelInfo = LLMModelInfo


class OllamaDetector:
    """Ollama service detector.

    Probes the local Ollama service (``/api/tags``) and lists available models.
    Returns :class:`LLMModelInfo` instances.
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def is_running(self) -> bool:
        """Check whether the Ollama service is reachable."""
        try:
            response = self._get_client().get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[LLMModelInfo]:
        """List available models. Returns ``[]`` if the service is down."""
        if not self.is_running():
            return []
        try:
            response = self._get_client().get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        models: list[LLMModelInfo] = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            models.append(
                LLMModelInfo(
                    name=str(item.get("name", "unknown")),
                    provider=ModelProvider.OLLAMA,
                    status=ModelStatus.AVAILABLE,
                    size=item.get("size"),
                    modified_at=item.get("modified_at"),
                    parameters=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
            )
        return models

    def get_model_info(self, model_name: str) -> LLMModelInfo | None:
        """Get info for a model by name (matches ``name`` or ``name:tag``)."""
        for model in self.list_models():
            if model.name == model_name or model.name.startswith(f"{model_name}:"):
                return model
        return None

    def pull_model(
        self, model_name: str, stream: bool = False
    ) -> Union[bool, Iterator[dict[str, Any]]]:
        """Pull a model. Returns success bool (non-stream) or progress iterator."""
        if stream:
            return self._pull_model_stream(model_name)
        try:
            response = self._get_client().post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _pull_model_stream(self, model_name: str) -> Iterator[dict[str, Any]]:
        """Stream model pull progress as JSON dicts."""
        with self._get_client().stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": True},
            timeout=300.0,
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaDetector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class LLMEngine:
    """Backward-compatible LLM engine facade.

    Delegates to a :class:`LLMProvider` selected via
    :func:`create_provider`. The legacy single-prompt API (``generate(prompt,
    system_prompt)``) and the ``chat``/``stream``/``chat_stream`` methods are
    preserved; they coerce to :class:`ChatMessage` internally.

    Example:
        # Default local model (Ollama)
        engine = LLMEngine()
        result = engine.generate("Hello!")
        print(result.content)

        # DeepSeek (Anthropic-compatible)
        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="deepseek-chat",
        )
        engine = LLMEngine(config)

        # Multi-turn chat (dict or ChatMessage both accepted)
        messages = [
            {"role": "user", "content": "Hi"},
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(role="user", "content": "Who are you?"),
        ]
        result = engine.chat(messages)
    """

    _MODEL_NAME_PATTERN = r"^[a-zA-Z0-9._-]+$"

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._detector = OllamaDetector(self.config.base_url)
        self._provider = create_provider(self.config)

    # -- provider access --------------------------------------------------

    @property
    def provider(self) -> LLMProvider:
        """The underlying :class:`LLMProvider` instance."""
        return self._provider

    @property
    def current_model(self) -> str:
        """The configured model name."""
        return self.config.model

    @property
    def is_available(self) -> bool:
        """Whether the service is available.

        For Anthropic-compatible/OpenAI providers, availability means an API
        key is configured; for Ollama it means the service is running.
        """
        if self.config.provider is ModelProvider.OLLAMA:
            return self._detector.is_running()
        return self.config.api_key is not None

    # -- model management (Ollama convenience) ----------------------------

    def list_models(self) -> list[LLMModelInfo]:
        """List available Ollama models."""
        return self._detector.list_models()

    def switch_model(self, model_name: str) -> bool:
        """Switch the active model (Ollama only).

        Raises ``RuntimeError`` if the Ollama service is not running and
        ``ValueError`` if the model is not found or the name is invalid.
        """
        if not re.match(self._MODEL_NAME_PATTERN, model_name):
            raise ValueError(
                "Model name contains illegal characters. "
                "Only letters, digits, dot (.), underscore (_), and hyphen (-) "
                "are allowed."
            )
        if not self._detector.is_running():
            raise RuntimeError(
                "Ollama service is not running. Start it with:\n"
                "  1. Install: https://ollama.ai\n"
                "  2. Run: ollama serve\n"
                "  3. Pull a model: ollama pull <model>"
            )
        model_info = self._detector.get_model_info(model_name)
        if model_info is None:
            available = [m.name for m in self.list_models()]
            if available:
                raise ValueError(
                    "Requested model not found.\n"
                    f"Available models: {', '.join(available)}\n"
                    "Use 'ollama pull' to fetch new models."
                )
            raise ValueError(
                "Requested model not found and no models are available locally.\n"
                "Use 'ollama pull' to fetch a model."
            )
        # Mutate config in place so the provider picks up the new model.
        self.config = self.config.model_copy(update={"model": model_name})
        self._provider = create_provider(self.config)
        return True

    # -- generation (legacy single-prompt API) ----------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        """Generate text from a single prompt (legacy convenience API).

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt (overrides config).
            **kwargs: Extra generation parameters forwarded to the provider.
        """
        self._ensure_available()
        messages: list[ChatMessage] = []
        sys_text = system_prompt or self.config.system_prompt
        if sys_text:
            messages.append(ChatMessage(role="system", content=sys_text))
        messages.append(ChatMessage(role="user", content=prompt))
        return self._provider.generate(messages, **kwargs)

    def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text from a single prompt (yields ``str`` for compat).

        For typed ``StreamEvent`` values, use ``self.provider.stream(...)``
        directly.
        """
        self._ensure_available()
        messages: list[ChatMessage] = []
        sys_text = system_prompt or self.config.system_prompt
        if sys_text:
            messages.append(ChatMessage(role="system", content=sys_text))
        messages.append(ChatMessage(role="user", content=prompt))
        for event in self._provider.stream(messages, **kwargs):
            if isinstance(event, TextDeltaEvent):
                yield event.text

    # -- chat -------------------------------------------------------------

    def chat(
        self,
        messages: Union[list[ChatMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> GenerateResult:
        """Multi-turn chat.

        ``messages`` accepts ``list[ChatMessage]`` or ``list[dict]`` (each dict
        must have ``role`` and ``content``); both are coerced uniformly via
        :meth:`ChatMessage.coerce`.
        """
        self._ensure_available()
        coerced = [ChatMessage.coerce(m) for m in messages]
        return self._provider.generate(coerced, **kwargs)

    def chat_stream(
        self,
        messages: Union[list[ChatMessage], list[dict[str, Any]]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a multi-turn chat (yields ``str`` for compat)."""
        self._ensure_available()
        coerced = [ChatMessage.coerce(m) for m in messages]
        for event in self._provider.stream(coerced, **kwargs):
            if isinstance(event, TextDeltaEvent):
                yield event.text

    # -- embeddings (Ollama / OpenAI-style; S3 will refactor) -------------

    def embeddings(self, text: Union[str, list[str]]) -> list[list[float]]:
        """Get embeddings.

        For Ollama: uses ``/api/embeddings``. For Anthropic-compatible/OpenAI
        providers: uses ``/v1/embeddings`` (OpenAI format). S3 introduces a
        dedicated :class:`EmbeddingProvider` abstraction; this method is kept
        only for backward compatibility.
        """
        self._ensure_available()
        texts = [text] if isinstance(text, str) else text
        if self.config.provider is ModelProvider.OLLAMA:
            return self._embeddings_ollama(texts)
        return self._embeddings_cloud(texts)

    def _embeddings_ollama(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        client = self._get_sync_client()
        for t in texts:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/api/embeddings",
                json={"model": self.config.model, "prompt": t},
            )
            response.raise_for_status()
            data = response.json()
            results.append(list(data.get("embedding", [])))
        return results

    def _embeddings_cloud(self, texts: list[str]) -> list[list[float]]:
        headers: dict[str, str]
        if self.config.is_openai_compatible():
            headers = {"Authorization": f"Bearer {self.config.api_key or ''}"}
        else:
            headers = {
                "x-api-key": self.config.api_key or "",
                "anthropic-version": ANTHROPIC_API_VERSION,
            }
        results: list[list[float]] = []
        client = self._get_sync_client()
        for t in texts:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/v1/embeddings",
                json={"input": t, "model": self.config.model},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            embedding_data = data.get("data", [])
            if embedding_data:
                results.append(list(embedding_data[0].get("embedding", [])))
            else:
                results.append([])
        return results

    # -- internals --------------------------------------------------------

    def _get_sync_client(self) -> httpx.Client:
        # Reuse the provider's sync client for embeddings (shared lifecycle).
        return self._provider._get_sync_client()

    def _ensure_available(self) -> None:
        """Ensure the configured service is usable."""
        if self.config.provider is ModelProvider.OLLAMA:
            if not self._detector.is_running():
                raise RuntimeError(
                    "Ollama service is not running.\n\n"
                    "Steps:\n"
                    "  1. Install Ollama: https://ollama.ai\n"
                    "  2. Start the service: ollama serve\n"
                    "  3. Pull a model: ollama pull <model>\n"
                    "Check the base_url configuration."
                )
            return
        if not self.config.api_key:
            raise RuntimeError(
                "API requires an API key. Provide it via the "
                "VERTAI_API_KEY / ANTHROPIC_API_KEY environment variable or "
                "LLMConfig(api_key=...)."
            )

    def close(self) -> None:
        """Close the engine and underlying clients."""
        self._provider.close()
        self._detector.close()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
