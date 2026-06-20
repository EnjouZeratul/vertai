"""Lightweight observability callbacks.

Defines the :class:`Callback` :class:`~typing.Protocol` and two built-in
implementations (:class:`LoggingCallback`, :class:`TokenCountCallback`). The
design is intentionally minimal: callbacks are event hooks only, not a full
OpenTelemetry/tracing stack (that is a 1.x extension point per
``docs/ARCHITECTURE.md`` section 3.8).

The :class:`Callback` protocol uses *default* method bodies so users can
implement only the hooks they care about (partial implementations are fine):
``Callback`` is ``@runtime_checkable`` and :class:`Agent` dispatches each hook
defensively, calling a hook only when the object actually provides it. That
keeps duck-typed callbacks working under ``mypy --strict`` without forcing a
common base class.

Events, in order over a single :meth:`Agent.run` call::

    on_agent_start(input)
    on_llm_start(messages)  ─┐
    on_llm_end(result)      ─┘   (repeated per iteration)
    on_tool_start(name, args) ─┐ (per tool call)
    on_tool_end(name, result) ─┘
    ...
    on_agent_end(result)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Protocol, runtime_checkable

from vertai.core.provider import GenerateResult

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover - executed only on 3.10
    from typing_extensions import Self


@runtime_checkable
class Callback(Protocol):
    """Observability hook protocol.

    Implementations may provide any subset of the hooks below. The default
    method bodies are empty so a concrete class that subclasses this Protocol
    (or simply duck-types it) is not forced to override every hook. Agent
    dispatches each hook defensively (it checks the attribute is callable), so
    objects that only implement e.g. ``on_agent_end`` work correctly.

    Note on ``isinstance``: ``@runtime_checkable`` protocols only return
    ``True`` from ``isinstance`` when *every* member is present on the
    instance. A partial implementation therefore fails the isinstance check.
    This is by design and matches the contract: the agent dispatches hooks
    via ``getattr`` (see :func:`dispatch`), not via ``isinstance``, so partial
    callbacks are fully functional at runtime regardless of the isinstance
    result.
    """

    def on_agent_start(self, input: str) -> None:
        """Called once at the start of :meth:`Agent.run` / :meth:`Agent.arun`."""

    def on_llm_start(self, messages: list[Any]) -> None:
        """Called before each provider ``generate`` / ``agenerate`` call."""

    def on_llm_end(self, result: GenerateResult) -> None:
        """Called after each provider ``generate`` / ``agenerate`` call."""

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        """Called before each tool invocation."""

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        """Called after each tool invocation (result is a string from Tool)."""

    def on_agent_end(self, result: Any) -> None:
        """Called once at the end of the run with the final :class:`AgentResult`."""


def _call_hook(callback: Any, hook: str, *args: Any) -> None:
    """Invoke ``callback.hook(*args)`` if ``callback`` actually provides it.

    The :class:`Callback` protocol uses default empty bodies, so an object
    that only implements a subset of hooks still satisfies the protocol. We
    check for a non-``Callback``-default attribute so partial implementations
    are honoured: if the object's attribute resolves to the empty
    ``Callback.<hook>`` method, we skip it (it would do nothing anyway, but
    this also avoids noisy logs for users who did not opt in).

    The check is a plain ``hasattr`` + ``callable`` test, not an
    ``isinstance`` of a private sentinel, so duck-typed callbacks (e.g. a
    plain class that defines ``on_agent_end`` only) are dispatched correctly.
    """
    method = getattr(callback, hook, None)
    if not callable(method):
        return
    # Skip the protocol's own empty default bodies so an object that merely
    # subclasses ``Callback`` without overriding anything stays silent.
    proto_method = getattr(Callback, hook, None)
    if getattr(method, "__func__", None) is proto_method:
        return
    method(*args)


def dispatch(callbacks: list[Callback] | None, hook: str, *args: Any) -> None:
    """Dispatch ``hook(*args)`` to every callback that implements it.

    A user-supplied callback that raises propagates the error (no silent
    swallowing); wrap in try/except in your callback if you want resilience.
    ``None`` / empty ``callbacks`` is a no-op.
    """
    if not callbacks:
        return
    for cb in callbacks:
        _call_hook(cb, hook, *args)


class LoggingCallback:
    """A :class:`Callback` that records every event as a log record.

    By default it logs at ``INFO`` level to the ``vertai.agent`` logger. A
    custom ``logger`` can be injected (e.g. for tests capturing records). The
    emitted records are also appended to :attr:`events` as
    ``(hook, payload_repr)`` tuples, which is handy for assertions without
    inspecting log capture machinery.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("vertai.agent")
        self.events: list[tuple[str, str]] = []

    def _emit(self, hook: str, payload: Any) -> None:
        self.events.append((hook, repr(payload)))
        self._logger.info("%s: %r", hook, payload)

    def on_agent_start(self, input: str) -> None:
        self._emit("on_agent_start", input)

    def on_llm_start(self, messages: list[Any]) -> None:
        self._emit("on_llm_start", f"{len(messages)} messages")

    def on_llm_end(self, result: GenerateResult) -> None:
        self._emit(
            "on_llm_end",
            f"tokens={result.total_tokens} tool_calls={len(result.tool_calls)}",
        )

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        self._emit("on_tool_start", f"{tool_name}({args})")

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        self._emit("on_tool_end", f"{tool_name} -> {result!r}")

    def on_agent_end(self, result: Any) -> None:
        # Avoid importing AgentResult at module load (circular: agent imports
        # callbacks). A duck-typed repr is sufficient for the log line.
        self._emit("on_agent_end", result)


class TokenCountCallback:
    """A :class:`Callback` that accumulates token usage across iterations.

    Sums :attr:`GenerateResult.total_tokens` (falling back to
    ``prompt_tokens + completion_tokens`` when ``total_tokens`` is unset) from
    every :meth:`on_llm_end` call. Exposes the running total via
    :attr:`total_tokens` and per-iteration counts via :attr:`per_iteration`.

    This lets an application report cost/usage without inspecting the
    :class:`AgentResult` directly (e.g. when streaming through a callback
    pipeline).
    """

    def __init__(self) -> None:
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.per_iteration: list[int] = []

    def on_llm_end(self, result: GenerateResult) -> None:
        total = result.total_tokens or (
            result.prompt_tokens + result.completion_tokens
        )
        self.total_tokens += total
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.per_iteration.append(total)

    def reset(self) -> Self:
        """Reset accumulated counters. Returns ``self`` for chaining."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.per_iteration = []
        return self


__all__ = [
    "Callback",
    "LoggingCallback",
    "TokenCountCallback",
    "dispatch",
]
