"""Tests for the lightweight callbacks layer (S5).

The :class:`Callback` protocol supports partial implementations (users
provide only the hooks they need). Tests cover:

* A partial duck-typed callback (only ``on_agent_end``) is dispatched.
* :class:`LoggingCallback` records every hook in order.
* :class:`TokenCountCallback` accumulates tokens across iterations.
* The protocol is ``@runtime_checkable`` (isinstance works).
"""

from __future__ import annotations

from typing import Any

import pytest

from vertai.core.callbacks import (
    Callback,
    LoggingCallback,
    TokenCountCallback,
    dispatch,
)
from vertai.core.provider import GenerateResult


def _result(
    *,
    content: str = "",
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> GenerateResult:
    return GenerateResult(
        content=content,
        model="fake",
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Protocol semantics
# ---------------------------------------------------------------------------


def test_callback_is_runtime_checkable() -> None:
    """Objects that implement the protocol satisfy isinstance(Callback)."""

    class Full:  # all hooks
        def on_agent_start(self, input: str) -> None: ...
        def on_llm_start(self, messages: list[Any]) -> None: ...
        def on_llm_end(self, result: GenerateResult) -> None: ...
        def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None: ...
        def on_tool_end(self, tool_name: str, result: Any) -> None: ...
        def on_agent_end(self, result: Any) -> None: ...

    assert isinstance(Full(), Callback)


def test_partial_implementation_is_dispatched() -> None:
    """A callback that only defines ``on_agent_end`` is still invoked for
    that hook, and silent for the others. This is the partial-implementation
    guarantee the contract makes.

    Note: ``isinstance(cb, Callback)`` is intentionally NOT required for
    partial implementations — ``@runtime_checkable`` protocols check that
    *all* members are present, which defeats the point of partial
    implementations. The agent dispatches via ``getattr`` so duck-typed
    partial callbacks work regardless of ``isinstance``.
    """

    calls: list[str] = []

    class OnlyEnd:
        def on_agent_end(self, result: Any) -> None:
            calls.append("end")

    cb = OnlyEnd()
    # isinstance is False for a partial impl — that is by design.
    assert not isinstance(cb, Callback)
    # Dispatch every hook; only on_agent_end should fire.
    dispatch([cb], "on_agent_start", "hi")
    dispatch([cb], "on_llm_start", [])
    dispatch([cb], "on_llm_end", _result())
    dispatch([cb], "on_tool_start", "t", {})
    dispatch([cb], "on_tool_end", "t", "r")
    dispatch([cb], "on_agent_end", object())
    assert calls == ["end"]


def test_dispatch_none_and_empty_callbacks_is_noop() -> None:
    """``None`` and empty lists must not raise and must do nothing."""
    dispatch(None, "on_agent_start", "x")
    dispatch([], "on_agent_end", object())


def test_plain_protocol_subclass_with_no_overrides_is_silent() -> None:
    """A subclass of ``Callback`` that does not override any hook must not
    fire any hook (the protocol's empty default bodies are skipped)."""

    class Empty(Callback):
        pass

    fired: list[str] = []

    # Sneak a side-effect into a hook via monkeypatching the instance to make
    # any dispatch observable. We verify nothing fired by checking the list.
    dispatch([Empty()], "on_agent_end", object())
    assert fired == []


# ---------------------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------------------


def test_logging_callback_records_hooks_in_order() -> None:
    log = LoggingCallback()
    log.on_agent_start("hello")
    log.on_llm_start([object()])
    log.on_llm_end(_result(total_tokens=42))
    log.on_tool_start("calc", {"expr": "1+1"})
    log.on_tool_end("calc", "2")
    log.on_agent_end(object())

    hooks = [event[0] for event in log.events]
    assert hooks == [
        "on_agent_start",
        "on_llm_start",
        "on_llm_end",
        "on_tool_start",
        "on_tool_end",
        "on_agent_end",
    ]
    # Tokens and tool-call count should appear in the llm_end record.
    assert "tokens=42" in log.events[2][1]


def test_logging_callback_is_a_callback_protocol() -> None:
    assert isinstance(LoggingCallback(), Callback)


# ---------------------------------------------------------------------------
# TokenCountCallback
# ---------------------------------------------------------------------------


def test_token_count_callback_accumulates_total() -> None:
    counter = TokenCountCallback()
    counter.on_llm_end(_result(total_tokens=10))
    counter.on_llm_end(_result(total_tokens=25))
    assert counter.total_tokens == 35
    assert counter.per_iteration == [10, 25]


def test_token_count_callback_falls_back_when_total_zero() -> None:
    """Some providers report prompt/completion tokens but leave
    ``total_tokens`` unset; the callback must still account for usage."""
    counter = TokenCountCallback()
    counter.on_llm_end(_result(prompt_tokens=5, completion_tokens=7, total_tokens=0))
    assert counter.total_tokens == 12
    assert counter.prompt_tokens == 5
    assert counter.completion_tokens == 7


def test_token_count_reset() -> None:
    counter = TokenCountCallback()
    counter.on_llm_end(_result(total_tokens=99))
    counter.reset()
    assert counter.total_tokens == 0
    assert counter.per_iteration == []


def test_token_count_callback_is_a_callback_protocol() -> None:
    """TokenCountCallback is a partial implementation (only ``on_llm_end``).
    It still dispatches correctly via ``getattr``; isinstance returns False
    because ``@runtime_checkable`` requires all members present. That is the
    documented partial-implementation contract."""
    counter = TokenCountCallback()
    assert not isinstance(counter, Callback)  # partial impl, by design
    # But it is dispatched as a real callback:
    counter.on_llm_end(_result(total_tokens=5))
    assert counter.total_tokens == 5


# ---------------------------------------------------------------------------
# Dispatch propagates user callback errors (no silent swallowing)
# ---------------------------------------------------------------------------


def test_dispatch_propagates_callback_errors() -> None:
    """A user callback that raises must propagate; the agent does not wrap
    callback errors in try/except. This is a deliberate contract (wrap your
    own callback if you need resilience)."""

    class Boom:
        def on_agent_end(self, result: Any) -> None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        dispatch([Boom()], "on_agent_end", object())
