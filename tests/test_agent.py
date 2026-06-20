"""Tests for the Agent tool-calling loop (S5).

Testing strategy (per ROADMAP test table):
- The provider is a real fake (``_ScriptedProvider``) that returns canned
  ``GenerateResult`` values in sequence. No mock-of-a-mock: the agent really
  drives ``provider.generate`` / ``provider.agenerate``, the registry really
  invokes ``Tool.execute`` / ``Tool.aexecute``, and the resulting messages
  really flow back into the next ``generate`` call. This exercises the actual
  tool-calling loop end to end inside the unit test.
- Termination: a script that ends without tool_calls stops the loop
  (``truncated=False``); a script that always requests a tool is capped by
  ``max_iterations`` (``truncated=True``).
- Tool failure: a tool that raises surfaces as a friendly string (via S4's
  ``failure_error_function``) and does NOT crash the agent.
- Callbacks: a recording callback verifies hook order across a multi-turn run.
- Token accounting: token totals accumulate across iterations.
- Async: the same loop runs under ``arun`` with the async provider methods.
- Integration: a real ``@tool``-decorated ``calculator``-like function is
  resolved by the registry from a model-issued ToolCall.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vertai.core.agent import Agent
from vertai.core.callbacks import LoggingCallback, TokenCountCallback
from vertai.core.memory import SessionMemory
from vertai.core.provider import (
    ChatMessage,
    GenerateResult,
    LLMProvider,
    ToolCall,
    ToolSpec,
)
from vertai.core.tool import tool


# ---------------------------------------------------------------------------
# Scripted fake provider
# ---------------------------------------------------------------------------


class _ScriptedProvider(LLMProvider):
    """A minimal LLMProvider that returns canned ``GenerateResult`` values in
    sequence, recording the messages it was called with.

    ``generate`` and ``agenerate`` share the same script and append to the
    same call log, so a test can assert the loop really fed tool results back
    in.
    """

    def __init__(self, script: list[GenerateResult]) -> None:
        # Bypass LLMProvider.__init__ (no LLMConfig needed for a fake).
        self.config = None  # type: ignore[assignment]
        self._script = list(script)
        self._index = 0
        self.calls: list[list[ChatMessage]] = []

    def _next(self, messages: list[ChatMessage]) -> GenerateResult:
        self.calls.append(list(messages))
        if self._index >= len(self._script):
            raise AssertionError(
                f"ScriptedProvider exhausted: script had "
                f"{len(self._script)} entries, requested #{self._index + 1}"
            )
        result = self._script[self._index]
        self._index += 1
        return result

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        return self._next(messages)

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        return self._next(messages)

    def stream(
        self, messages: list[ChatMessage], *, tools: list[ToolSpec] | None = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def astream(
        self, messages: list[ChatMessage], *, tools: list[ToolSpec] | None = None, **kwargs: Any
    ) -> Any:
        raise NotImplementedError


def _gr(
    *,
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    total_tokens: int = 0,
    finish_reason: str = "stop",
) -> GenerateResult:
    return GenerateResult(
        content=content,
        model="fake",
        tool_calls=tool_calls or [],
        total_tokens=total_tokens,
        finish_reason=finish_reason,
    )


# A simple real tool used across tests. @tool auto-generates the schema.


@tool
def echo(x: str) -> str:
    """Echo the argument back, uppercased.

    Args:
        x: The value to echo.
    """
    return x.upper()


@tool
def boom() -> str:
    """Always raises ValueError to exercise failure handling."""
    raise ValueError("boom")


# ---------------------------------------------------------------------------
# Loop execution: multi-turn tool calling
# ---------------------------------------------------------------------------


def test_run_executes_multi_turn_tool_loop() -> None:
    """Iteration 1: model calls echo('hi'). Iteration 2: model returns final
    text. We verify the tool was really invoked (history carries the result)
    and the tool output was fed back as a 'tool' message before iteration 2."""
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"x": "hi"})
                ],
                total_tokens=5,
            ),
            _gr(content="HI was echoed", total_tokens=7),
        ]
    )
    agent = Agent(provider, [echo])

    result = agent.run("go")

    assert result.iterations == 2
    assert not result.truncated
    assert result.final_output == "HI was echoed"
    assert result.total_tokens == 12
    # The tool was really executed: registry produced "HI".
    assert len(result.tool_calls_history) == 1
    call = result.tool_calls_history[0]
    assert call["tool_call"]["name"] == "echo"
    assert call["result"] == "HI"
    assert call["iteration"] == 1

    # Second generate() call must include the tool result message.
    second_messages = provider.calls[1]
    tool_msgs = [m for m in second_messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "HI"
    # The assistant's tool-call descriptor must precede the tool result.
    assistant_msgs = [m for m in second_messages if m.role == "assistant"]
    assert assistant_msgs  # at least the assistant turn with [tool_calls]


def test_run_terminates_immediately_without_tool_calls() -> None:
    provider = _ScriptedProvider([_gr(content="done", total_tokens=3)])
    agent = Agent(provider, [echo])

    result = agent.run("hi")

    assert result.iterations == 1
    assert not result.truncated
    assert result.final_output == "done"
    assert result.tool_calls_history == []
    assert provider.calls and provider.calls[0][0].role == "user"


def test_multiple_tool_calls_in_one_iteration() -> None:
    """A single generation may request several tool calls; all are executed
    and all results are appended before the next generation."""
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"x": "a"}),
                    ToolCall(id="c2", name="echo", arguments={"x": "b"}),
                ],
                total_tokens=4,
            ),
            _gr(content="final", total_tokens=2),
        ]
    )
    agent = Agent(provider, [echo])

    result = agent.run("two")

    assert result.iterations == 2
    assert [c["result"] for c in result.tool_calls_history] == ["A", "B"]
    # The second generation saw both tool messages.
    second = provider.calls[1]
    assert [m.content for m in second if m.role == "tool"] == ["A", "B"]


# ---------------------------------------------------------------------------
# max_iterations (runaway guard)
# ---------------------------------------------------------------------------


def test_run_caps_at_max_iterations_and_marks_truncated() -> None:
    """A model that always requests a tool must be stopped at
    ``max_iterations`` with ``truncated=True``."""
    # Provide more script entries than max_iterations; the loop must stop early.
    always_tool = _gr(
        tool_calls=[ToolCall(id="c", name="echo", arguments={"x": "z"})],
        total_tokens=1,
    )
    script = [always_tool] * 50
    provider = _ScriptedProvider(script)
    agent = Agent(provider, [echo], max_iterations=3)

    result = agent.run("loop")

    assert result.iterations == 3
    assert result.truncated is True
    assert len(result.tool_calls_history) == 3
    # All three iterations actually invoked the tool.
    assert all(c["result"] == "Z" for c in result.tool_calls_history)


def test_max_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Agent(_ScriptedProvider([]), [echo], max_iterations=0)


def test_max_iterations_one_runs_single_turn() -> None:
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"x": "q"})
                ],
                total_tokens=1,
            )
        ]
    )
    agent = Agent(provider, [echo], max_iterations=1)
    result = agent.run("one")
    assert result.iterations == 1
    # The model requested a tool but we only had one iteration, so truncated.
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Tool failure does not crash the agent
# ---------------------------------------------------------------------------


def test_tool_failure_surfaces_friendly_message_and_loop_continues() -> None:
    """When a tool raises, S4's default failure_error_function returns a
    friendly string. The agent loop must keep running rather than crashing."""
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="boom", arguments={})],
                total_tokens=2,
            ),
            _gr(content="recovered", total_tokens=2),
        ]
    )
    agent = Agent(provider, [boom])

    result = agent.run("fail")

    assert result.iterations == 2
    assert result.final_output == "recovered"
    assert len(result.tool_calls_history) == 1
    # The default friendly handler stringifies the exception for the model.
    assert "failed" in result.tool_calls_history[0]["result"]
    assert "boom" in result.tool_calls_history[0]["result"]


# ---------------------------------------------------------------------------
# Callbacks fire in the right order
# ---------------------------------------------------------------------------


def test_callbacks_fire_in_order() -> None:
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "y"})],
                total_tokens=1,
            ),
            _gr(content="done", total_tokens=1),
        ]
    )
    log = LoggingCallback()
    agent = Agent(provider, [echo], callbacks=[log])

    agent.run("start")

    hooks = [name for name, _ in log.events]
    # Expected sequence: agent_start, llm_start, llm_end, tool_start,
    # tool_end, llm_start, llm_end, agent_end.
    assert hooks == [
        "on_agent_start",
        "on_llm_start",
        "on_llm_end",
        "on_tool_start",
        "on_tool_end",
        "on_llm_start",
        "on_llm_end",
        "on_agent_end",
    ]


def test_token_count_callback_accumulates_via_agent() -> None:
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "y"})],
                total_tokens=10,
            ),
            _gr(content="done", total_tokens=20),
        ]
    )
    counter = TokenCountCallback()
    agent = Agent(provider, [echo], callbacks=[counter])

    result = agent.run("count")
    assert result.total_tokens == 30
    assert counter.total_tokens == 30
    assert counter.per_iteration == [10, 20]


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


def test_arun_runs_async_tool_loop() -> None:
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "hi"})],
                total_tokens=4,
            ),
            _gr(content="async done", total_tokens=6),
        ]
    )
    agent = Agent(provider, [echo])

    result = asyncio.run(agent.arun("go"))

    assert result.iterations == 2
    assert not result.truncated
    assert result.final_output == "async done"
    assert result.tool_calls_history[0]["result"] == "HI"


def test_arun_respects_max_iterations() -> None:
    always_tool = _gr(
        tool_calls=[ToolCall(id="c", name="echo", arguments={"x": "z"})],
        total_tokens=1,
    )
    provider = _ScriptedProvider([always_tool] * 20)
    agent = Agent(provider, [echo], max_iterations=2)

    result = asyncio.run(agent.arun("loop"))

    assert result.iterations == 2
    assert result.truncated is True


# ---------------------------------------------------------------------------
# System prompt + memory integration
# ---------------------------------------------------------------------------


def test_system_prompt_is_prepended() -> None:
    provider = _ScriptedProvider([_gr(content="ok", total_tokens=1)])
    agent = Agent(
        provider, [echo], system_prompt="You are a calculator."
    )

    agent.run("hi")

    first_call = provider.calls[0]
    assert first_call[0].role == "system"
    assert first_call[0].content == "You are a calculator."


def test_memory_seeds_and_records_conversation() -> None:
    memory = SessionMemory()
    memory.add_message("assistant", "prior answer")
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "h"})],
                total_tokens=1,
            ),
            _gr(content="final", total_tokens=1),
        ]
    )
    agent = Agent(provider, [echo], memory=memory)

    agent.run("next")

    # The model saw the prior assistant turn and the new user turn.
    first = provider.calls[0]
    contents = [(m.role, m.content) for m in first]
    assert ("assistant", "prior answer") in contents
    assert ("user", "next") in contents

    # Memory now contains: prior assistant, new user, tool result (stored as
    # assistant), assistant tool-call descriptor, and final assistant.
    stored = [(m.role, m.content) for m in memory.get_history()]
    assert ("user", "next") in stored
    # Tool result folded into an assistant message tagged [tool:echo].
    assert any("H" in c and r == "assistant" for r, c in stored)


# ---------------------------------------------------------------------------
# No-tools agent (plain chat)
# ---------------------------------------------------------------------------


def test_agent_without_tools_runs_single_turn() -> None:
    provider = _ScriptedProvider([_gr(content="hello there", total_tokens=2)])
    agent = Agent(provider)  # no tools

    result = agent.run("hi")

    assert result.iterations == 1
    assert result.final_output == "hello there"
    assert result.tool_calls_history == []
    # to_specs() on an empty registry is [], passed to the provider.
    assert provider.calls


def test_agent_registry_is_mutable_before_run() -> None:
    """Tools can be added to a previously-constructed agent's registry."""
    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"x": "d"})],
                total_tokens=1,
            ),
            _gr(content="ok", total_tokens=1),
        ]
    )
    agent = Agent(provider)  # empty
    agent.registry.register(echo)

    result = agent.run("d")
    assert result.tool_calls_history[0]["result"] == "D"


# ---------------------------------------------------------------------------
# Integration: real @tool calculator-like function executes real math
# ---------------------------------------------------------------------------


def test_integration_real_calculator_tool() -> None:
    """A real ``@tool``-decorated calculator is invoked end-to-end: the fake
    model issues a ToolCall naming it, the registry resolves it, and the real
    function runs."""

    @tool
    def add(a: int, b: int) -> str:
        """Add two integers.

        Args:
            a: First addend.
            b: Second addend.
        """
        return str(a + b)

    provider = _ScriptedProvider(
        [
            _gr(
                tool_calls=[
                    ToolCall(id="c1", name="add", arguments={"a": 40, "b": 2})
                ],
                total_tokens=3,
            ),
            _gr(content="The answer is 42", total_tokens=4),
        ]
    )
    agent = Agent(provider, [add])

    result = agent.run("what is 40+2?")

    assert result.iterations == 2
    assert result.final_output == "The answer is 42"
    assert result.tool_calls_history[0]["result"] == "42"
