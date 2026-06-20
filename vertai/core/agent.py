"""Minimal tool-calling Agent loop.

The :class:`Agent` is the agent-SDK centrepiece for VertAI: it wires together
an :class:`~vertai.core.provider.LLMProvider` (S2) and a
:class:`~vertai.core.tool.ToolRegistry` (S4) into a real tool-calling loop and
enforces a hard ``max_iterations`` upper bound so a runaway model cannot spin
forever.

Loop shape (per ``docs/ARCHITECTURE.md`` section 3.7)::

    build messages (system + user input)
    for iteration in range(max_iterations):
        result = provider.generate(messages, tools=registry.to_specs())
        if not result.tool_calls:        # terminal: model is done
            break
        append assistant message (text + tool-call descriptor)
        for each tool_call:
            output = registry.call(name, args)   # Tool failure -> friendly
            append tool message(output)          #   string via failure handler
    else:
        truncated = True                # hit the upper bound

The same loop runs under :meth:`arun` using ``agenerate`` / ``acall``. Tool
failures are surfaced to the model as tool-result strings (S4's
``failure_error_function`` already handles this), so the agent never crashes
on a tool error -- the model gets a chance to recover.

``SessionMemory`` integration is intentionally optional and minimal in 1.0:
when supplied, the agent seeds the message list from the memory's history and
writes every produced message back so a later run can continue the
conversation. Full memory reliability (atomic write, real tokenizer) lands in
S9; here we only consume the existing stable API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from vertai.core.callbacks import Callback, dispatch
from vertai.core.memory import SessionMemory
from vertai.core.provider import (
    ChatMessage,
    GenerateResult,
    LLMProvider,
    ToolCall,
)
from vertai.core.tool import Tool, ToolRegistry


@dataclass
class AgentResult:
    """Outcome of an :class:`Agent` run.

    Attributes:
        final_output: The model's final text answer (last ``content``).
        tool_calls_history: Per-call records
            ``{iteration, tool_call, result}`` so callers can audit the
            sequence of tool invocations and their outputs.
        iterations: Number of provider generations executed (capped by
            ``max_iterations``).
        total_tokens: Sum of ``GenerateResult.total_tokens`` across
            iterations.
        elapsed_seconds: Wall-clock seconds from start to finish.
        truncated: ``True`` if the loop stopped because it hit
            ``max_iterations`` (the model kept requesting tool calls).
        finish_reason: The last observed ``GenerateResult.finish_reason``.
    """

    final_output: str
    tool_calls_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    truncated: bool = False
    finish_reason: str = "stop"


class Agent:
    """A minimal tool-calling agent.

    Args:
        provider: The :class:`LLMProvider` to drive the loop. Its
            ``generate`` / ``agenerate`` must accept ``tools=``.
        tools: Tools the agent can call. May be empty / ``None`` for a plain
            chat agent (the loop terminates after one generation).
        system_prompt: Optional system message prepended to every run.
        max_iterations: Hard upper bound on provider generations. Defaults to
            ``10``. Must be ``>= 1``.
        callbacks: Optional observability hooks (see :class:`Callback`).
        memory: Optional :class:`SessionMemory`. When provided, the agent
            seeds the conversation with prior history and writes each new
            message back. Full memory reliability is S9; here we only consume
            the stable API.

    The agent is single-turn with respect to ``input``: each
    :meth:`run` / :meth:`arun` call processes one user input through the
    tool-calling loop and returns an :class:`AgentResult`. Multi-turn state
    is carried via ``memory``.
    """

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        callbacks: list[Callback] | None = None,
        memory: SessionMemory | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}"
            )
        self._provider = provider
        self._registry = ToolRegistry(tools) if tools else ToolRegistry()
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._callbacks = list(callbacks) if callbacks else []
        self._memory = memory

    # -- introspection ----------------------------------------------------

    @property
    def provider(self) -> LLMProvider:
        """The backing LLM provider."""
        return self._provider

    @property
    def registry(self) -> ToolRegistry:
        """The agent's :class:`ToolRegistry` (mutable; callers may register
        more tools before a run)."""
        return self._registry

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def memory(self) -> SessionMemory | None:
        return self._memory

    # -- message construction --------------------------------------------

    def _seed_messages(self, input: str) -> list[ChatMessage]:
        """Build the initial message list for a run.

        Order: system prompt -> memory history -> user input. Memory history
        is replayed verbatim; we exclude ``system`` rows already stored in
        memory (the agent's own system prompt takes precedence to avoid
        prompt drift across runs).
        """
        messages: list[ChatMessage] = []
        if self._system_prompt:
            messages.append(
                ChatMessage(role="system", content=self._system_prompt)
            )
        if self._memory is not None:
            for prior in self._memory.get_history():
                if prior.role == "system":
                    continue
                messages.append(
                    ChatMessage(role=prior.role, content=prior.content)
                )
        messages.append(ChatMessage(role="user", content=input))
        return messages

    def _assistant_message(self, result: GenerateResult) -> ChatMessage:
        """Render a GenerateResult's assistant turn as a ChatMessage.

        The model's text goes into ``content``; any tool calls are summarised
        into a compact descriptor so a subsequent generation can see that the
        assistant asked for tools (and which ones). Provider-specific
        representations are normalised here so the loop is provider-agnostic.
        """
        parts: list[str] = []
        if result.content:
            parts.append(result.content)
        if result.tool_calls:
            calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in result.tool_calls
            ]
            parts.append(
                "[tool_calls] " + json.dumps(calls, ensure_ascii=False)
            )
        return ChatMessage(role="assistant", content="\n".join(parts))

    def _record_to_memory(self, message: ChatMessage) -> None:
        if self._memory is None:
            return
        # SessionMemory.add_message accepts only system/user/assistant; tool
        # results are folded into the assistant turn text instead of stored
        # as a separate role.
        if message.role in ("system", "user", "assistant"):
            self._memory.add_message(message.role, message.content)

    # -- tool execution --------------------------------------------------

    def _execute_tool_calls_sync(
        self,
        iteration: int,
        tool_calls: list[ToolCall],
    ) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
        """Run every tool call synchronously, returning the tool-result
        messages and a per-call history record."""
        new_messages: list[ChatMessage] = []
        history: list[dict[str, Any]] = []
        for tc in tool_calls:
            dispatch(self._callbacks, "on_tool_start", tc.name, tc.arguments)
            # ToolRegistry.call routes through Tool.execute, which already
            # applies the per-tool failure_error_function. A failed tool
            # therefore yields a friendly string, not an exception -- the
            # agent loop never crashes on a tool error.
            result = self._registry.call(tc.name, dict(tc.arguments))
            dispatch(self._callbacks, "on_tool_end", tc.name, result)
            record = {
                "iteration": iteration,
                "tool_call": {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                },
                "result": result,
            }
            history.append(record)
            msg = ChatMessage(role="tool", content=result)
            new_messages.append(msg)
            # Memory stores tool results as assistant text (no 'tool' role).
            if self._memory is not None:
                self._memory.add_message("assistant", f"[tool:{tc.name}] {result}")
        return new_messages, history

    async def _execute_tool_calls_async(
        self,
        iteration: int,
        tool_calls: list[ToolCall],
    ) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
        """Asynchronous counterpart of
        :meth:`_execute_tool_calls_sync`."""
        new_messages: list[ChatMessage] = []
        history: list[dict[str, Any]] = []
        for tc in tool_calls:
            dispatch(self._callbacks, "on_tool_start", tc.name, tc.arguments)
            result = await self._registry.acall(tc.name, dict(tc.arguments))
            dispatch(self._callbacks, "on_tool_end", tc.name, result)
            record = {
                "iteration": iteration,
                "tool_call": {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": dict(tc.arguments),
                },
                "result": result,
            }
            history.append(record)
            new_messages.append(ChatMessage(role="tool", content=result))
            if self._memory is not None:
                self._memory.add_message("assistant", f"[tool:{tc.name}] {result}")
        return new_messages, history

    # -- public API -------------------------------------------------------

    def run(self, input: str) -> AgentResult:
        """Run the synchronous tool-calling loop and return the result.

        The loop:
        1. Build messages (system + history + input).
        2. Call :meth:`LLMProvider.generate` with the tool specs.
        3. If the model returned tool calls, execute them and append the
           results, then loop again.
        4. Stop when the model returns no tool calls, or after
           ``max_iterations`` generations (``truncated=True``).
        """
        dispatch(self._callbacks, "on_agent_start", input)
        start = time.monotonic()
        messages = self._seed_messages(input)
        self._record_to_memory(ChatMessage(role="user", content=input))

        tool_specs = self._registry.to_specs()
        total_tokens = 0
        history: list[dict[str, Any]] = []
        iterations = 0
        truncated = False
        final_output = ""
        finish_reason = "stop"

        for iteration in range(1, self._max_iterations + 1):
            iterations = iteration
            dispatch(self._callbacks, "on_llm_start", list(messages))
            result = self._provider.generate(messages, tools=tool_specs)
            dispatch(self._callbacks, "on_llm_end", result)
            total_tokens += result.total_tokens or (
                result.prompt_tokens + result.completion_tokens
            )
            finish_reason = result.finish_reason
            final_output = result.content

            assistant_msg = self._assistant_message(result)
            messages.append(assistant_msg)
            self._record_to_memory(assistant_msg)

            if not result.tool_calls:
                # Model is done: no further tool calls requested.
                break

            tool_msgs, call_records = self._execute_tool_calls_sync(
                iteration, result.tool_calls
            )
            messages.extend(tool_msgs)
            history.extend(call_records)
        else:
            # The for-loop ran to exhaustion: the model kept requesting
            # tools through max_iterations. Mark truncated.
            truncated = True

        elapsed = time.monotonic() - start
        agent_result = AgentResult(
            final_output=final_output,
            tool_calls_history=history,
            iterations=iterations,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            truncated=truncated,
            finish_reason=finish_reason,
        )
        dispatch(self._callbacks, "on_agent_end", agent_result)
        return agent_result

    async def arun(self, input: str) -> AgentResult:
        """Asynchronous variant of :meth:`run`.

        Uses :meth:`LLMProvider.agenerate` and :meth:`ToolRegistry.acall`.
        The loop semantics (termination, truncation, token accounting,
        callbacks) are identical to the synchronous version.
        """
        dispatch(self._callbacks, "on_agent_start", input)
        start = time.monotonic()
        messages = self._seed_messages(input)
        self._record_to_memory(ChatMessage(role="user", content=input))

        tool_specs = self._registry.to_specs()
        total_tokens = 0
        history: list[dict[str, Any]] = []
        iterations = 0
        truncated = False
        final_output = ""
        finish_reason = "stop"

        for iteration in range(1, self._max_iterations + 1):
            iterations = iteration
            dispatch(self._callbacks, "on_llm_start", list(messages))
            result = await self._provider.agenerate(messages, tools=tool_specs)
            dispatch(self._callbacks, "on_llm_end", result)
            total_tokens += result.total_tokens or (
                result.prompt_tokens + result.completion_tokens
            )
            finish_reason = result.finish_reason
            final_output = result.content

            assistant_msg = self._assistant_message(result)
            messages.append(assistant_msg)
            self._record_to_memory(assistant_msg)

            if not result.tool_calls:
                break

            tool_msgs, call_records = await self._execute_tool_calls_async(
                iteration, result.tool_calls
            )
            messages.extend(tool_msgs)
            history.extend(call_records)
        else:
            truncated = True

        elapsed = time.monotonic() - start
        agent_result = AgentResult(
            final_output=final_output,
            tool_calls_history=history,
            iterations=iterations,
            total_tokens=total_tokens,
            elapsed_seconds=elapsed,
            truncated=truncated,
            finish_reason=finish_reason,
        )
        dispatch(self._callbacks, "on_agent_end", agent_result)
        return agent_result


__all__ = ["Agent", "AgentResult"]
