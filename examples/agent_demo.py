"""Agent demo: a tool-calling loop with the Agent abstraction (S5).

Shows the minimal viable VertAI agent: an :class:`LLMProvider` plus a set of
``@tool``-decorated functions are wired into an :class:`Agent`, which drives
the tool-calling loop itself (generate -> tool_use -> execute -> result ->
generate) until the model has no more tool calls or ``max_iterations`` is
reached.

Run: python examples/agent_demo.py

Without API credentials the demo falls back to a scripted fake provider so
the file is always runnable and the loop is always exercised. Set
VERTAI_API_KEY (and tweak the provider/model below) to hit a real model.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from vertai import (
    Agent,
    LLMConfig,
    LoggingCallback,
    ModelProvider,
    TokenCountCallback,
    create_provider,
    tool,
)
from vertai.core.provider import ChatMessage, GenerateResult, ToolCall


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Args:
        expression: A numeric expression such as ``2 + 3 * 4``.
    """
    # Tiny safe evaluator for the demo (vertai.core.tools.calculator is the
    # real, hardened one — used here only to keep the example dependency-free).
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def _ev(node: object) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_ev(node.operand)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expression, mode="eval")
    return str(_ev(tree.body))


def build_agent() -> Agent:
    """Build the demo Agent. Uses a real provider when VERTAI_API_KEY is set;
    otherwise returns an Agent backed by a scripted fake provider so the demo
    always runs end-to-end."""
    if os.environ.get("VERTAI_API_KEY"):
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            base_url="https://api.anthropic.com",
            model="claude-3-5-haiku-latest",
        )
        provider = create_provider(config)
    else:
        print("[demo] VERTAI_API_KEY not set; using a scripted fake provider.")
        provider = _ScriptedProvider(
            [
                GenerateResult(
                    content="",
                    model="fake",
                    total_tokens=5,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="calculator",
                            arguments={"expression": "6 * 7"},
                        )
                    ],
                ),
                GenerateResult(
                    content="The answer is 42.",
                    model="fake",
                    total_tokens=4,
                ),
            ]
        )
    return Agent(
        provider,
        [calculator],
        system_prompt="You are a helpful assistant. Use tools when useful.",
        max_iterations=5,
        callbacks=[LoggingCallback(), TokenCountCallback()],
    )


class _ScriptedProvider:
    """Minimal provider returning canned results in sequence. Only used when
    no real API key is available so the demo is always runnable."""

    def __init__(self, script: list[GenerateResult]) -> None:
        self._script = list(script)
        self._i = 0

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list | None = None,
        **kwargs: object,
    ) -> GenerateResult:
        result = self._script[self._i]
        self._i += 1
        return result

    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list | None = None,
        **kwargs: object,
    ) -> GenerateResult:
        return self.generate(messages, tools=tools, **kwargs)

    def stream(self, *a: object, **k: object) -> object:
        raise NotImplementedError

    def astream(self, *a: object, **k: object) -> object:
        raise NotImplementedError


def main() -> None:
    agent = build_agent()
    result = agent.run("What is six times seven?")
    print("\n--- AgentResult ---")
    print(f"final_output : {result.final_output!r}")
    print(f"iterations   : {result.iterations}")
    print(f"total_tokens : {result.total_tokens}")
    print(f"truncated    : {result.truncated}")
    print("tool history :")
    for call in result.tool_calls_history:
        tc = call["tool_call"]
        print(f"  iter={call['iteration']} {tc['name']}({tc['arguments']}) -> {call['result']!r}")


if __name__ == "__main__":
    main()
