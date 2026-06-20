"""Tests for the Tool abstraction and built-in tools (S4).

Testing strategy (per ROADMAP test table):
- Pure logic (schema generation, docstring parsing, calculator safe_eval,
  path resolution) -> real assertions, no mocking.
- Tool execution / timeout / failure handling -> real calls against real
  functions; timeout uses a real sleeping function with a short budget.
- Tool calling end-to-end -> a fake LLMProvider that returns a hand-written
  GenerateResult with tool_calls; the registry resolves and invokes the
  FunctionTool for real.
- HTTP built-in tools -> httpx.MockTransport stubs the wire so the request
  construction and response parsing are verified against real-shaped payloads.
- File built-in tools -> tmp_path real filesystem, including traversal guard.
- Calculator -> real safe_eval against both valid and adversarial inputs.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from pydantic import Field
from typing_extensions import Annotated

from vertai.core.provider import (
    ChatMessage,
    GenerateResult,
    LLMProvider,
    ToolCall,
    ToolSpec,
)
from vertai.core.tool import (
    FunctionTool,
    ToolError,
    ToolRegistry,
    ToolTimeoutError,
    tool,
)
from vertai.core.tools.calculator import (
    UnsafeExpressionError,
    calculator,
    safe_eval,
)
from vertai.core.tools.file import (
    PathTraversalError,
    make_file_read_tool,
    make_file_write_tool,
)
from vertai.core.tools.http import make_http_request_tool
from vertai.core.tools.web_search import make_web_search_tool


# ---------------------------------------------------------------------------
# Schema generation (the heart of the @tool decorator)
# ---------------------------------------------------------------------------


def test_annotated_field_constraints_propagate_to_schema() -> None:
    """The headline feature: ``Annotated[int, Field(ge=0, le=100)]`` must
    appear in the generated JSON Schema."""

    @tool
    def paginate(
        query: Annotated[str, Field(description="Search query", min_length=1)],
        limit: Annotated[int, Field(ge=1, le=100, description="Max results")] = 10,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> str:
        """Paginate search results.

        Args:
            query: The search query.
            limit: Maximum results per page.
            offset: Skip this many results.
        """
        return f"{query} {limit} {offset}"

    schema = paginate.parameters
    assert schema["type"] == "object"
    props = schema["properties"]
    # query: required, min_length, description
    assert "query" in schema["required"]
    assert props["query"]["minLength"] == 1
    assert props["query"]["description"] == "Search query"
    # limit: optional (has default), ge/le present; explicit Field(description)
    # takes precedence over the docstring (OpenAI Agents SDK behaviour).
    assert "limit" not in schema.get("required", [])
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 100
    assert props["limit"]["default"] == 10
    assert props["limit"]["description"] == "Max results"
    # offset: optional, ge present; description falls back to the docstring
    # because no Field(description=...) is supplied.
    assert props["offset"]["minimum"] == 0
    assert props["offset"]["default"] == 0
    assert props["offset"]["description"] == "Skip this many results."


def test_default_as_field_form_constraints_propagate() -> None:
    """``x: int = Field(0.5, ge=0.0, le=1.0)`` must also carry constraints."""

    @tool
    def threshold(score: float = Field(0.5, ge=0.0, le=1.0)) -> str:
        """Apply a threshold.

        Args:
            score: Cutoff in [0, 1].
        """
        return f"{score}"

    props = threshold.parameters["properties"]["score"]
    assert props["minimum"] == 0.0
    assert props["maximum"] == 1.0
    assert props["default"] == 0.5


def test_plain_defaults_are_optional() -> None:
    """A plain ``= 5`` default with no Field metadata still yields an
    optional schema entry with a default."""

    @tool
    def echo(a: int, b: int = 5, c: str = "hi") -> str:
        """Echo.

        Args:
            a: required.
            b: optional int.
            c: optional str.
        """
        return f"{a}/{b}/{c}"

    schema = echo.parameters
    assert schema["required"] == ["a"]
    assert schema["properties"]["b"]["default"] == 5
    assert schema["properties"]["c"]["default"] == "hi"


def test_name_and_description_overrides() -> None:
    @tool(name="custom_name", description="Custom description.")
    def func(x: int) -> str:
        """Original docstring (should be overridden)."""
        return str(x)

    assert func.name == "custom_name"
    assert func.description == "Custom description."


def test_description_falls_back_to_docstring_summary() -> None:
    @tool
    def greet(name: str) -> str:
        """Say hello to someone.

        Args:
            name: Their name.
        """
        return f"hi {name}"

    assert greet.description == "Say hello to someone."


def test_to_spec_emits_tool_spec_matching_provider_contract() -> None:
    @tool
    def add(a: int, b: int) -> str:
        """Add two numbers.

        Args:
            a: First.
            b: Second.
        """
        return str(a + b)

    spec = add.to_spec()
    assert isinstance(spec, ToolSpec)
    assert spec.name == "add"
    assert spec.description == "Add two numbers."
    assert spec.input_schema["type"] == "object"
    # The provider expects a JSON Schema under input_schema.
    assert set(spec.input_schema["properties"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Docstring parsing (Google / NumPy / Sphinx)
# ---------------------------------------------------------------------------


def test_google_style_docstring_parsing() -> None:
    @tool
    def t(a: int, b: str) -> str:
        """Summary line.

        Args:
            a: First parameter description.
            b: Second parameter with
                continuation.
        """
        return f"{a}{b}"

    props = t.parameters["properties"]
    assert props["a"]["description"] == "First parameter description."
    assert props["b"]["description"] == (
        "Second parameter with continuation."
    )


def test_numpy_style_docstring_parsing() -> None:
    @tool
    def t(a: int, b: str) -> str:
        """Summary line.

        Parameters
        ----------
        a : int
            First parameter description.
        b : str
            Second parameter description.
        """
        return f"{a}{b}"

    props = t.parameters["properties"]
    assert props["a"]["description"] == "First parameter description."
    assert props["b"]["description"] == "Second parameter description."


def test_sphinx_style_docstring_parsing() -> None:
    @tool
    def t(a: int, b: str) -> str:
        """Summary line.

        :param a: First parameter description.
        :param b: Second parameter description.
        """
        return f"{a}{b}"

    props = t.parameters["properties"]
    assert props["a"]["description"] == "First parameter description."
    assert props["b"]["description"] == "Second parameter description."


def test_no_docstring_does_not_break_tool_creation() -> None:
    @tool
    def t(a: int) -> str:
        return str(a)

    # Falls back to a generic description; schema still works.
    assert t.name == "t"
    assert "t" in t.description
    assert t.parameters["properties"]["a"]["type"] == "integer"


# ---------------------------------------------------------------------------
# execute / aexecute
# ---------------------------------------------------------------------------


def test_execute_returns_string_and_validates_args() -> None:
    @tool
    def double(x: int) -> str:
        """Double.

        Args:
            x: A number.
        """
        return x * 2  # type: ignore[return-value]

    # int 4 -> str via _stringify_result
    assert double.execute(x=4) == "8"
    # Out-of-range constraint -> default failure handler returns a message
    @tool
    def clamped(
        n: Annotated[int, Field(ge=0, le=10)] = 0,  # noqa: B008
    ) -> str:
        """Clamped.

        Args:
            n: A number.
        """
        return str(n)

    result = clamped.execute(n=99)
    assert "failed" in result.lower() and "clamped" in result


def test_aexecute_runs_sync_function_in_executor() -> None:
    @tool
    def add(a: int, b: int) -> str:
        """Add.

        Args:
            a: First.
            b: Second.
        """
        return str(a + b)

    result = asyncio.run(add.aexecute(a=2, b=3))
    assert result == "5"


def test_aexecute_awaits_async_function() -> None:
    @tool
    async def slow_double(x: int) -> str:
        """Async double.

        Args:
            x: A number.
        """
        await asyncio.sleep(0)
        return str(x * 2)

    result = asyncio.run(slow_double.aexecute(x=21))
    assert result == "42"


def test_result_stringification_handles_various_types() -> None:
    @tool
    def returns_none() -> str:  # type: ignore[empty-body]
        """Returns nothing."""

    @tool
    def returns_list() -> str:  # type: ignore[empty-body]
        """Returns a list."""

    # Override the functions to return typed payloads via direct FunctionTool.
    t_none = FunctionTool(lambda: None, name_override="n")  # type: ignore[arg-type]
    t_list = FunctionTool(lambda: [1, 2, 3], name_override="l")  # type: ignore[arg-type]
    t_bytes = FunctionTool(lambda: b"raw", name_override="b")  # type: ignore[arg-type]

    assert t_none.execute() == ""
    assert json.loads(t_list.execute()) == [1, 2, 3]
    assert t_bytes.execute() == "raw"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_timeout_error_as_result_returns_message() -> None:
    @tool(timeout=0.15)
    def slow() -> str:
        """Slow tool."""
        time.sleep(0.5)
        return "done"

    start = time.monotonic()
    result = slow.execute()
    elapsed = time.monotonic() - start
    assert "timed out" in result.lower()
    # Should not have waited the full sleep.
    assert elapsed < 0.5


def test_timeout_raise_exception_raises_tool_timeout_error() -> None:
    @tool(timeout=0.15, timeout_mode="raise_exception")
    def slow() -> str:
        """Slow tool."""
        time.sleep(0.5)
        return "done"

    with pytest.raises(ToolTimeoutError):
        slow.execute()


def test_async_timeout_error_as_result_returns_message() -> None:
    @tool(timeout=0.15)
    async def slow() -> str:
        """Slow tool."""
        await asyncio.sleep(0.5)
        return "done"

    result = asyncio.run(slow.aexecute())
    assert "timed out" in result.lower()


def test_async_timeout_raise_exception_raises_tool_timeout_error() -> None:
    @tool(timeout=0.15, timeout_mode="raise_exception")
    async def slow() -> str:
        """Slow tool."""
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(ToolTimeoutError):
        asyncio.run(slow.aexecute())


def test_timeout_mode_validation() -> None:
    with pytest.raises(ValueError):

        @tool(timeout=1.0, timeout_mode="bogus")  # type: ignore[arg-type]
        def t() -> str:
            """T."""
            return "x"


# ---------------------------------------------------------------------------
# failure_error_function
# ---------------------------------------------------------------------------


def test_default_failure_handler_returns_friendly_message() -> None:
    @tool
    def boom() -> str:
        """Boom."""
        raise ValueError("kaput")

    result = boom.execute()
    assert "failed" in result.lower()
    assert "ValueError" in result
    assert "kaput" in result


def test_custom_failure_handler() -> None:
    @tool(failure_error_function=lambda exc: f"CUSTOM: {exc}")
    def boom() -> str:
        """Boom."""
        raise ValueError("kaput")

    assert boom.execute() == "CUSTOM: kaput"


def test_failure_handler_none_reraises() -> None:
    @tool(failure_error_function=None)
    def boom() -> str:
        """Boom."""
        raise ValueError("kaput")

    with pytest.raises(ValueError, match="kaput"):
        boom.execute()


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def _make_tool(name: str) -> FunctionTool:
    return FunctionTool(
        lambda: name, name_override=name, description_override=name
    )


def test_registry_register_get_and_to_specs() -> None:
    a, b = _make_tool("a"), _make_tool("b")
    reg = ToolRegistry([a, b])
    assert len(reg) == 2
    assert reg.get("a") is a
    assert reg.get("b") is b
    specs = reg.to_specs()
    # Deterministic name order.
    assert [s.name for s in specs] == ["a", "b"]
    assert all(isinstance(s, ToolSpec) for s in specs)


def test_registry_duplicate_name_raises() -> None:
    reg = ToolRegistry([_make_tool("x")])
    with pytest.raises(ToolError, match="already registered"):
        reg.register(_make_tool("x"))


def test_registry_get_unknown_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError, match="No tool"):
        reg.get("nope")


def test_registry_unregister_and_contains() -> None:
    t = _make_tool("x")
    reg = ToolRegistry([t])
    assert "x" in reg
    assert reg.unregister("x") is t
    assert "x" not in reg
    assert reg.unregister("missing") is None


def test_registry_call_and_acall_resolve_tool_calls() -> None:
    @tool
    def greet(name: str, excited: bool = False) -> str:
        """Greet.

        Args:
            name: Name.
            excited: Exclaim?
        """
        suffix = "!" if excited else "."
        return f"hi {name}{suffix}"

    reg = ToolRegistry([greet])
    assert reg.call("greet", {"name": "world", "excited": True}) == "hi world!"
    assert asyncio.run(reg.acall("greet", {"name": "ann"})) == "hi ann."


def test_registry_only_accepts_tool_instances() -> None:
    reg = ToolRegistry()
    with pytest.raises(TypeError):
        reg.register("not a tool")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tool calling end-to-end: fake LLM -> tool_use -> execute
# ---------------------------------------------------------------------------


class _FakeProvider(LLMProvider):
    """A minimal LLMProvider that returns canned tool calls so the registry
    can resolve and execute them. No HTTP involved."""

    def __init__(self, canned: GenerateResult) -> None:
        # Bypass LLMProvider.__init__ to avoid needing an LLMConfig.
        self.config = None  # type: ignore[assignment]
        self._canned = canned

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult:
        return self._canned

    def stream(self, messages: list[ChatMessage], *, tools: list[ToolSpec] | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def agenerate(self, messages: list[ChatMessage], *, tools: list[ToolSpec] | None = None, **kwargs: Any) -> GenerateResult:
        return self._canned

    def astream(self, messages: list[ChatMessage], *, tools: list[ToolSpec] | None = None, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_tool_calling_end_to_end_sync() -> None:
    @tool
    def add(a: int, b: int) -> str:
        """Add.

        Args:
            a: First.
            b: Second.
        """
        return str(a + b)

    registry = ToolRegistry([add])
    # The "model" decides to call add(2, 3).
    fake = _FakeProvider(
        GenerateResult(
            content="",
            model="fake",
            tool_calls=[
                ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})
            ],
        )
    )

    result = fake.generate([ChatMessage(role="user", content="add")], tools=registry.to_specs())
    assert result.tool_calls
    tc = result.tool_calls[0]
    # Registry resolves the ToolCall and runs the real tool.
    output = registry.call(tc.name, tc.arguments)
    assert output == "5"


def test_tool_calling_end_to_end_async() -> None:
    @tool
    async def mul(a: int, b: int) -> str:
        """Multiply.

        Args:
            a: First.
            b: Second.
        """
        return str(a * b)

    registry = ToolRegistry([mul])
    fake = _FakeProvider(
        GenerateResult(
            content="",
            model="fake",
            tool_calls=[
                ToolCall(id="call_1", name="mul", arguments={"a": 6, "b": 7})
            ],
        )
    )

    async def _run() -> str:
        result = await fake.agenerate(
            [ChatMessage(role="user", content="mul")], tools=registry.to_specs()
        )
        tc = result.tool_calls[0]
        return await registry.acall(tc.name, tc.arguments)

    assert asyncio.run(_run()) == "42"


# ---------------------------------------------------------------------------
# Built-in: calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("10 / 4", "2.5"),
        ("10 // 3", "3"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("7 % 3", "1"),
        ("sqrt(16)", "4.0"),
        ("pow(2, 3) + abs(-5)", "13"),
        ("round(3.7)", "4"),
        ("min(1, 2, 3)", "1"),
        ("max(1, 2, 3)", "3"),
        ("floor(3.9)", "3"),
        ("ceil(3.1)", "4"),
        ("pi", str(3.141592653589793)),
        ("2 * pi", str(2 * 3.141592653589793)),
        ("log(e)", "1.0"),
        ("log2(8)", "3.0"),
        ("log10(1000)", "3.0"),
    ],
)
def test_calculator_valid_expressions(expr: str, expected: str) -> None:
    assert calculator.execute(expression=expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",       # call to non-whitelisted name
        "open('x')",              # name not whitelisted
        "os.system('rm -rf')",    # attribute access (parsed as Call on Attribute)
        "(1).__class__",          # attribute access on literal
        "[x for x in range(3)]",  # comprehension
        "x = 5",                  # assignment (statement, not expr)
        "import os",              # import statement
        "lambda x: x",            # lambda
        "1 if True else 2",       # bool/trinary not in whitelist
        "{'a': 1}",               # dict literal
        "[1, 2, 3]",              # list literal
        "('a'.upper)()",          # method call via attr
    ],
)
def test_calculator_rejects_dangerous_inputs(expr: str) -> None:
    # safe_eval should raise UnsafeExpressionError; the tool's failure handler
    # turns it into a friendly message.
    with pytest.raises(UnsafeExpressionError):
        safe_eval(expr)
    result = calculator.execute(expression=expr)
    assert "Cannot evaluate" in result


def test_calculator_rejects_empty_input() -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_eval("")
    with pytest.raises(UnsafeExpressionError):
        safe_eval("   ")


# ---------------------------------------------------------------------------
# Built-in: file_read / file_write (real tmp_path)
# ---------------------------------------------------------------------------


def test_file_write_then_read_roundtrip(tmp_path: Path) -> None:
    wr = make_file_write_tool(base_dir=str(tmp_path))
    rd = make_file_read_tool(base_dir=str(tmp_path))

    write_result = wr.execute(path="notes/hello.txt", content="hello world")
    assert "Wrote" in write_result
    assert (tmp_path / "notes" / "hello.txt").read_text(encoding="utf-8") == "hello world"

    read_result = rd.execute(path="notes/hello.txt")
    assert read_result == "hello world"


def test_file_read_missing_file_reports_error(tmp_path: Path) -> None:
    rd = make_file_read_tool(base_dir=str(tmp_path))
    result = rd.execute(path="nope.txt")
    assert "failed" in result.lower()
    assert "no such file" in result.lower()


def test_file_read_enforces_byte_limit(tmp_path: Path) -> None:
    rd = make_file_read_tool(base_dir=str(tmp_path), max_bytes=10)
    (tmp_path / "big.txt").write_text("x" * 100, encoding="utf-8")
    result = rd.execute(path="big.txt")
    assert "failed" in result.lower()
    assert "exceeds" in result.lower()


def test_file_tools_block_path_traversal(tmp_path: Path) -> None:
    wr = make_file_write_tool(base_dir=str(tmp_path))
    # Relative traversal escapes base_dir.
    result = wr.execute(path="../../escape.txt", content="evil")
    assert "PathTraversalError" in result
    # Absolute path outside base_dir is also blocked.
    outside = tmp_path.parent.parent / "evil.txt"
    result = wr.execute(path=str(outside), content="evil")
    assert "PathTraversalError" in result
    # The escaped file must NOT exist.
    assert not (tmp_path.parent.parent / "escape.txt").exists()


def test_file_tools_direct_traversal_error(tmp_path: Path) -> None:
    """The underlying _resolve_and_check raises PathTraversalError directly
    (not via the failure handler)."""
    from vertai.core.tools.file import _resolve_and_check

    with pytest.raises(PathTraversalError):
        _resolve_and_check("../../escape", str(tmp_path))


def test_file_tools_without_base_dir_allow_absolute_paths(tmp_path: Path) -> None:
    """Without base_dir the tool is unconstrained (caller-sandboxed)."""
    from vertai.core.tools.file import file_read, file_write

    target = tmp_path / "free.txt"
    result = file_write.execute(path=str(target), content="ok")
    assert "Wrote" in result
    assert file_read.execute(path=str(target)) == "ok"


# ---------------------------------------------------------------------------
# Built-in: http_request (httpx.MockTransport)
# ---------------------------------------------------------------------------


def _mock_handler(body: bytes, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers={"X-Test": "yes"})
    return handler


def test_http_request_get_returns_json_summary() -> None:
    t = make_http_request_tool(max_body_bytes=1024)

    import sys

    http_mod = sys.modules["vertai.core.tools.http"]
    original_client = http_mod.httpx.Client

    def _fake_client(**kwargs: Any) -> httpx.Client:
        return original_client(
            transport=httpx.MockTransport(_mock_handler(b"hello body")),
            **{k: v for k, v in kwargs.items() if k != "transport"},
        )

    http_mod.httpx.Client = _fake_client  # type: ignore[assignment]
    try:
        result = t.execute(
            url="https://example.com/api", method="GET", params={"q": "1"}
        )
    finally:
        http_mod.httpx.Client = original_client  # type: ignore[assignment]

    parsed = json.loads(result)
    assert parsed["status_code"] == 200
    assert parsed["body"] == "hello body"
    assert parsed["headers"]["x-test"] == "yes"


def test_http_request_rejects_unknown_method() -> None:
    t = make_http_request_tool()
    result = t.execute(url="https://example.com", method="TRACE")
    assert "not allowed" in result.lower()


def test_http_request_async_uses_async_client() -> None:
    """Async path uses a real httpx.AsyncClient (verified by swapping the
    AsyncClient constructor for one with a MockTransport)."""
    import sys

    http_mod = sys.modules["vertai.core.tools.http"]
    original = http_mod.httpx.AsyncClient

    def _fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        return original(
            transport=httpx.MockTransport(_mock_handler(b"async body")),
            **{k: v for k, v in kwargs.items() if k != "transport"},
        )

    http_mod.httpx.AsyncClient = _fake_async_client  # type: ignore[assignment]
    try:
        result = asyncio.run(
            t_async_request(
                url="https://example.com", method="GET"
            )
        )
    finally:
        http_mod.httpx.AsyncClient = original  # type: ignore[assignment]

    parsed = json.loads(result)
    assert parsed["body"] == "async body"


def t_async_request(url: str, method: str) -> Any:
    return make_http_request_tool().aexecute(url=url, method=method)


# ---------------------------------------------------------------------------
# Built-in: web_search (httpx.MockTransport)
# ---------------------------------------------------------------------------


_DDGO_SAMPLE = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F1">Result One</a>
  <a class="result__snippet" href="">First snippet text</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F2">Result Two</a>
  <a class="result__snippet" href="">Second snippet text</a>
</div>
"""


def test_web_search_parses_results() -> None:
    import sys

    ws_mod = sys.modules["vertai.core.tools.web_search"]
    original = ws_mod.httpx.Client

    def _fake_client(**kwargs: Any) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_DDGO_SAMPLE.encode("utf-8"))
        return original(transport=httpx.MockTransport(handler))

    ws_mod.httpx.Client = _fake_client  # type: ignore[assignment]
    try:
        t = make_web_search_tool(max_results=5)
        result = t.execute(query="test")
    finally:
        ws_mod.httpx.Client = original  # type: ignore[assignment]

    parsed = json.loads(result)
    assert parsed["query"] == "test"
    assert len(parsed["results"]) == 2
    assert parsed["results"][0]["title"] == "Result One"
    assert parsed["results"][0]["url"] == "https://example.com/1"
    assert parsed["results"][0]["snippet"] == "First snippet text"


def test_web_search_async() -> None:
    import sys

    ws_mod = sys.modules["vertai.core.tools.web_search"]
    original = ws_mod.httpx.AsyncClient

    def _fake_async(**kwargs: Any) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_DDGO_SAMPLE.encode("utf-8"))
        return original(transport=httpx.MockTransport(handler))

    ws_mod.httpx.AsyncClient = _fake_async  # type: ignore[assignment]
    try:
        t = make_web_search_tool(max_results=1)
        result = asyncio.run(t.aexecute(query="async test"))
    finally:
        ws_mod.httpx.AsyncClient = original  # type: ignore[assignment]

    parsed = json.loads(result)
    assert len(parsed["results"]) == 1


def test_default_registry_contains_all_builtins() -> None:
    from vertai.core.tools import default_registry

    reg = default_registry()
    assert reg.names == [
        "calculator",
        "file_read",
        "file_write",
        "http_request",
        "web_search",
    ]
    # Each spec has a real schema, not an empty stub.
    for spec in reg.to_specs():
        assert spec.input_schema["type"] == "object"
        assert "properties" in spec.input_schema
