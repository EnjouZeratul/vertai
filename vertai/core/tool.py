"""Tool abstraction layer.

Defines the ``Tool`` ABC, :class:`FunctionTool` (generated from a Python
function), the :func:`tool` decorator, and :class:`ToolRegistry`. The design
targets the OpenAI Agents SDK ``@function_tool`` behaviour (2025-2026):

* ``inspect`` is used to read the function signature.
* The function docstring is parsed best-effort (Google / NumPy / Sphinx
  ``:param:`` styles) to extract per-parameter descriptions.
* A Pydantic model is built dynamically so the generated JSON Schema carries
  type information **and** Field constraints (``ge``/``le``/``gt``/``lt``/
  ``min_length``/``max_length``/``pattern``/``description``), whether the
  constraint is supplied via ``Annotated[int, Field(...)]`` or via a default
  ``int = Field(...)``.
* ``timeout`` enforces a per-tool wall-clock budget; the default behaviour is
  ``error_as_result`` (return a timeout message so the agent can recover),
  with ``raise_exception`` raising :class:`ToolTimeoutError`.
* ``failure_error_function`` lets callers customise how tool failures are
  surfaced to the model (default: a friendly message; ``None``: re-raise).
* ``to_specs()`` emits :class:`~vertai.core.provider.ToolSpec` objects that
  ``LLMProvider.generate(tools=...)`` consumes.

The module is the contract layer defined by ``docs/ARCHITECTURE.md`` section
3.6.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Protocol, TypeVar, runtime_checkable

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from vertai.core.provider import ToolSpec

T = TypeVar("T")


class _DefaultFailureHandler:
    """Sentinel type used so ``failure_error_function=None`` ("re-raise") is
    distinguishable from "use the default friendly handler". A dedicated
    sentinel instance avoids the awkward ``...``/Ellipsis trick."""


_DEFAULT_FAILURE_HANDLER = _DefaultFailureHandler()


class ToolTimeoutError(TimeoutError):
    """Raised when a tool exceeds its configured ``timeout`` in
    ``raise_exception`` mode."""


class ToolError(RuntimeError):
    """Raised when a tool is invoked before it has been registered, or when
    arguments fail validation and no failure handler is configured."""


# ---------------------------------------------------------------------------
# Docstring parsing (best-effort, Google / NumPy / Sphinx styles)
# ---------------------------------------------------------------------------


_PARAM_GOOGLE_RE = re.compile(
    r"^\s*(?P<name>\w+)\s*(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<desc>.*)$"
)
_PARAM_NUMPY_RE = re.compile(r"^\s*(?P<name>\w+)\s*:\s*(?P<type>[^\n]*)$")
_PARAM_SPHINX_RE = re.compile(
    r"^\s*:param\s+(?:(?P<type>\S+)\s+)?(?P<name>\w+)\s*:\s*(?P<desc>.*)$"
)


def _parse_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Parse a docstring into ``(summary, {param_name: description})``.

    Best-effort parser supporting the common styles used in the Python
    ecosystem: Google, NumPy, and Sphinx ``:param:``. Unknown formats simply
    yield the leading paragraph as ``summary`` and an empty param map (the
    decorator still works; per-param descriptions fall back to the Field
    ``description`` if any, otherwise to an empty string).

    The parser never raises: malformed docstrings degrade to no param
    descriptions rather than breaking tool creation.
    """
    if not docstring:
        return "", {}
    # ``inspect.cleandoc`` normalises leading indentation consistently.
    lines = inspect.cleandoc(docstring).splitlines()
    if not lines:
        return "", {}

    # Summary = leading non-blank lines until a blank line or a recognised
    # section header.
    summary_lines: list[str] = []
    i = 0
    section_headers = {
        "args",
        "arguments",
        "parameters",
        "params",
        "returns",
        "return",
        "yields",
        "raises",
        "examples",
        "note",
        "notes",
    }
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            break
        # Stop if we hit a section header like "Args:" / "Parameters -----".
        lower = stripped.rstrip(":").lower().split()[0] if stripped else ""
        if lower in section_headers and (
            stripped.endswith(":") or _is_numpy_header(lines, i)
        ):
            break
        summary_lines.append(stripped)
        i += 1
    summary = " ".join(summary_lines).strip()

    params: dict[str, str] = {}
    # Walk the rest looking for an Args/Parameters section (Google style) or
    # NumPy-style "Parameters\n----------\n" block, and Sphinx ``:param:``.
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        lower = stripped.lower()

        # Sphinx :param: lines can appear anywhere.
        m = _PARAM_SPHINX_RE.match(line)
        if m:
            name = m.group("name")
            desc = (m.group("desc") or "").strip()
            params.setdefault(name, desc)
            i += 1
            continue

        # Google-style section header: "Args:" / "Parameters:".
        if lower in ("args:", "arguments:", "parameters:", "params:"):
            i += 1
            i = _consume_google_block(lines, i, params)
            continue

        # NumPy-style section header: "Parameters\n----------".
        if _is_numpy_header(lines, i):
            header_word = stripped.lower().split()[0]
            if header_word in ("parameters", "args", "params"):
                i += 2  # skip header + underline
                i = _consume_numpy_block(lines, i, params)
                continue

        i += 1

    return summary, params


def _is_numpy_header(lines: list[str], idx: int) -> bool:
    """Detect a NumPy-style section header: a word followed by a line of
    ``-``/``=`` underlining of (roughly) the same width."""
    if idx + 1 >= len(lines):
        return False
    header = lines[idx].strip()
    underline = lines[idx + 1].strip()
    if not header or not underline:
        return False
    if not re.fullmatch(r"[-=]{2,}", underline):
        return False
    # Underline should be at least as long as the header word.
    return len(underline) >= len(header.split()[0]) if header.split() else False


def _consume_google_block(
    lines: list[str], idx: int, params: dict[str, str]
) -> int:
    """Consume a Google-style indented Args block, populating ``params``.

    Returns the index past the block. Continuation lines (deeper-indented)
    extend the previous parameter's description.
    """
    if idx >= len(lines):
        return idx
    base_indent = _indent_width(lines[idx])
    last_name: str | None = None
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        indent = _indent_width(line)
        if indent < base_indent:
            # Dedent ends the block.
            break
        if indent == base_indent:
            m = _PARAM_GOOGLE_RE.match(line)
            if m:
                last_name = m.group("name")
                params[last_name] = (m.group("desc") or "").strip()
            idx += 1
        else:
            # Continuation of the previous param description.
            if last_name is not None:
                params[last_name] = (params[last_name] + " " + line.strip()).strip()
            idx += 1
    return idx


def _consume_numpy_block(
    lines: list[str], idx: int, params: dict[str, str]
) -> int:
    """Consume a NumPy-style Parameters block. Entries look like
    ``name : type`` followed by an indented description."""
    if idx >= len(lines):
        return idx
    base_indent = _indent_width(lines[idx]) if lines[idx].strip() else 4
    current_name: str | None = None
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        indent = _indent_width(line)
        if indent < base_indent and line.strip():
            break
        m = _PARAM_NUMPY_RE.match(line)
        if m and indent == base_indent:
            current_name = m.group("name")
            # NumPy type info on the same line is not used as description.
            params[current_name] = ""
            idx += 1
            # Consume following indented lines as the description.
            while idx < len(lines):
                dline = lines[idx]
                if not dline.strip():
                    idx += 1
                    continue
                if _indent_width(dline) <= base_indent:
                    break
                params[current_name] = (
                    params[current_name] + " " + dline.strip()
                ).strip()
                idx += 1
        else:
            idx += 1
    return idx


def _indent_width(line: str) -> int:
    """Return the number of leading whitespace characters (tabs count as 1)."""
    return len(line) - len(line.lstrip())


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


def _build_param_model(
    func: Callable[..., Any],
    param_docs: dict[str, str],
) -> type[Any]:
    """Build a Pydantic model from ``func``'s signature.

    Honours ``Annotated[T, Field(...)]`` metadata **and** default-as-Field
    (``x: int = Field(...)``). Parameter descriptions come from the docstring,
    falling back to a Field ``description`` if present.

    Type hints are resolved up-front via :func:`typing.get_type_hints` with
    ``include_extras=True`` so that ``Annotated[...]`` metadata survives and
    the resolved types do not depend on the dynamic model's module globals
    being able to re-resolve the original string annotations.
    """
    import typing

    sig = inspect.signature(func)
    fields: dict[str, tuple[Any, Any]] = {}

    # Resolve the function's type hints to real objects (not strings), keeping
    # Annotated metadata. Falls back gracefully if hints cannot be resolved
    # (e.g. some forward refs); in that case we use the raw annotation.
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs are not representable as JSON Schema fields.
            continue

        annotation = hints.get(name, param.annotation)
        default = param.default
        doc = param_docs.get(name, "").strip()

        if annotation is inspect.Parameter.empty:
            # Treat as Any when unannotated so we still generate a schema.
            annotation = Any

        field_info = _field_info_from_param(annotation, default, doc)

        # Whether required or optional, the field tuple is (annotation, FieldInfo).
        fields[name] = (annotation, field_info)

    model = create_model(  # type: ignore[call-overload]
        f"_{func.__name__}_Input",
        __base__=None,
        **fields,
    )
    return model  # type: ignore[no-any-return]


def _field_info_from_param(
    annotation: Any, default: Any, doc: str
) -> FieldInfo:
    """Construct a :class:`FieldInfo` honouring ``Annotated[..., Field(...)]``
    metadata, default-as-Field, and a plain default value.

    Always constructs the FieldInfo via :func:`FieldInfo.merge_field_infos`
    (or a fresh :func:`Field`) so that ``default`` is baked in at creation
    time. Mutating ``FieldInfo.default`` after construction is a Pydantic v2
    footgun: the JSON schema still marks the field as required. Passing
    ``default=...`` as a merge kwarg avoids that.
    """
    metadata = getattr(annotation, "__metadata__", None)
    annotated_field: FieldInfo | None = None
    if metadata:
        for meta in metadata:
            if isinstance(meta, FieldInfo):
                annotated_field = meta
                break

    has_param_default = default is not inspect.Parameter.empty

    # Build the FieldInfo sources for ``merge_field_infos``: the Annotated
    # FieldInfo (constraints) plus, when present, a default-as-Field default.
    sources: list[FieldInfo] = []
    if annotated_field is not None:
        sources.append(annotated_field)
    if isinstance(default, FieldInfo):
        sources.append(default)

    # Does any Field source already carry an explicit description? An explicit
    # Field(description=...) takes precedence over the docstring; the docstring
    # is used only as a fallback (matching the OpenAI Agents SDK behaviour).
    field_has_description = any(
        isinstance(s, FieldInfo) and s.description for s in sources
    )
    description_for_field = None if field_has_description else (doc or None)

    # Kwargs shared by both code paths below.
    shared_kwargs: dict[str, Any] = {}
    if description_for_field:
        shared_kwargs["description"] = description_for_field

    if sources:
        # When a plain parameter default is present (e.g. ``= 5``), it must
        # be routed through the merge kwargs (not post-hoc mutation) so the
        # generated JSON schema treats the field as optional.
        merge_kwargs = dict(shared_kwargs)
        if has_param_default and not isinstance(default, FieldInfo):
            merge_kwargs["default"] = default
        return FieldInfo.merge_field_infos(*sources, **merge_kwargs)

    # No FieldInfo sources: build a fresh Field from scratch.
    if has_param_default:
        return Field(default=default, **shared_kwargs)  # type: ignore[no-any-return]
    return Field(**shared_kwargs)  # type: ignore[no-any-return]




def _model_to_json_schema(model: type[Any]) -> dict[str, Any]:
    """Render a Pydantic model's schema as a JSON Schema dict suitable for
    tool ``input_schema``. Drops Pydantic-internal keys."""
    schema: dict[str, Any] = dict(model.model_json_schema())
    # Strip ``title`` (the model name) so the schema looks like a plain
    # parameter object schema; keep ``type``/``properties``/``required``.
    schema.pop("title", None)
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict):
                prop.pop("title", None)
    return schema


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Tool abstraction.

    Subclasses expose a ``name``, ``description``, a JSON-Schema
    ``parameters`` dict, and ``execute`` / ``aexecute`` entry points. The
    result is currently a string (LLM-friendly); structured outputs are a
    1.x extension point.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema describing this tool's input parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool synchronously and return a string result."""

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> str:
        """Run the tool asynchronously and return a string result."""

    def to_spec(self) -> ToolSpec:
        """Render this tool as a :class:`ToolSpec` for LLMProvider consumption."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.parameters,
        )


# ---------------------------------------------------------------------------
# FunctionTool (generated from a Python function)
# ---------------------------------------------------------------------------


_FailureHandler = Callable[[Exception], str]
"""Type alias for a function that converts a caught exception into a
string surfaced to the model."""


class FunctionTool(Tool):
    """A :class:`Tool` generated from a Python function.

    The function signature is inspected and a Pydantic input model is built
    so that ``execute`` validates arguments and the generated JSON Schema
    carries Field constraints. ``timeout`` (seconds) wraps execution;
    ``failure_error_function`` controls how exceptions are surfaced.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        timeout: float | None = None,
        failure_error_function: _FailureHandler | _DefaultFailureHandler = _DEFAULT_FAILURE_HANDLER,
        timeout_mode: str = "error_as_result",
    ) -> None:
        if not callable(func):
            raise TypeError(f"FunctionTool requires a callable, got {type(func)!r}")
        self._func = func
        self._name = name_override or func.__name__
        summary, param_docs = _parse_docstring(func.__doc__)
        self._description = description_override or summary or f"Tool {self._name}"
        self._param_docs = param_docs
        self._input_model = _build_param_model(func, param_docs)
        self._parameters = _model_to_json_schema(self._input_model)
        self._timeout = timeout
        self._failure_error_function = failure_error_function
        if timeout_mode not in ("error_as_result", "raise_exception"):
            raise ValueError(
                f"timeout_mode must be 'error_as_result' or 'raise_exception', "
                f"got {timeout_mode!r}"
            )
        self._timeout_mode = timeout_mode

    # -- Tool API ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        # Return a shallow copy so callers cannot mutate cached schema.
        return dict(self._parameters)

    @property
    def func(self) -> Callable[..., Any]:
        """The wrapped function (read-only)."""
        return self._func

    @property
    def timeout(self) -> float | None:
        return self._timeout

    # -- execution --------------------------------------------------------

    def _validate(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        validated = self._input_model.model_validate(kwargs)
        return dict(validated)

    def execute(self, **kwargs: Any) -> str:
        """Run the wrapped function synchronously with validated arguments.

        ``timeout`` is enforced when set; on timeout the behaviour is
        controlled by ``timeout_mode`` (``error_as_result`` returns a message,
        ``raise_exception`` raises :class:`ToolTimeoutError`). Other
        exceptions are routed through ``failure_error_function`` (default
        friendly message; ``None`` re-raises).
        """
        try:
            args = self._validate(kwargs)
        except Exception as exc:  # validation errors
            return self._handle_failure(exc)

        try:
            if self._timeout is None:
                result = self._func(**args)
            else:
                result = self._run_with_timeout_sync(args)
        except ToolTimeoutError:
            if self._timeout_mode == "raise_exception":
                raise
            return f"Tool '{self._name}' timed out after {self._timeout}s."
        except Exception as exc:
            return self._handle_failure(exc)
        return _stringify_result(result)

    async def aexecute(self, **kwargs: Any) -> str:
        """Run the wrapped function asynchronously.

        If the wrapped function is a coroutine function it is awaited; any
        other callable is run in a thread so the event loop is not blocked.
        Timeout and failure handling mirror :meth:`execute`.
        """
        try:
            args = self._validate(kwargs)
        except Exception as exc:
            return self._handle_failure(exc)

        try:
            if self._timeout is None:
                result = await self._invoke_async(args)
            else:
                result = await self._run_with_timeout_async(args)
        except ToolTimeoutError:
            if self._timeout_mode == "raise_exception":
                raise
            return f"Tool '{self._name}' timed out after {self._timeout}s."
        except Exception as exc:
            return self._handle_failure(exc)
        return _stringify_result(result)

    # -- helpers ----------------------------------------------------------

    async def _invoke_async(self, args: dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(self._func):
            return await self._func(**args)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._func(**args))

    def _run_with_timeout_sync(self, args: dict[str, Any]) -> Any:
        """Enforce a wall-clock timeout for a (potentially blocking) sync
        function by running it in a worker thread and joining with a deadline.
        """
        result_box: dict[str, Any] = {}

        def _runner() -> None:
            try:
                result_box["value"] = self._func(**args)
            except BaseException as exc:  # propagate to caller
                result_box["error"] = exc

        thread = _start_daemon(_runner)
        thread.join(timeout=self._timeout)
        if thread.is_alive():
            # Thread is still running past the deadline. We cannot forcefully
            # kill it in CPython; leave it as a daemon and raise a timeout.
            raise ToolTimeoutError(
                f"Tool '{self._name}' exceeded timeout of {self._timeout}s"
            )
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value")

    async def _run_with_timeout_async(self, args: dict[str, Any]) -> Any:
        # ``asyncio.timeout`` is only available on Python 3.11+; ``wait_for``
        # works on 3.10 (the project minimum) and is sufficient here.
        try:
            return await asyncio.wait_for(
                self._invoke_async(args), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise ToolTimeoutError(
                f"Tool '{self._name}' exceeded timeout of {self._timeout}s"
            ) from exc

    def _handle_failure(self, exc: Exception) -> str:
        handler = self._failure_error_function
        if isinstance(handler, _DefaultFailureHandler):
            return (
                f"Tool '{self._name}' failed: {type(exc).__name__}: {exc}"
            )
        if handler is None:
            # ``failure_error_function=None`` means re-raise.
            raise exc
        return handler(exc)


def _start_daemon(target: Callable[[], None]) -> threading.Thread:
    """Start a daemon thread running ``target``."""
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _stringify_result(result: Any) -> str:
    """Coerce a tool result into the LM-friendly string form. ``str`` and
    ``bytes`` are handled directly; pydantic models / dataclasses / dicts are
    JSON-serialised so the model receives structured content."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        try:
            return result.decode("utf-8")
        except UnicodeDecodeError:
            return repr(result)
    if isinstance(result, (dict, list, tuple, int, float, bool)):
        return json.dumps(result, default=str, ensure_ascii=False)
    # Fall back to str() for arbitrary objects.
    return str(result)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    failure_error_function: _FailureHandler | _DefaultFailureHandler = _DEFAULT_FAILURE_HANDLER,
    timeout_mode: str = "error_as_result",
    registry: "ToolRegistry | None" = None,
) -> Any:
    """Decorator that turns a Python function into a :class:`FunctionTool`.

    Mirrors the OpenAI Agents SDK ``@function_tool`` shape:

    * ``name`` / ``description`` override the function name / docstring
      summary when supplied.
    * ``timeout`` enforces a per-tool wall-clock budget. ``timeout_mode``
      selects between ``error_as_result`` (default; return a timeout message
      so the agent can recover) and ``raise_exception`` (raise
      :class:`ToolTimeoutError`).
    * ``failure_error_function`` controls how exceptions are surfaced to the
      model. Default: a friendly ``"Tool '<name>' failed: ..."`` string.
      Pass ``None`` to re-raise. Pass a callable to customise.
    * ``registry`` optionally registers the produced tool immediately.

    Usable bare (``@tool``) or parameterised (``@tool(name="...")``).
    """
    def _build(target: Callable[..., Any]) -> FunctionTool:
        ft = FunctionTool(
            target,
            name_override=name,
            description_override=description,
            timeout=timeout,
            failure_error_function=failure_error_function,
            timeout_mode=timeout_mode,
        )
        if registry is not None:
            registry.register(ft)
        # Attach the tool to the function so callers can introspect it.
        setattr(target, "__vertai_tool__", ft)
        return ft

    if func is not None and callable(func):
        # Bare ``@tool`` usage.
        return _build(func)
    return _build


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolLike(Protocol):
    """Minimal protocol satisfied by :class:`Tool` (and any duck-typed tool
    that exposes ``name``/``description``/``parameters``/``execute``)."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    def execute(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    """Registry of tools, keyed by name.

    ``to_specs()`` emits the :class:`ToolSpec` list consumed by
    :class:`~vertai.core.provider.LLMProvider`. ``call()`` resolves a
    :class:`~vertai.core.provider.ToolCall` returned by a provider into a
    real tool invocation and string result, so an agent loop can wire
    provider -> registry -> tool with one call.
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises :class:`ToolError` on duplicate names."""
        if not isinstance(tool, Tool):
            raise TypeError(
                f"ToolRegistry.register requires a Tool, got {type(tool)!r}"
            )
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered with name '{tool.name}'")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> Tool | None:
        """Remove and return a tool by name, or ``None`` if not present."""
        return self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name``.

        Raises :class:`ToolError` if no such tool is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"No tool registered with name '{name}'. "
                f"Known tools: {sorted(self._tools)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        """Sorted list of registered tool names."""
        return sorted(self._tools)

    def to_specs(self) -> list[ToolSpec]:
        """Emit :class:`ToolSpec` objects for every registered tool, in
        deterministic name order."""
        return [self._tools[name].to_spec() for name in sorted(self._tools)]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Resolve ``name`` and invoke ``execute(**arguments)`` synchronously."""
        return self.get(name).execute(**arguments)

    async def acall(self, name: str, arguments: dict[str, Any]) -> str:
        """Resolve ``name`` and invoke ``aexecute(**arguments)``."""
        return await self.get(name).aexecute(**arguments)


__all__ = [
    "FunctionTool",
    "Tool",
    "ToolError",
    "ToolLike",
    "ToolRegistry",
    "ToolTimeoutError",
    "tool",
]
