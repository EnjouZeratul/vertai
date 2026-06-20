"""Built-in tools.

Five ready-to-use tools that cover the common vertical-domain integration
needs: ``web_search`` (HTTP search), ``file_read`` / ``file_write`` (sandboxed
filesystem access), ``http_request`` (generic HTTP via httpx), and
``calculator`` (safe arithmetic via ``ast``).

Each tool is a :class:`~vertai.core.tool.FunctionTool` produced by the
:func:`~vertai.core.tool.tool` decorator, so its JSON Schema (with
constraints) and docstring-derived description are auto-generated.

All built-in tools can also be obtained as a ready-made registry via
:func:`default_registry`.
"""

from __future__ import annotations

from vertai.core.tool import ToolRegistry
from vertai.core.tools.calculator import calculator
from vertai.core.tools.file import file_read, file_write
from vertai.core.tools.http import http_request
from vertai.core.tools.web_search import web_search

__all__ = [
    "calculator",
    "default_registry",
    "file_read",
    "file_write",
    "http_request",
    "web_search",
]


def default_registry() -> ToolRegistry:
    """Return a fresh :class:`ToolRegistry` pre-populated with all built-in
    tools. Each call returns a new registry so callers can extend or trim it
    without mutating shared state."""
    return ToolRegistry(
        [
            calculator,
            file_read,
            file_write,
            http_request,
            web_search,
        ]
    )
