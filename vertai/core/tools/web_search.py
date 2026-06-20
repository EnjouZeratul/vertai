"""Web search built-in tool.

A lightweight HTTP-only web search backed by DuckDuckGo's HTML endpoint (no
API key required). The tool returns the top result titles + URLs/snippets so
an agent can reason about web content without a heavyweight search dependency.

Because VertAI keeps a hard-dependency surface of only ``httpx`` + ``pydantic``,
we do not pull in a dedicated search SDK. If you need richer results, swap
this tool out via the factory pattern (build your own
:class:`~vertai.core.tool.FunctionTool` backed by a paid search API).

The async path (:meth:`_WebSearchTool.aexecute`) uses a real
:class:`httpx.AsyncClient`.
"""

from __future__ import annotations

import html as _html
import json as _json
import re
from typing import Any

import httpx

from vertai.core.tool import FunctionTool

_DEFAULT_ENDPOINT = "https://html.duckduckgo.com/html/"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RESULTS = 5

# DuckDuckGo HTML result links are redirect links of the form
# ``//duckduckgo.com/l/?uddg=<urlencoded>``. We extract the ``uddg`` param.
_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_results(html: str, max_results: int) -> list[dict[str, str]]:
    """Parse DuckDuckGo HTML into a list of ``{title, url, snippet}`` dicts."""
    results: list[dict[str, str]] = []
    for match in _RESULT_LINK_RE.finditer(html):
        href = match.group("href")
        title = _html.unescape(_TAG_RE.sub("", match.group("title"))).strip()
        url = _resolve_ddg_href(href)
        if not title or not url:
            continue
        # Find the nearest snippet following this link.
        snippet_match = _SNIPPET_RE.search(html, match.end())
        snippet = ""
        if snippet_match:
            snippet = _html.unescape(
                _TAG_RE.sub("", snippet_match.group("snippet"))
            ).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _resolve_ddg_href(href: str) -> str:
    """Resolve a DuckDuckGo redirect link to its target URL."""
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        # Pull the uddg param value.
        from urllib.parse import parse_qs, urlsplit

        qs = parse_qs(urlsplit(href).query)
        targets = qs.get("uddg") or qs.get("uddg=")
        if targets:
            return _html.unescape(targets[0])
    return href


def _sync_web_search(
    query: str,
    max_results: int,
    endpoint: str,
    timeout: float,
) -> str:
    """Synchronous web search implementation."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.post(
            endpoint,
            data={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; VertAI-Agent/1.0; "
                    "+https://github.com/EnjouZeratul/vertai)"
                )
            },
        )
        response.raise_for_status()
        results = _extract_results(response.text, max_results)
    return _json.dumps(
        {"query": query, "results": results}, ensure_ascii=False
    )


async def _async_web_search(
    query: str,
    max_results: int,
    endpoint: str,
    timeout: float,
) -> str:
    """Asynchronous web search implementation (real AsyncClient)."""
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        response = await client.post(
            endpoint,
            data={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; VertAI-Agent/1.0; "
                    "+https://github.com/EnjouZeratul/vertai)"
                )
            },
        )
        response.raise_for_status()
        results = _extract_results(response.text, max_results)
    return _json.dumps(
        {"query": query, "results": results}, ensure_ascii=False
    )


class _WebSearchTool(FunctionTool):
    """Web search tool with a real async path."""

    def __init__(
        self,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout: float = _DEFAULT_TIMEOUT,
        max_results: int = _DEFAULT_MAX_RESULTS,
        name: str = "web_search",
        description: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._search_timeout = timeout
        self._max_results = max_results

        def _web_search(query: str) -> str:
            """Search the web for a query and return the top results.

            Args:
                query: The search query string.
            """
            return _sync_web_search(
                query,
                self._max_results,
                self._endpoint,
                self._search_timeout,
            )

        super().__init__(
            _web_search,
            name_override=name,
            description_override=description
            or (
                "Search the web (DuckDuckGo HTML endpoint) and return the "
                "top results as a JSON string with title/url/snippet fields."
            ),
        )

    async def aexecute(self, **kwargs: Any) -> str:
        try:
            args = self._validate(kwargs)
        except Exception as exc:
            return self._handle_failure(exc)
        try:
            return await _async_web_search(
                str(args.get("query", "")),
                self._max_results,
                self._endpoint,
                self._search_timeout,
            )
        except Exception as exc:
            return self._handle_failure(exc)


def make_web_search_tool(
    *,
    endpoint: str = _DEFAULT_ENDPOINT,
    timeout: float = _DEFAULT_TIMEOUT,
    max_results: int = _DEFAULT_MAX_RESULTS,
    name: str = "web_search",
    description: str | None = None,
) -> _WebSearchTool:
    """Build a configured ``web_search`` tool."""
    return _WebSearchTool(
        endpoint=endpoint,
        timeout=timeout,
        max_results=max_results,
        name=name,
        description=description,
    )


# Default instance for the convenience registry.
web_search: _WebSearchTool = _WebSearchTool()


__all__ = [
    "make_web_search_tool",
    "web_search",
]
