"""Generic HTTP request built-in tool.

Uses :mod:`httpx` (a hard dependency of VertAI) to perform a single HTTP
request. Only a small whitelist of methods is allowed; response bodies are
returned as text with a cap to avoid flooding the model context. Timeouts
and a configurable response size limit keep the tool safe to expose to an
agent.

The async path (:meth:`_HTTPRequestTool.aexecute`) uses a real
:class:`httpx.AsyncClient` so the tool honours VertAI's async-first contract
rather than wrapping the sync client in an executor.
"""

from __future__ import annotations

import json as _json
from typing import Any

import httpx

from vertai.core.tool import FunctionTool

_ALLOWED_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
)
_DEFAULT_MAX_BODY = 64 * 1024
_DEFAULT_TIMEOUT = 30.0


def _build_request_kwargs(
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    body: Any | None,
) -> dict[str, Any]:
    """Translate optional headers/params/body into httpx request kwargs."""
    request_kwargs: dict[str, Any] = {}
    if headers:
        request_kwargs["headers"] = dict(headers)
    if params:
        request_kwargs["params"] = dict(params)
    if body is not None:
        if isinstance(body, (dict, list)):
            request_kwargs["json"] = body
        elif isinstance(body, str):
            request_kwargs["content"] = body
        else:
            request_kwargs["json"] = body
    return request_kwargs


def _summarise_response(
    response: httpx.Response, max_body_bytes: int
) -> str:
    """Render an :class:`httpx.Response` as a JSON string summary."""
    raw = response.content
    truncated = len(raw) > max_body_bytes
    body_text = raw[:max_body_bytes].decode("utf-8", errors="replace")
    summary = {
        "status_code": response.status_code,
        "reason": response.reason_phrase,
        "url": str(response.url),
        "headers": dict(response.headers),
        "body": body_text,
        "truncated": truncated,
    }
    try:
        return _json.dumps(summary, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(summary)


def _validate_method(method: str) -> str:
    method_up = method.upper()
    if method_up not in _ALLOWED_METHODS:
        raise ValueError(
            f"Method '{method}' not allowed. "
            f"Allowed: {sorted(_ALLOWED_METHODS)}"
        )
    return method_up


def _sync_http_request(
    url: str,
    method: str,
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    body: Any | None,
    timeout: float,
    max_body_bytes: int,
) -> str:
    """Synchronous HTTP request implementation."""
    method_up = _validate_method(method)
    request_kwargs = _build_request_kwargs(headers, params, body)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.request(method_up, url, **request_kwargs)
    return _summarise_response(response, max_body_bytes)


async def _async_http_request(
    url: str,
    method: str,
    headers: dict[str, str] | None,
    params: dict[str, str] | None,
    body: Any | None,
    timeout: float,
    max_body_bytes: int,
) -> str:
    """Asynchronous HTTP request implementation (real AsyncClient)."""
    method_up = _validate_method(method)
    request_kwargs = _build_request_kwargs(headers, params, body)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        response = await client.request(method_up, url, **request_kwargs)
    return _summarise_response(response, max_body_bytes)


class _HTTPRequestTool(FunctionTool):
    """HTTP request tool with a real async path."""

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_body_bytes: int = _DEFAULT_MAX_BODY,
        name: str = "http_request",
        description: str | None = None,
    ) -> None:
        self._http_timeout = timeout
        self._max_body_bytes = max_body_bytes

        def _http_request(
            url: str,
            method: str = "GET",
            headers: dict[str, str] | None = None,
            params: dict[str, str] | None = None,
            body: Any | None = None,
        ) -> str:
            """Perform a single HTTP request and return the response as text.

            Args:
                url: The absolute URL to request (must include scheme).
                method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD,
                    OPTIONS). Defaults to GET.
                headers: Optional request headers.
                params: Optional query-string parameters.
                body: Optional request body. Dict/list values are
                    JSON-encoded; strings are sent as-is.
            """
            return _sync_http_request(
                url,
                method,
                headers,
                params,
                body,
                self._http_timeout,
                self._max_body_bytes,
            )

        super().__init__(
            _http_request,
            name_override=name,
            description_override=description
            or (
                "Perform a single HTTP request and return the response "
                "(status, headers, body) as a JSON string."
            ),
        )

    async def aexecute(self, **kwargs: Any) -> str:
        # Skip argument validation via the pydantic model: FunctionTool.aexecute
        # runs that step, but we override here to use the real async client.
        # Validate arguments first so behaviour matches the sync path.
        try:
            args = self._validate(kwargs)
        except Exception as exc:
            return self._handle_failure(exc)
        try:
            return await _async_http_request(
                str(args.get("url", "")),
                str(args.get("method", "GET")),
                args.get("headers"),
                args.get("params"),
                args.get("body"),
                self._http_timeout,
                self._max_body_bytes,
            )
        except Exception as exc:
            return self._handle_failure(exc)


def make_http_request_tool(
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_body_bytes: int = _DEFAULT_MAX_BODY,
    name: str = "http_request",
    description: str | None = None,
) -> _HTTPRequestTool:
    """Build a configured ``http_request`` tool."""
    return _HTTPRequestTool(
        timeout=timeout,
        max_body_bytes=max_body_bytes,
        name=name,
        description=description,
    )


# Default instance for the convenience registry.
http_request: _HTTPRequestTool = _HTTPRequestTool()


__all__ = [
    "http_request",
    "make_http_request_tool",
]
