"""File read/write built-in tools.

Both tools use :class:`pathlib.Path` and validate the resolved path against an
optional ``base_dir`` to prevent path traversal (``../../etc/passwd`` etc.).
When ``base_dir`` is not provided, no traversal guard is applied (the caller
is responsible for sandboxing); this keeps the tools usable in unconstrained
local-first scenarios while making the safe path opt-in.

The tools are constructed via :func:`vertai.core.tool.FunctionTool` so that
``base_dir`` (and ``max_bytes``) are captured as factory-time configuration
rather than exposed to the model.
"""

from __future__ import annotations

from pathlib import Path

from vertai.core.tool import FunctionTool

# Default cap on a single file_read to avoid slurping arbitrarily large files
# into the model context.
_DEFAULT_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB


class PathTraversalError(PermissionError):
    """Raised when a resolved path escapes the configured ``base_dir``."""


def _resolve_and_check(path: str, base_dir: str | None) -> Path:
    """Resolve ``path`` and ensure it stays within ``base_dir`` if set.

    When ``base_dir`` is provided, *relative* paths are interpreted as
    relative to ``base_dir`` (so an agent passing ``"notes.txt"`` lands
    inside the sandbox rather than the process CWD). Absolute paths must
    still resolve inside ``base_dir``.
    """
    raw = Path(path).expanduser()
    if base_dir is not None:
        root = Path(base_dir).expanduser().resolve()
        if raw.is_absolute():
            target = raw.resolve()
        else:
            target = (root / raw).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PathTraversalError(
                f"Path '{path}' resolves outside the allowed base_dir "
                f"'{base_dir}' (resolved to '{target}')"
            ) from exc
        return target
    return raw.resolve()


def make_file_read_tool(
    base_dir: str | None = None,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    name: str = "file_read",
    description: str | None = None,
) -> FunctionTool:
    """Build a ``file_read`` tool bounded by ``base_dir`` and ``max_bytes``.

    The tool reads text files (UTF-8) and returns their contents as a string.
    Paths are resolved and checked against ``base_dir`` when set.
    """

    def _file_read(path: str) -> str:
        """Read a UTF-8 text file from disk.

        Args:
            path: Path to the file to read (absolute, or relative to the
                tool's configured base_dir).
        """
        target = _resolve_and_check(path, base_dir)
        if not target.exists():
            raise FileNotFoundError(f"No such file: '{target}'")
        if not target.is_file():
            raise IsADirectoryError(f"Not a regular file: '{target}'")
        data = target.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(
                f"File '{target}' is {len(data)} bytes which exceeds the "
                f"{max_bytes}-byte limit"
            )
        return data.decode("utf-8")

    desc = description or (
        "Read a UTF-8 text file from disk"
        + (f" (confined to {base_dir})" if base_dir else "")
        + "."
    )
    return FunctionTool(
        _file_read,
        name_override=name,
        description_override=desc,
    )


def make_file_write_tool(
    base_dir: str | None = None,
    *,
    name: str = "file_write",
    description: str | None = None,
) -> FunctionTool:
    """Build a ``file_write`` tool bounded by ``base_dir``.

    Writes text content (UTF-8) to disk, creating parent directories as
    needed. When ``base_dir`` is set, the resolved path must stay inside it.
    """

    def _file_write(path: str, content: str) -> str:
        """Write UTF-8 text content to a file (overwriting if it exists).

        Args:
            path: Destination path (absolute, or relative to the tool's
                configured base_dir).
            content: The text content to write.
        """
        target = _resolve_and_check(path, base_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to '{target}'"

    desc = description or (
        "Write UTF-8 text content to a file"
        + (f" (confined to {base_dir})" if base_dir else "")
        + "."
    )
    return FunctionTool(
        _file_write,
        name_override=name,
        description_override=desc,
    )


# Default instances (unbounded) for the convenience registry. Callers who
# want sandboxing should construct their own via the ``make_*`` factories.
file_read: FunctionTool = make_file_read_tool()
file_write: FunctionTool = make_file_write_tool()


__all__: list[str] = [
    "PathTraversalError",
    "file_read",
    "file_write",
    "make_file_read_tool",
    "make_file_write_tool",
]
