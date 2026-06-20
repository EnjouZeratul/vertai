"""Session Memory Module - Conversation history and context persistence.

Provides:
- In-memory and file-based session storage
- Conversation history with role-based messages
- Context window management with a language-aware token estimate
- Atomic, corruption-resilient session persistence

Design notes (S9 contract — see ``docs/ARCHITECTURE.md`` §3.9):

* ``session_id`` is whitelisted to ``^[a-zA-Z0-9_-]+$`` to prevent path
  traversal when writing ``{session_id}.json`` under ``persist_directory``.
* ``SessionMemory.save`` writes a sibling temp file then ``os.replace``-es it
  into place — a crash mid-write never leaves a half-written session file.
* ``SessionMemory.load`` raises ``SessionCorruptedError`` (with the failing
  path) on truncated / malformed JSON instead of bubbling up ``KeyError`` /
  ``TypeError``.
* ``_generate_session_id`` uses ``uuid4`` so two sessions created in the same
  millisecond never collide on disk.
* ``_estimate_tokens`` prefers ``tiktoken`` when the optional dependency is
  installed, otherwise falls back to a language-aware heuristic (one token per
  CJK / common non-ASCII character, ~4 ASCII characters per token) — far more
  accurate than the previous ``len(text) // 4`` for the Chinese-first SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

# ``session_id`` whitelist. Anything outside [A-Za-z0-9_-] is rejected to keep
# the id safe as a filename component under ``persist_directory`` (defends
# against ``../`` traversal, absolute paths, separators, NULs, ...).
_SESSION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_-]+$")


class SessionCorruptedError(ValueError):
    """Raised when a persisted session file cannot be parsed.

    Carries the offending path so callers can surface a clear, actionable
    message rather than a bare ``json.JSONDecodeError`` / ``KeyError``.
    """

    def __init__(self, path: str | Path, reason: str) -> None:
        self.path = str(path)
        self.reason = reason
        super().__init__(f"Session file corrupted ({reason}): {path}")


class _TiktokenBackend:
    """Lazy wrapper around ``tiktoken`` so the import stays optional.

    ``tiktoken`` is an optional dependency. When it is unavailable we fall back
    to the language-aware heuristic. The wrapper is a singleton instantiated at
    module import time so the success/failure of the import is resolved once.
    """

    def __init__(self) -> None:
        self._available = False
        self._encoder: Any = None
        try:
            import tiktoken  # type: ignore[import-not-found]

            # cl100k_base is the encoding used by GPT-4/4o-family models and is
            # a reasonable cross-model approximation for chat token counts.
            self._encoder = tiktoken.get_encoding("cl100k_base")
            self._available = True
        except Exception:  # pragma: no cover - environment dependent
            # Any failure (missing module, missing encoding files, etc.) simply
            # disables the fast path; the heuristic remains correct.
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def count(self, text: str) -> int:
        if not self._available or self._encoder is None:
            return _heuristic_token_count(text)
        return len(self._encoder.encode(text))


_TIKTOKEN = _TiktokenBackend()


def _is_cjk_or_other_non_ascii(char: str) -> bool:
    """Return True for CJK ideographs / kana / hangul / other non-ASCII chars.

    These tend to encode to roughly one token each across modern BPE tokenizers
    (cl100k_base, qwen, glm, ...), which makes "1 char == 1 token" a far better
    estimate than treating them as part of a 4-chars-per-token average.
    """
    cp = ord(char)
    # CJK Unified Ideographs + Extensions A, B
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF:
        return True
    # CJK Compatibility Ideographs + radicals
    if 0xF900 <= cp <= 0xFAFF or 0x2E80 <= cp <= 0x2EFF:
        return True
    # Hiragana / Katakana (Japanese)
    if 0x3040 <= cp <= 0x30FF:
        return True
    # Hangul Syllables + Jamo (Korean)
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return True
    # Any other non-ASCII non-whitespace char (emoji, accented Latin, full-
    # width punctuation, ...) — also billed as ~1 token since BPE splits them
    # far more aggressively than plain ASCII letters/digits.
    if not char.isascii() and not char.isspace():
        return True
    return False


def _heuristic_token_count(text: str) -> int:
    """Language-aware token estimate used when ``tiktoken`` is unavailable.

    Heuristic (deliberately documented, not magic):
    - Each CJK / kana / hangul / non-ASCII character counts as ~1 token.
    - ASCII letters/digits/punctuation count as ~1 token per 4 characters
      (the classical ``len // 4`` rule for English text).
    - Whitespace is free.

    This matches modern BPE tokenizers much more closely than the legacy
    ``len(text) // 4 + 1`` rule, which under-counted Chinese by ~3-4x. For
    pure-ASCII English text it degrades to the same behavior as before, so
    there is no regression on the English path.

    Always returns at least 1 for non-empty text so a single short message
    never reports a zero-token context.
    """
    if not text:
        return 0
    non_ascii_tokens = 0
    ascii_chars = 0
    for char in text:
        if char.isspace():
            continue
        if _is_cjk_or_other_non_ascii(char):
            non_ascii_tokens += 1
        else:
            ascii_chars += 1
    estimate = non_ascii_tokens + (ascii_chars // 4)
    return max(estimate, 1)


@dataclass
class Message:
    """Conversation message."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionConfig:
    """Session memory configuration.

    Attributes:
        max_messages: Maximum number of messages to keep in history.
        max_tokens: Maximum total tokens for context window (estimated).
        auto_save: Automatically save session on changes.
        persist_directory: Directory for persistent session storage.
    """

    max_messages: int = 100
    max_tokens: int = 4096
    auto_save: bool = False
    persist_directory: str | None = None


class SessionMemory:
    """Session memory for conversation history and context persistence.

    Features:
    - Store and retrieve conversation history
    - Automatic context window management
    - Atomic persistence (tmp file + ``os.replace``) so a crash mid-write
      never corrupts an existing session file
    - Friendly handling of corrupted session files
      (``SessionCorruptedError``)
    - ``session_id`` whitelist defends against path traversal

    Examples:
        >>> from vertai.core import SessionMemory
        >>> memory = SessionMemory()
        >>> memory.add_message("user", "Hello!")
        >>> memory.add_message("assistant", "Hi there!")
        >>> history = memory.get_history()
        >>> memory.save("session.json")
        >>> loaded = SessionMemory.from_file("session.json")
    """

    def __init__(
        self,
        session_id: str | None = None,
        config: SessionConfig | None = None,
    ) -> None:
        """Initialize session memory.

        Args:
            session_id: Unique session identifier. Auto-generated (uuid4) if
                None. If supplied, must match ``^[a-zA-Z0-9_-]+$`` (an empty
                string is *not* auto-generated — it is rejected, so callers
                learn about bad input instead of silently getting a new id).
            config: Session configuration.

        Raises:
            ValueError: If a supplied ``session_id`` is empty or contains
                characters outside ``[A-Za-z0-9_-]``.
        """
        if session_id is None:
            self.session_id = self._generate_session_id()
        else:
            # Explicit (incl. empty) id: validate strictly. We deliberately do
            # NOT treat "" as "please auto-generate" — silent fallback would
            # mask caller bugs.
            self._validate_session_id(session_id)
            self.session_id = session_id
        self.config = config or SessionConfig()
        # A plain list, not a deque(maxlen=...): maxlen would silently evict
        # from the left on append, which conflicts with pinning the system
        # prompt. _trim_if_needed enforces max_messages with awareness of the
        # pinned system prompt.
        self._messages: list[Message] = []
        self._context: dict[str, Any] = {}
        self._created_at = time.time()
        self._updated_at = time.time()
        self._token_estimate = 0

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject session IDs that are unsafe as filename components."""
        if not session_id or not _SESSION_ID_PATTERN.match(session_id):
            raise ValueError(
                "Invalid session_id: must match '^[a-zA-Z0-9_-]+$' "
                f"(got {session_id!r}). Path traversal separators and "
                "non-filename-safe characters are rejected."
            )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID using uuid4.

        ``uuid4`` removes the same-millisecond collision the previous
        ``int(time.time() * 1000)`` scheme had when two sessions were created
        back-to-back (which would silently overwrite each other's persisted
        file). 12 hex chars give 48 bits of entropy, ample for a single host.
        """
        return f"session_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text.

        Uses ``tiktoken`` (cl100k_base) when the optional dependency is
        installed; otherwise falls back to ``_heuristic_token_count`` which
        counts each CJK / non-ASCII character as one token and ~4 ASCII
        characters as one token. The heuristic is documented and consistently
        better than the legacy ``len // 4`` rule for Chinese text, which is
        important because the SDK targets a Chinese-first audience.
        """
        return _TIKTOKEN.count(text)

    @property
    def message_count(self) -> int:
        """Get number of messages in history."""
        return len(self._messages)

    @property
    def token_estimate(self) -> int:
        """Get estimated total tokens in history."""
        return self._token_estimate

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to the conversation history.

        Args:
            role: Message role ("system", "user", "assistant").
            content: Message content.
            metadata: Optional metadata.

        Returns:
            The created message.

        Raises:
            ValueError: If ``role`` is not one of system/user/assistant.
        """
        if role not in ("system", "user", "assistant"):
            raise ValueError(
                f"Invalid role: {role}. Must be 'system', 'user', or 'assistant'."
            )

        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self._messages.append(message)
        self._token_estimate += self._estimate_tokens(content)
        self._updated_at = time.time()

        # Trim history to fit max_messages and max_tokens, keeping the leading
        # system prompt pinned (see _trim_if_needed).
        self._trim_if_needed()

        # Auto-save if enabled
        if self.config.auto_save and self.config.persist_directory:
            self.save(self._get_default_save_path())

        return message

    def _trim_if_needed(self) -> None:
        """Trim history to fit within ``max_messages`` and ``max_tokens``.

        The leading message is pinned when it is the system prompt - losing
        it silently would change agent behavior mid-conversation. Everything
        else is FIFO-evicted from the front of the (possibly system-skipped)
        region. At least one message is always kept so the agent loop never
        loses all context.
        """
        # The first message is pinned iff it's a system prompt AND there is
        # more than one message (a lone system prompt is just "the history").
        pinned_system = (
            len(self._messages) > 1 and self._messages[0].role == "system"
        )
        # Index of the first evictable message.
        first_evictable = 1 if pinned_system else 0

        # Enforce max_messages. Several may need to be evicted at once if the
        # caller lowered max_messages after construction.
        max_messages = max(self.config.max_messages, 1)
        while (
            len(self._messages) > max_messages
            and first_evictable < len(self._messages)
        ):
            removed = self._messages.pop(first_evictable)
            self._token_estimate -= self._estimate_tokens(removed.content)

        # Enforce max_tokens on top of the message-count cap. Keep at least
        # one message so token trimming alone never empties the history.
        while (
            self._token_estimate > self.config.max_tokens
            and len(self._messages) > 1
            and first_evictable < len(self._messages)
        ):
            removed = self._messages.pop(first_evictable)
            self._token_estimate -= self._estimate_tokens(removed.content)

    def get_history(
        self,
        limit: int | None = None,
        roles: list[str] | None = None,
    ) -> list[Message]:
        """Get conversation history.

        Args:
            limit: Maximum number of messages to return.
            roles: Filter by roles (e.g., ["user", "assistant"]).

        Returns:
            List of messages.
        """
        messages: list[Message] = list(self._messages)

        if roles:
            messages = [m for m in messages if m.role in roles]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_context(self) -> dict[str, Any]:
        """Get session context data."""
        return self._context.copy()

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key.
            value: Context value.
        """
        self._context[key] = value
        self._updated_at = time.time()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
        self._token_estimate = 0
        self._updated_at = time.time()

    def clear_context(self) -> None:
        """Clear session context."""
        self._context.clear()
        self._updated_at = time.time()

    def clear(self) -> None:
        """Clear all session data."""
        self.clear_history()
        self.clear_context()

    def _get_default_save_path(self) -> str:
        """Get default save path for this session."""
        if not self.config.persist_directory:
            raise ValueError("persist_directory not configured")
        # session_id is whitelisted at construction + save() time, so this
        # path join is safe from traversal.
        return str(
            Path(self.config.persist_directory) / f"{self.session_id}.json"
        )

    def save(self, filepath: str | Path | None = None) -> Path:
        """Save session to a file atomically.

        Writes a sibling temp file in the same directory and then
        ``os.replace``-es it into the final path. ``os.replace`` is atomic on
        POSIX and on Windows for same-filesystem renames, so a crash mid-write
        leaves the previously-saved file (or no file at all if this is the
        first save) untouched rather than half-written.

        Args:
            filepath: Path to save file. Uses default if None.

        Returns:
            The final path the session was written to.

        Raises:
            ValueError: If the session_id fails whitelist validation (defence
                in depth — it is also validated at construction time).
        """
        # Defence in depth: re-validate in case session_id was mutated after
        # construction.
        self._validate_session_id(self.session_id)

        path = Path(filepath or self._get_default_save_path())
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "messages": [m.to_dict() for m in self._messages],
            "context": self._context,
            "config": {
                "max_messages": self.config.max_messages,
                "max_tokens": self.config.max_tokens,
            },
        }

        serialized = json.dumps(data, ensure_ascii=False, indent=2)

        # Write to a temp file in the *same directory* so os.replace stays on
        # one filesystem (and therefore atomic). tempfile.mkstemp gives us a
        # predictable, race-free fd.
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # fsync is best-effort on some filesystems / platforms;
                    # a missing fsync must not abort the save.
                    logger.debug("fsync failed for %s (non-fatal)", tmp_path)
            os.replace(tmp_path, path)
        except BaseException:
            # Clean up the orphaned temp file on any failure (including
            # KeyboardInterrupt) so we don't leak .*.tmp files.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

        logger.info("Session saved: %s", path)
        return path

    def load(self, filepath: str | Path | None = None) -> None:
        """Load session from a file into this instance (in place).

        Replaces this session's messages/context/metadata with the contents of
        the file. On a corrupted (truncated / malformed JSON / missing
        required keys) file, raises ``SessionCorruptedError`` with the path
        rather than a low-level ``json.JSONDecodeError`` / ``KeyError``.

        Args:
            filepath: Path to load file. Uses default if None.

        Raises:
            FileNotFoundError: If the file does not exist.
            SessionCorruptedError: If the file exists but cannot be parsed.
        """
        path = Path(filepath or self._get_default_save_path())

        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SessionCorruptedError(path, f"invalid JSON: {exc.msg}") from exc
        except UnicodeDecodeError as exc:
            raise SessionCorruptedError(path, f"not UTF-8: {exc.reason}") from exc

        if not isinstance(data, dict):
            raise SessionCorruptedError(
                path, f"expected JSON object, got {type(data).__name__}"
            )
        for required in ("session_id", "created_at", "updated_at", "messages"):
            if required not in data:
                raise SessionCorruptedError(
                    path, f"missing required key: {required}"
                )

        try:
            loaded_session_id = str(data["session_id"])
            self._validate_session_id(loaded_session_id)
            self.session_id = loaded_session_id
            self._created_at = float(data["created_at"])
            self._updated_at = float(data["updated_at"])
            messages_data = data["messages"]
            if not isinstance(messages_data, list):
                raise SessionCorruptedError(path, "'messages' must be a list")
            self._messages = [Message.from_dict(m) for m in messages_data]
            loaded_context = data.get("context", {})
            if not isinstance(loaded_context, dict):
                raise SessionCorruptedError(
                    path, "'context' must be a JSON object"
                )
            self._context = loaded_context
        except SessionCorruptedError:
            raise
        except (TypeError, ValueError) as exc:
            # ValueError covers both _validate_session_id and float() failures.
            raise SessionCorruptedError(path, f"invalid field: {exc}") from exc

        # Recalculate token estimate from the loaded messages.
        self._token_estimate = sum(
            self._estimate_tokens(m.content) for m in self._messages
        )

        logger.info("Session loaded: %s", path)

    @classmethod
    def from_file(cls, filepath: str | Path) -> "SessionMemory":
        """Create a session from a file (class-level convenience).

        Mirrors the contract documented in ARCHITECTURE §3.9: returns a fresh
        ``SessionMemory`` populated from the file, with friendly corruption
        handling via ``SessionCorruptedError``.

        Args:
            filepath: Path to session file.

        Returns:
            Loaded session.

        Raises:
            FileNotFoundError: If the file does not exist.
            SessionCorruptedError: If the file cannot be parsed.
        """
        session = cls()
        session.load(filepath)
        return session

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary.

        Returns:
            Dictionary representation of the session.
        """
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "message_count": len(self._messages),
            "token_estimate": self._token_estimate,
            "messages": [m.to_dict() for m in self._messages],
            "context": self._context,
        }

    def get_last_n_messages(self, n: int) -> list[Message]:
        """Get last N messages from history.

        Args:
            n: Number of messages to retrieve.

        Returns:
            List of most recent messages.
        """
        if n <= 0:
            return []
        return list(self._messages)[-n:]

    def get_formatted_history(
        self,
        format_type: str = "default",
    ) -> str:
        """Get formatted conversation history.

        Args:
            format_type: Output format ("default", "markdown", "json").

        Returns:
            Formatted history string.
        """
        if not self._messages:
            return ""

        if format_type == "json":
            return json.dumps(
                [m.to_dict() for m in self._messages],
                ensure_ascii=False,
                indent=2,
            )

        lines = []
        for msg in self._messages:
            if format_type == "markdown":
                lines.append(f"**{msg.role.title()}**: {msg.content}")
            else:
                lines.append(f"[{msg.role}] {msg.content}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SessionMemory(session_id={self.session_id!r}, "
            f"messages={len(self._messages)}, tokens~={self._token_estimate})"
        )
