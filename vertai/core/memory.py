"""Session Memory Module - Conversation history and context persistence.

Supports:
- In-memory and file-based session storage
- Conversation history with role-based messages
- Context window management
- Session serialization/deserialization
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    - Persist sessions to disk
    - Load previous sessions

    Examples:
        >>> from vertai.core import SessionMemory
        >>> memory = SessionMemory()
        >>> memory.add_message("user", "Hello!")
        >>> memory.add_message("assistant", "Hi there!")
        >>> history = memory.get_history()
        >>> memory.save("session.json")
        >>> memory.load("session.json")
    """

    def __init__(
        self,
        session_id: str | None = None,
        config: SessionConfig | None = None,
    ):
        """Initialize session memory.

        Args:
            session_id: Unique session identifier. Auto-generated if None.
            config: Session configuration.
        """
        self.session_id = session_id or self._generate_session_id()
        self.config = config or SessionConfig()
        self._messages: list[Message] = []
        self._context: dict[str, Any] = {}
        self._created_at = time.time()
        self._updated_at = time.time()
        self._token_estimate = 0

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"session_{int(time.time() * 1000)}"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text.

        Uses simple heuristic: ~4 characters per token.
        """
        return len(text) // 4 + 1

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

        # Trim history if needed
        self._trim_if_needed()

        # Auto-save if enabled
        if self.config.auto_save and self.config.persist_directory:
            self.save(self._get_default_save_path())

        return message

    def _trim_if_needed(self) -> None:
        """Trim history to fit within limits."""
        # Trim by message count
        while len(self._messages) > self.config.max_messages:
            removed = self._messages.pop(0)
            self._token_estimate -= self._estimate_tokens(removed.content)

        # Trim by token estimate
        while self._token_estimate > self.config.max_tokens and len(self._messages) > 1:
            removed = self._messages.pop(0)
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
        messages = self._messages

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
        return str(
            Path(self.config.persist_directory) / f"{self.session_id}.json"
        )

    def save(self, filepath: str | None = None) -> None:
        """Save session to a file.

        Args:
            filepath: Path to save file. Uses default if None.
        """
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

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Session saved: {path}")

    def load(self, filepath: str | None = None) -> None:
        """Load session from a file.

        Args:
            filepath: Path to load file. Uses default if None.
        """
        path = Path(filepath or self._get_default_save_path())

        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.session_id = data["session_id"]
        self._created_at = data["created_at"]
        self._updated_at = data["updated_at"]
        self._messages = [Message.from_dict(m) for m in data["messages"]]
        self._context = data.get("context", {})

        # Recalculate token estimate
        self._token_estimate = sum(
            self._estimate_tokens(m.content) for m in self._messages
        )

        logger.info(f"Session loaded: {path}")

    @classmethod
    def from_file(cls, filepath: str) -> "SessionMemory":
        """Create session from a file.

        Args:
            filepath: Path to session file.

        Returns:
            Loaded session.
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
        return self._messages[-n:] if n > 0 else []

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
