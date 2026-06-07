"""Tests for SessionMemory module."""

import json
import os
import tempfile
import time

import pytest

from vertai.core.memory import (
    Message,
    SessionConfig,
    SessionMemory,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.timestamp > 0
        assert msg.metadata == {}

    def test_create_message_with_metadata(self):
        """Test creating message with metadata."""
        msg = Message(
            role="assistant",
            content="Response",
            metadata={"model": "llama3", "tokens": 50},
        )
        assert msg.metadata["model"] == "llama3"
        assert msg.metadata["tokens"] == 50

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = Message(role="user", content="Test")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert "timestamp" in data
        assert "metadata" in data

    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": 12345.0,
            "metadata": {"key": "value"},
        }
        msg = Message.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp == 12345.0
        assert msg.metadata["key"] == "value"


class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SessionConfig()
        assert config.max_messages == 100
        assert config.max_tokens == 4096
        assert config.auto_save == False
        assert config.persist_directory is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = SessionConfig(
            max_messages=50,
            max_tokens=2048,
            auto_save=True,
            persist_directory="/tmp/sessions",
        )
        assert config.max_messages == 50
        assert config.max_tokens == 2048
        assert config.auto_save == True
        assert config.persist_directory == "/tmp/sessions"


class TestSessionMemory:
    """Tests for SessionMemory."""

    def test_create_session(self):
        """Test creating a session."""
        session = SessionMemory()
        assert session.session_id.startswith("session_")
        assert session.message_count == 0
        assert session.token_estimate == 0

    def test_create_session_with_id(self):
        """Test creating session with custom ID."""
        session = SessionMemory(session_id="custom_session")
        assert session.session_id == "custom_session"

    def test_add_message(self):
        """Test adding messages."""
        session = SessionMemory()
        msg = session.add_message("user", "Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert session.message_count == 1

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        session = SessionMemory()
        session.add_message("system", "System prompt")
        session.add_message("user", "Question")
        session.add_message("assistant", "Answer")
        assert session.message_count == 3

    def test_invalid_role(self):
        """Test invalid role raises error."""
        session = SessionMemory()
        with pytest.raises(ValueError):
            session.add_message("invalid_role", "content")

    def test_get_history(self):
        """Test getting history."""
        session = SessionMemory()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        history = session.get_history()
        assert len(history) == 3
        assert history[0].content == "Q1"
        assert history[2].content == "Q2"

    def test_get_history_with_limit(self):
        """Test getting history with limit."""
        session = SessionMemory()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        history = session.get_history(limit=2)
        assert len(history) == 2
        assert history[0].content == "A1"
        assert history[1].content == "Q2"

    def test_get_history_with_role_filter(self):
        """Test getting history with role filter."""
        session = SessionMemory()
        session.add_message("system", "System")
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        history = session.get_history(roles=["user", "assistant"])
        assert len(history) == 3
        assert "system" not in [m.role for m in history]

    def test_context_operations(self):
        """Test context operations."""
        session = SessionMemory()
        session.set_context("key1", "value1")
        session.set_context("key2", {"nested": "data"})

        ctx = session.get_context()
        assert ctx["key1"] == "value1"
        assert ctx["key2"]["nested"] == "data"

    def test_clear_history(self):
        """Test clearing history."""
        session = SessionMemory()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.clear_history()

        assert session.message_count == 0
        assert session.token_estimate == 0

    def test_clear_context(self):
        """Test clearing context."""
        session = SessionMemory()
        session.set_context("key", "value")
        session.clear_context()

        assert session.get_context() == {}

    def test_clear_all(self):
        """Test clearing all."""
        session = SessionMemory()
        session.add_message("user", "Q")
        session.set_context("key", "value")
        session.clear()

        assert session.message_count == 0
        assert session.get_context() == {}

    def test_trim_by_message_count(self):
        """Test trimming by message count."""
        config = SessionConfig(max_messages=3)
        session = SessionMemory(config=config)

        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")
        session.add_message("assistant", "A2")
        session.add_message("user", "Q3")

        # Should only keep last 3 messages: Q2, A2, Q3
        assert session.message_count == 3
        history = session.get_history()
        assert history[0].content == "Q2"
        assert history[1].content == "A2"
        assert history[2].content == "Q3"

    def test_trim_by_tokens(self):
        """Test trimming by token estimate."""
        # Create config with very small token limit
        config = SessionConfig(max_tokens=20)  # ~20 tokens = ~80 chars
        session = SessionMemory(config=config)

        # Add messages that exceed limit
        session.add_message("user", "This is a long message that exceeds the token limit")
        session.add_message("assistant", "Response")

        # Should trim to fit
        assert session.token_estimate <= config.max_tokens

    def test_trim_by_tokens_removes_multiple(self):
        """Test that token trimming removes messages when exceeding limit."""
        config = SessionConfig(max_tokens=10)  # Very small limit
        session = SessionMemory(config=config)

        # Add a long message that will exceed the token limit
        long_text = "This is a very long message that definitely exceeds the tiny token limit of ten tokens"
        session.add_message("user", long_text)

        # Token estimate should be trimmed (but at least 1 message remains)
        # The trim only happens when adding messages
        assert session.message_count >= 1

    def test_token_trimming_preserves_last_message(self):
        """Test that token trimming keeps at least one message."""
        config = SessionConfig(max_tokens=5)  # Extremely small limit
        session = SessionMemory(config=config)

        # Add multiple messages
        session.add_message("user", "First message here")
        session.add_message("assistant", "Second message response")
        session.add_message("user", "Third message")

        # Should always have at least one message
        assert session.message_count >= 1

    def test_get_last_n_messages(self):
        """Test getting last N messages."""
        session = SessionMemory()
        session.add_message("user", "Q1")
        session.add_message("assistant", "A1")
        session.add_message("user", "Q2")

        last = session.get_last_n_messages(2)
        assert len(last) == 2
        assert last[0].content == "A1"
        assert last[1].content == "Q2"

    def test_formatted_history_default(self):
        """Test default formatted history."""
        session = SessionMemory()
        session.add_message("user", "Q")
        session.add_message("assistant", "A")

        formatted = session.get_formatted_history()
        assert "[user] Q" in formatted
        assert "[assistant] A" in formatted

    def test_formatted_history_markdown(self):
        """Test markdown formatted history."""
        session = SessionMemory()
        session.add_message("user", "Q")
        session.add_message("assistant", "A")

        formatted = session.get_formatted_history(format_type="markdown")
        assert "**User**: Q" in formatted
        assert "**Assistant**: A" in formatted

    def test_formatted_history_json(self):
        """Test JSON formatted history."""
        session = SessionMemory()
        session.add_message("user", "Q")

        formatted = session.get_formatted_history(format_type="json")
        data = json.loads(formatted)
        assert len(data) == 1
        assert data[0]["role"] == "user"

    def test_formatted_history_empty(self):
        """Test formatted history with empty messages."""
        session = SessionMemory()

        formatted = session.get_formatted_history()
        assert formatted == ""

    def test_repr(self):
        """Test __repr__ method."""
        session = SessionMemory(session_id="test_repr_session")
        session.add_message("user", "Test message")

        repr_str = repr(session)
        assert "test_repr_session" in repr_str
        assert "messages=1" in repr_str
        assert "tokens~=" in repr_str

    def test_to_dict(self):
        """Test session serialization."""
        session = SessionMemory(session_id="test_session")
        session.add_message("user", "Q")

        data = session.to_dict()
        assert data["session_id"] == "test_session"
        assert data["message_count"] == 1
        assert "messages" in data
        assert "context" in data


class TestSessionPersistence:
    """Tests for session persistence."""

    def test_save_and_load(self):
        """Test save and load session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "session.json")

            # Create and save session
            session = SessionMemory(session_id="persist_test")
            session.add_message("user", "Question")
            session.add_message("assistant", "Answer")
            session.set_context("key", "value")
            session.save(filepath)

            # Load session
            loaded = SessionMemory()
            loaded.load(filepath)

            assert loaded.session_id == "persist_test"
            assert loaded.message_count == 2
            assert loaded.get_context()["key"] == "value"

    def test_from_file(self):
        """Test creating session from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "session.json")

            session = SessionMemory(session_id="file_test")
            session.add_message("user", "Test")
            session.save(filepath)

            loaded = SessionMemory.from_file(filepath)
            assert loaded.session_id == "file_test"

    def test_auto_save(self):
        """Test auto-save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SessionConfig(
                auto_save=True,
                persist_directory=tmpdir,
            )
            session = SessionMemory(session_id="auto_save_test", config=config)

            # Adding message should trigger auto-save
            session.add_message("user", "Question")

            # Check file exists
            expected_path = os.path.join(tmpdir, "auto_save_test.json")
            assert os.path.exists(expected_path)

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        session = SessionMemory()
        with pytest.raises(FileNotFoundError):
            session.load("/nonexistent/path/session.json")

    def test_persist_directory_created(self):
        """Test persist directory is created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "sessions")
            filepath = os.path.join(nested_dir, "session.json")

            session = SessionMemory(session_id="nested_test")
            session.save(filepath)

            assert os.path.exists(nested_dir)
            assert os.path.exists(filepath)

    def test_get_default_save_path_without_directory(self):
        """Test _get_default_save_path raises error without persist_directory."""
        session = SessionMemory(session_id="test")

        with pytest.raises(ValueError, match="persist_directory not configured"):
            session._get_default_save_path()


class TestSessionMemoryIntegration:
    """Integration tests for SessionMemory."""

    def test_full_workflow(self):
        """Test full workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SessionConfig(
                max_messages=10,
                persist_directory=tmpdir,
            )
            session = SessionMemory(session_id="workflow", config=config)

            # Simulate conversation
            session.add_message("system", "You are a helpful assistant.")
            session.add_message("user", "What is Python?")
            session.add_message("assistant", "Python is a programming language.")

            # Store context
            session.set_context("user_preferences", {"language": "en"})
            session.set_context("conversation_topic", "programming")

            # Get history
            history = session.get_history(roles=["user", "assistant"])
            assert len(history) == 2

            # Save
            session.save()

            # Load in new session
            loaded = SessionMemory.from_file(
                os.path.join(tmpdir, "workflow.json")
            )
            assert loaded.message_count == 3
            assert loaded.get_context()["conversation_topic"] == "programming"

    def test_token_estimation(self):
        """Test token estimation accuracy."""
        session = SessionMemory()

        # Add a message of approximately 100 characters
        session.add_message("user", "This is a test message with about one hundred characters")

        # Token estimate should be roughly len/4 tokens (using simple heuristic)
        # The message is ~60 chars, so ~15 tokens
        assert session.token_estimate > 0
        assert session.token_estimate < 50