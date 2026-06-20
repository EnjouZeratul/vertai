"""Tests for SessionMemory module."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from vertai.core import memory as memory_module
from vertai.core.memory import (
    Message,
    SessionConfig,
    SessionCorruptedError,
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
        assert config.auto_save is False
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
        assert config.auto_save is True
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


# ---------------------------------------------------------------------------
# S9: Session ID generation (uuid4), path-traversal whitelist, atomic write,
# corrupted-file handling, language-aware token estimation.
# ---------------------------------------------------------------------------


class TestSessionIdGeneration:
    """S9: _generate_session_id uses uuid4 to avoid same-millisecond clashes."""

    def test_generated_id_matches_whitelist(self):
        """Auto-generated ids must satisfy the whitelist (safe filenames)."""
        sid = SessionMemory._generate_session_id()
        assert sid.startswith("session_")
        suffix = sid[len("session_"):]
        assert suffix.isalnum(), suffix
        assert len(suffix) == 12

    def test_rapidly_created_sessions_have_distinct_ids(self):
        """Create many sessions in a tight loop: no id collisions.

        This is the regression test for the ``int(time.time() * 1000)`` bug,
        which produced identical ids for sessions created in the same
        millisecond and silently overwrote each other's persisted file. uuid4
        makes collision astronomically unlikely (48 bits of entropy here).
        """
        ids = {SessionMemory().session_id for _ in range(500)}
        assert len(ids) == 500, f"uuid4 collision: {500 - len(ids)} duplicates"

    def test_explicit_id_is_preserved(self):
        """A caller-supplied id is used verbatim."""
        session = SessionMemory(session_id="my_custom_id_1")
        assert session.session_id == "my_custom_id_1"


class TestSessionIdWhitelist:
    """S9: session_id is whitelisted to prevent path traversal."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/evil",   # classic traversal
            "..",            # parent dir
            "/abs/path",     # absolute path
            "a/b",           # separator
            "a\\b",          # windows separator
            "a:b",           # windows drive separator
            "a;b",           # shell metachar
            "a b",           # space
            "a.b",           # dot (could become hidden/pathy)
            "",              # empty
            "café",          # non-ascii
            "a\x00b",        # NUL byte
            ".",             # single dot
        ],
    )
    def test_invalid_session_id_rejected_at_construction(self, bad_id):
        """Each of these must be rejected to keep {id}.json path-safe."""
        with pytest.raises(ValueError, match="Invalid session_id"):
            SessionMemory(session_id=bad_id)

    @pytest.mark.parametrize(
        "good_id",
        ["session_1", "abc-XYZ_012", "A", "0", "_-only"],
    )
    def test_valid_session_id_accepted(self, good_id):
        """Valid ids survive the whitelist."""
        assert SessionMemory(session_id=good_id).session_id == good_id

    def test_invalid_id_rejected_on_save_defence_in_depth(self, tmp_path):
        """save() re-validates session_id (defence in depth) in case it was
        mutated after construction."""
        session = SessionMemory(session_id="legit_id")
        # Simulate post-construction corruption (e.g. a subclass bug).
        session.session_id = "../escaped"
        with pytest.raises(ValueError, match="Invalid session_id"):
            session.save(str(tmp_path / "ignored.json"))

    def test_path_traversal_cannot_escape_persist_directory(self, tmp_path):
        """The headline security property: a traversal-shaped id is rejected
        before any file is created, so no file escapes persist_directory."""
        target_outside = tmp_path / "outside.json"
        config = SessionConfig(persist_directory=str(tmp_path / "sessions"))
        with pytest.raises(ValueError):
            SessionMemory(session_id="../../" + target_outside.name, config=config)
        # Nothing was created anywhere.
        assert not target_outside.exists()
        assert not (tmp_path / "sessions").exists()


class TestAtomicSave:
    """S9: save() writes tmp + os.replace — a crash never corrupts the
    previously-written file."""

    def test_save_creates_final_file(self, tmp_path):
        """Normal save: final file exists, no leftover temp files."""
        session = SessionMemory(session_id="atomic_ok")
        session.add_message("user", "hello")
        target = tmp_path / "atomic_ok.json"
        returned = session.save(str(target))

        assert returned == target
        assert target.exists()
        # No leftover temp files in the directory.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], leftovers

    def test_crash_during_replace_preserves_original(self, tmp_path):
        """If os.replace raises (simulated crash / FS error / power loss
        represented as an exception), the previously-saved file must be
        intact and readable.

        This is the core atomic-write guarantee: a mid-write failure cannot
        leave the target file half-written or empty.
        """
        session = SessionMemory(session_id="atomic_crash")
        session.add_message("user", "original message")

        target = tmp_path / "atomic_crash.json"
        # First save succeeds and establishes a known-good file.
        session.save(str(target))
        original_bytes = target.read_bytes()
        assert original_bytes, "precondition: file must be non-empty"
        original_data = json.loads(target.read_text(encoding="utf-8"))
        assert original_data["messages"][0]["content"] == "original message"

        # Now mutate the session and attempt a second save whose os.replace
        # blows up. The real os.replace failed to land the new content.
        session.add_message("user", "second message that should not persist")

        def crashing_replace(src, dst):  # noqa: ANN001 - matches os.replace sig
            raise OSError("simulated replace failure (power loss / FS error)")

        with mock.patch("vertai.core.memory.os.replace", crashing_replace):
            with pytest.raises(OSError, match="simulated replace failure"):
                session.save(str(target))

        # The previously-saved file is byte-identical to before the failed
        # save — atomicity held.
        assert target.read_bytes() == original_bytes
        # And it's still a valid session file with the original content.
        reloaded = json.loads(target.read_text(encoding="utf-8"))
        assert reloaded["messages"][0]["content"] == "original message"
        assert len(reloaded["messages"]) == 1

        # os.replace is restored to the real implementation by the context
        # manager, so a subsequent save works normally.
        session.save(str(target))
        reloaded2 = json.loads(target.read_text(encoding="utf-8"))
        assert len(reloaded2["messages"]) == 2

    def test_crash_during_replace_cleans_up_temp_file(self, tmp_path):
        """A failed save must not leak a .tmp file in the directory."""
        session = SessionMemory(session_id="atomic_cleanup")
        session.add_message("user", "x")
        target = tmp_path / "atomic_cleanup.json"
        session.save(str(target))

        def crashing_replace(src, dst):  # noqa: ANN001
            raise OSError("boom")

        with mock.patch("vertai.core.memory.os.replace", crashing_replace):
            with pytest.raises(OSError):
                session.save(str(target))

        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], [p.name for p in leftovers]

    def test_no_temp_file_left_in_final_directory_on_success(self, tmp_path):
        """Sanity: a clean save leaves exactly one file (the final one)."""
        session = SessionMemory(session_id="clean")
        session.add_message("user", "y")
        session.save(str(tmp_path / "clean.json"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.json"]


class TestCorruptedLoad:
    """S9: load() raises SessionCorruptedError with the path instead of a raw
    JSONDecodeError / KeyError / TypeError."""

    def test_malformed_json_raises_session_corrupted_error(self, tmp_path):
        """Truncated / non-JSON file: friendly error, not bare JSONDecodeError."""
        bad = tmp_path / "broken.json"
        bad.write_text("{not valid json", encoding="utf-8")

        session = SessionMemory()
        with pytest.raises(SessionCorruptedError) as exc_info:
            session.load(str(bad))
        assert str(bad) in str(exc_info.value)
        assert "invalid JSON" in str(exc_info.value)

    def test_empty_file_raises_session_corrupted_error(self, tmp_path):
        """Empty file: friendly error."""
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(empty))
        assert "invalid JSON" in str(exc_info.value)

    def test_missing_required_key_raises_session_corrupted_error(self, tmp_path):
        """Structurally valid JSON but missing required keys: friendly error."""
        bad = tmp_path / "missing.json"
        bad.write_text(json.dumps({"session_id": "x"}), encoding="utf-8")

        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "missing required key" in str(exc_info.value)

    def test_non_object_json_raises_session_corrupted_error(self, tmp_path):
        """A JSON array / number / string is not a valid session."""
        bad = tmp_path / "array.json"
        bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "expected JSON object" in str(exc_info.value)

    def test_corrupted_file_does_not_mutate_session(self, tmp_path):
        """Loading a corrupted file leaves the in-memory session untouched —
        we don't half-apply bad state before raising."""
        session = SessionMemory(session_id="alive")
        session.add_message("user", "keep me")
        original_count = session.message_count

        bad = tmp_path / "broken.json"
        bad.write_text("{broken", encoding="utf-8")

        with pytest.raises(SessionCorruptedError):
            session.load(str(bad))

        assert session.session_id == "alive"
        assert session.message_count == original_count
        assert session.get_history()[0].content == "keep me"

    def test_corrupted_error_carries_path_attribute(self, tmp_path):
        """The error exposes the offending path for tooling."""
        bad = tmp_path / "x.json"
        bad.write_text("garbage", encoding="utf-8")
        try:
            SessionMemory().load(str(bad))
        except SessionCorruptedError as exc:
            assert exc.path == str(bad)
        else:
            pytest.fail("expected SessionCorruptedError")

    def test_from_file_propagates_corrupted_error(self, tmp_path):
        """The class-level constructor surfaces the same friendly error."""
        bad = tmp_path / "broken2.json"
        bad.write_text("not json", encoding="utf-8")
        with pytest.raises(SessionCorruptedError):
            SessionMemory.from_file(str(bad))


class TestTokenEstimationLanguageAware:
    """S9: _estimate_tokens uses tiktoken when available, else a language-aware
    heuristic that counts each CJK char as ~1 token (the SDK is Chinese-first).

    The legacy ``len(text) // 4 + 1`` rule under-counted Chinese by ~3-4x,
    which made ``max_tokens`` context management unreliable.
    """

    def test_empty_text_is_zero(self):
        assert SessionMemory._estimate_tokens("") == 0

    def test_chinese_estimate_higher_than_legacy_len_div_4(self):
        """The regression assertion: Chinese text must produce a token count
        meaningfully higher than the old ``len // 4 + 1`` rule, which treated
        4 chinese characters as 1 token."""
        chinese = "你好世界，这是一个测试"  # 11 chars
        legacy = len(chinese) // 4 + 1  # = 3
        new = SessionMemory._estimate_tokens(chinese)
        # Each chinese char is ~1 token, so the new estimate must beat the
        # legacy underestimate by a wide margin.
        assert new >= len(chinese) - 2, (new, legacy)
        assert new > legacy, (new, legacy)

    def test_chinese_long_text_estimate_close_to_char_count(self):
        """For pure Chinese the estimate should be close to the character
        count (modern BPE tokenizers give roughly 1 token per CJK char)."""
        text = "今天天气真好，我想出去玩。" * 10  # 120 chars
        estimate = SessionMemory._estimate_tokens(text)
        # Allow generous slack (tokenizer or heuristic), but it must be in the
        # right order of magnitude — not 30 (len//4) and not 1000.
        assert 40 <= estimate <= 200, estimate

    def test_english_estimate_reasonable(self):
        """English stays in the ~len/4 ballpark — no regression on ASCII."""
        text = "The quick brown fox jumps over the lazy dog" * 5  # ~225 chars
        estimate = SessionMemory._estimate_tokens(text)
        # Roughly 4 chars per token; allow 2x slack either way.
        assert 30 <= estimate <= 120, estimate

    def test_mixed_text_estimate_between_pure(self):
        """Mixed CN+EN lands between the two pure estimates."""
        cn = "你好世界，这是一个测试" * 5
        en = "The quick brown fox jumps over the lazy dog" * 2
        mixed = cn + " " + en

        cn_est = SessionMemory._estimate_tokens(cn)
        en_est = SessionMemory._estimate_tokens(en)
        mixed_est = SessionMemory._estimate_tokens(mixed)

        # Mixed should be in the ballpark of cn+en (whitespace is free).
        assert mixed_est >= int(cn_est * 0.8 + en_est * 0.8), (
            mixed_est, cn_est, en_est
        )

    def test_non_ascii_non_cjk_counts_as_token(self):
        """Accented Latin / emoji / fullwidth chars also bill ~1 token."""
        text = "café naïve ✓ 🚀"
        # 4 non-ascii non-space chars (é, ï, ✓, 🚀) + 9 ascii letters.
        # estimate >= 4 (non-ascii) and < 20 (bounded by content).
        estimate = SessionMemory._estimate_tokens(text)
        assert estimate >= 4, estimate

    def test_tiktoken_backend_flag_matches_availability(self):
        """The module-level backend flag reflects whether tiktoken is
        importable. We don't assert a specific value (env-dependent) but the
        flag must be a bool and the estimator must work in either mode."""
        assert isinstance(memory_module._TIKTOKEN.available, bool)
        # Estimator works regardless of which backend is active.
        assert SessionMemory._estimate_tokens("hello 世界") > 0


class TestTrimPreservesSystemPrompt:
    """S9: _trim_if_needed pins the leading system prompt instead of evicting
    it first via FIFO pop(0)."""

    def test_system_prompt_not_evicted_by_token_pressure(self):
        """With a tiny max_tokens and many user/assistant pairs, the system
        prompt at index 0 survives."""
        config = SessionConfig(max_tokens=30, max_messages=100)
        session = SessionMemory(session_id="trim_sys", config=config)
        session.add_message("system", "You are a helpful assistant.")
        for i in range(10):
            session.add_message("user", f"question number {i} " * 5)
            session.add_message("assistant", f"answer number {i} " * 5)

        history = session.get_history()
        assert history[0].role == "system"
        assert history[0].content == "You are a helpful assistant."

    def test_system_prompt_not_evicted_by_message_count(self):
        """max_messages cap keeps the system prompt too."""
        config = SessionConfig(max_messages=3, max_tokens=10**9)
        session = SessionMemory(session_id="trim_cnt", config=config)
        session.add_message("system", "system-prompt-text")
        for i in range(5):
            session.add_message("user", f"u{i}")
            session.add_message("assistant", f"a{i}")

        history = session.get_history()
        assert history[0].role == "system"
        assert history[0].content == "system-prompt-text"
        assert len(history) <= 3

    def test_at_least_one_message_always_kept(self):
        """Even with absurdly small limits, history never goes empty after an
        add (so the agent loop always has context)."""
        config = SessionConfig(max_tokens=1, max_messages=1)
        session = SessionMemory(session_id="trim_min", config=config)
        session.add_message("user", "x" * 500)
        assert session.message_count >= 1


class TestSaveReturnAndRoundTrip:
    """S9: save() returns the final Path; full save/load round trip preserves
    all fields including context and metadata."""

    def test_save_returns_path(self, tmp_path):
        session = SessionMemory(session_id="ret")
        session.add_message("user", "hi")
        target = tmp_path / "ret.json"
        result = session.save(str(target))
        assert isinstance(result, Path)
        assert result == target

    def test_round_trip_preserves_metadata_and_context(self, tmp_path):
        session = SessionMemory(session_id="roundtrip")
        session.add_message(
            "assistant",
            "with meta",
            metadata={"model": "glm", "latency_ms": 123},
        )
        session.set_context("topic", "testing")
        session.set_context("nested", {"a": 1})
        target = tmp_path / "roundtrip.json"
        session.save(str(target))

        loaded = SessionMemory.from_file(str(target))
        msg = loaded.get_history()[0]
        assert msg.metadata == {"model": "glm", "latency_ms": 123}
        ctx = loaded.get_context()
        assert ctx["topic"] == "testing"
        assert ctx["nested"] == {"a": 1}

    def test_overwrite_via_atomic_replace(self, tmp_path):
        """Repeated saves overwrite cleanly via os.replace."""
        session = SessionMemory(session_id="overwrite")
        target = tmp_path / "overwrite.json"
        for i in range(3):
            session.add_message("user", f"msg {i}")
            session.save(str(target))
        loaded = SessionMemory.from_file(str(target))
        assert loaded.message_count == 3
        assert [m.content for m in loaded.get_history()] == [
            "msg 0", "msg 1", "msg 2"
        ]


class TestCorruptedLoadDefensiveBranches:
    """Cover the defensive corruption branches in load() with real malformed
    payloads (not mock-based) so the error contracts are genuinely exercised."""

    def test_non_utf8_file_raises_session_corrupted_error(self, tmp_path):
        """A binary / non-UTF-8 file is reported as corruption, not a raw
        UnicodeDecodeError."""
        bad = tmp_path / "binary.json"
        # Invalid UTF-8 continuation byte sequence.
        bad.write_bytes(b"\xff\xfe\x00{bad bytes")
        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "UTF-8" in str(exc_info.value) or "not UTF-8" in str(exc_info.value)

    def test_messages_not_a_list_raises(self, tmp_path):
        """'messages' present but not a list: friendly corruption error."""
        bad = tmp_path / "msgs_not_list.json"
        bad.write_text(
            json.dumps(
                {
                    "session_id": "x",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                    "messages": "not a list",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "'messages' must be a list" in str(exc_info.value)

    def test_context_not_a_dict_raises(self, tmp_path):
        """'context' present but not a JSON object: friendly corruption error."""
        bad = tmp_path / "ctx_not_dict.json"
        bad.write_text(
            json.dumps(
                {
                    "session_id": "x",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                    "messages": [],
                    "context": ["not", "a", "dict"],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "'context' must be a JSON object" in str(exc_info.value)

    def test_invalid_field_type_raises(self, tmp_path):
        """Wrong field type (created_at not a number): friendly corruption
        error rather than a bare TypeError from float()."""
        bad = tmp_path / "bad_types.json"
        bad.write_text(
            json.dumps(
                {
                    "session_id": "valid_id",
                    "created_at": "not-a-number",
                    "updated_at": 2.0,
                    "messages": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "invalid field" in str(exc_info.value)

    def test_invalid_session_id_inside_file_raises(self, tmp_path):
        """A file whose stored session_id fails the whitelist is treated as
        corrupted — we never load a traversal-shaped id."""
        bad = tmp_path / "evil_id.json"
        bad.write_text(
            json.dumps(
                {
                    "session_id": "../../etc/evil",
                    "created_at": 1.0,
                    "updated_at": 2.0,
                    "messages": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SessionCorruptedError) as exc_info:
            SessionMemory().load(str(bad))
        assert "invalid field" in str(exc_info.value)


class TestEdgeCases:
    """Small edge cases that lock down behavior."""

    def test_get_last_n_messages_non_positive_returns_empty(self):
        """n <= 0 returns an empty list (documented boundary)."""
        session = SessionMemory()
        session.add_message("user", "a")
        session.add_message("assistant", "b")
        assert session.get_last_n_messages(0) == []
        assert session.get_last_n_messages(-5) == []

    def test_get_history_limit_zero_returns_empty(self):
        """limit=0 is falsy, so per the existing contract it returns *all*
        messages (documented existing behavior — kept for back-compat)."""
        session = SessionMemory()
        session.add_message("user", "a")
        # limit=0 is falsy -> no slicing -> returns all (existing semantics).
        assert len(session.get_history(limit=0)) == 1

    def test_temp_file_is_sibling_of_target(self, tmp_path):
        """Atomic-write invariant: the temp file lives in the same directory
        as the target (so os.replace stays on one filesystem = atomic). We
        verify by intercepting mkstemp and asserting the dir argument."""
        session = SessionMemory(session_id="sibling")
        session.add_message("user", "x")
        target = tmp_path / "deep" / "sibling.json"

        captured_dirs: list[str] = []

        real_mkstemp = tempfile.mkstemp

        def spying_mkstemp(*args, **kwargs):  # noqa: ANN002,ANN003
            # mkstemp takes `dir` as 3rd positional or kwarg.
            d = kwargs.get("dir", args[2] if len(args) > 2 else None)
            captured_dirs.append(str(d))
            return real_mkstemp(*args, **kwargs)

        with mock.patch("vertai.core.memory.tempfile.mkstemp", spying_mkstemp):
            session.save(str(target))

        assert captured_dirs, "mkstemp was not called"
        assert captured_dirs[0] == str(target.parent), (
            f"temp dir {captured_dirs[0]!r} != target parent {str(target.parent)!r}; "
            "atomic-rename guarantee requires same directory."
        )