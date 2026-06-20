"""Tests for the TextSplitter abstraction (S3).

Pure logic -> real assertions on actual chunking behavior. No mocks.
"""

from __future__ import annotations

import pytest

from vertai.core.text_splitter import (
    FixedLengthSplitter,
    RecursiveTextSplitter,
    TextSplitter,
)


class TestTextSplitterABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            TextSplitter()  # type: ignore[abstract]


class TestRecursiveTextSplitter:
    def test_empty_text_returns_empty(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split("") == []
        assert splitter.split("   \n  \t  ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "short text"
        chunks = splitter.split(text)
        assert chunks == ["short text"]

    def test_respects_chunk_size(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=20, chunk_overlap=0)
        text = "word " * 100  # 500 chars
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert all(len(c) <= 20 for c in chunks)

    def test_prefers_paragraph_breaks(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=30, chunk_overlap=0)
        text = "first paragraph here.\n\nsecond paragraph here."
        chunks = splitter.split(text)
        # Should not split inside a paragraph that fits; both paras together
        # exceed 30 so they split at the \n\n boundary.
        assert len(chunks) == 2
        assert "first paragraph" in chunks[0]
        assert "second paragraph" in chunks[1]

    def test_overlap_carries_between_chunks(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=10, chunk_overlap=5)
        # No separators present -> hard char split with overlap.
        text = "abcdefghij" * 3  # 30 chars, no separators
        chunks = splitter.split(text)
        assert len(chunks) > 1
        # The overlap tail of chunk i should appear at the start of chunk i+1.
        for i in range(len(chunks) - 1):
            tail = chunks[i][-5:]
            assert chunks[i + 1].startswith(tail)

    def test_all_chunks_nonempty_and_stripped(self) -> None:
        splitter = RecursiveTextSplitter(chunk_size=15, chunk_overlap=3)
        text = "\n\nhello world\n\nfoo bar baz\n\n" * 5
        chunks = splitter.split(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c != ""
            assert c == c.strip()

    def test_custom_separators(self) -> None:
        splitter = RecursiveTextSplitter(
            chunk_size=5, chunk_overlap=0, separators=["|", ""]
        )
        text = "ab|cd|ef"
        chunks = splitter.split(text)
        # Splits on "|" then merges up to size 5.
        assert all(len(c) <= 5 for c in chunks)
        joined = "".join(c.replace("|", "") for c in chunks)
        # No content lost (separators may be dropped, but letters preserved).
        assert "ab" in joined and "cd" in joined and "ef" in joined

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveTextSplitter(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveTextSplitter(chunk_size=10, chunk_overlap=10)

    def test_no_content_loss_for_simple_case(self) -> None:
        # When text fits in one chunk, it is returned verbatim.
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog."
        assert splitter.split(text) == [text]


class TestFixedLengthSplitter:
    def test_empty_returns_empty(self) -> None:
        splitter = FixedLengthSplitter(chunk_size=10, chunk_overlap=2)
        assert splitter.split("") == []

    def test_short_text_single_chunk(self) -> None:
        splitter = FixedLengthSplitter(chunk_size=100, chunk_overlap=10)
        assert splitter.split("hello") == ["hello"]

    def test_chunks_respect_size(self) -> None:
        splitter = FixedLengthSplitter(chunk_size=10, chunk_overlap=0)
        text = "x" * 35
        chunks = splitter.split(text)
        assert all(len(c) <= 10 for c in chunks)
        # 35 chars / 10 = 4 chunks (last one short).
        assert len(chunks) == 4

    def test_overlap_creates_more_chunks(self) -> None:
        no_overlap = FixedLengthSplitter(chunk_size=10, chunk_overlap=0)
        with_overlap = FixedLengthSplitter(chunk_size=10, chunk_overlap=4)
        text = "x" * 40
        assert len(with_overlap.split(text)) > len(no_overlap.split(text))

    def test_overlap_content_shared(self) -> None:
        splitter = FixedLengthSplitter(chunk_size=10, chunk_overlap=4)
        text = "abcdefghij" * 3  # 30 chars
        chunks = splitter.split(text)
        for i in range(len(chunks) - 1):
            tail = chunks[i][-4:]
            assert chunks[i + 1].startswith(tail)

    def test_invalid_args_raise(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            FixedLengthSplitter(chunk_size=-1)
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedLengthSplitter(chunk_size=5, chunk_overlap=5)
