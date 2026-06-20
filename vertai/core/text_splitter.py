"""Text splitter abstraction (S3).

Replaces the hard-coded paragraph chunking previously inlined in
``vertai.scenarios.knowledge_qa``. ``docs/ARCHITECTURE.md`` 3.5 defines the
contract. Two concrete strategies ship in 1.0: :class:`RecursiveTextSplitter`
(default; splits by a separator hierarchy) and :class:`FixedLengthSplitter`
(fixed-length chunks with overlap). ``SemanticTextSplitter`` lands in 1.x.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = [
    "TextSplitter",
    "RecursiveTextSplitter",
    "FixedLengthSplitter",
]


class TextSplitter(ABC):
    """Text splitter abstraction. Splits a long text into retrievable chunks."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split ``text`` into a list of non-empty chunks."""


class RecursiveTextSplitter(TextSplitter):
    """Split text by recursively trying a hierarchy of separators.

    Tries each separator in order (paragraph, line, sentence, word, character);
    when a piece is still longer than ``chunk_size`` it is split further by the
    next separator. The final fallback is a hard character split. Small adjacent
    pieces are merged greedily up to ``chunk_size``; ``chunk_overlap`` carries a
    tail of the previous chunk into the next to preserve boundary context.

    Every emitted chunk is non-empty, stripped, and no longer than
    ``chunk_size`` (the overlap tail is truncated when needed to keep this
    invariant).
    """

    _DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "。", " ", ""]

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not (0 <= chunk_overlap < chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = (
            list(separators) if separators is not None else list(self._DEFAULT_SEPARATORS)
        )

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        chunks = self._split(text, self.separators)
        return [c for c in chunks if c.strip()]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []

        # Pick the first separator present in the text; "" means char-level.
        sep = separators[-1]
        new_separators: list[str] = []
        for i, candidate in enumerate(separators):
            if candidate == "":
                sep = ""
                new_separators = []
                break
            if candidate in text:
                sep = candidate
                new_separators = separators[i + 1:]
                break

        if sep == "":
            # Character-level fallback.
            return self._hard_split(text)

        splits = text.split(sep)
        good: list[str] = []
        for piece in splits:
            if len(piece) < self.chunk_size:
                good.append(piece)
            else:
                if good:
                    final_chunks.extend(self._merge(good, sep))
                    good = []
                if not new_separators:
                    final_chunks.extend(self._hard_split(piece))
                else:
                    final_chunks.extend(self._split(piece, new_separators))
        if good:
            final_chunks.extend(self._merge(good, sep))
        return final_chunks

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            candidate = piece if not current else current + sep + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if self.chunk_overlap > 0 and current:
                    tail = current[-self.chunk_overlap:]
                    next_chunk = tail + (sep if sep else "") + piece
                    if len(next_chunk) <= self.chunk_size:
                        current = next_chunk
                    else:
                        # Overlap would overflow; drop overlap for this chunk.
                        current = piece
                else:
                    current = piece
        if current:
            chunks.append(current)
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        if not text:
            return []
        if self.chunk_overlap == 0:
            return [
                text[i:i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            chunk = text[start:start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= text_len:
                break
            start += step
        return chunks


class FixedLengthSplitter(TextSplitter):
    """Split text into fixed-length chunks with overlap.

    ``chunk_size`` is the maximum chunk length (in characters); ``chunk_overlap``
    is the number of characters carried from the end of one chunk into the next.
    """

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not (0 <= chunk_overlap < chunk_size):
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1
        chunks: list[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end >= text_len:
                break
            start += step
        return chunks
