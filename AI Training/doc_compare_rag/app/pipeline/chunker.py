"""
chunker.py – Split Document objects into overlapping text chunks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.ingestion.loader import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SIZE_DEFAULT = 500
_CHUNK_OVERLAP_DEFAULT = 50

# Matches any sentence-ending punctuation (., !, ?) optionally followed by a
# closing quote, then at least one whitespace character.
_SENTENCE_END_RE = re.compile(r'[.!?]["\']?\s+')


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A text chunk derived from a Document, ready for embedding.

    Attributes
    ----------
    text:     The chunk's text content.
    metadata: All metadata from the parent Document plus ``chunk_index``
              (0-based position within the document) and ``total_chunks``
              (total number of chunks produced from the same document).
    score:    Cosine similarity score populated by vector-store search;
              0.0 when the chunk has not been retrieved via search.
    """

    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


# ---------------------------------------------------------------------------
# TextChunker class
# ---------------------------------------------------------------------------


class TextChunker:
    """Split Document objects into overlapping chunks using a sliding window.

    Chunk boundaries are snapped to the nearest sentence ending within a
    look-back window, so chunks end on complete sentences wherever possible.
    When no sentence boundary is found the hard character limit is used.

    Parameters
    ----------
    chunk_size:
        Maximum character length of each chunk.
        Falls back to the ``CHUNK_SIZE`` environment variable (default 500).
    chunk_overlap:
        Number of characters shared between consecutive chunks.
        Falls back to the ``CHUNK_OVERLAP`` environment variable (default 50).

    Raises
    ------
    ValueError
        If ``chunk_overlap >= chunk_size``.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or int(
            os.environ.get("CHUNK_SIZE", _CHUNK_SIZE_DEFAULT)
        )
        self.chunk_overlap = chunk_overlap or int(
            os.environ.get("CHUNK_OVERLAP", _CHUNK_OVERLAP_DEFAULT)
        )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List["Document"]) -> List[Chunk]:
        """Split each Document into overlapping Chunk objects.

        Processes documents in order. Each output Chunk inherits all metadata
        from its parent Document and additionally contains:

        - ``chunk_index``: 0-based position within the document.
        - ``total_chunks``: total number of chunks produced from that document.

        Parameters
        ----------
        documents:
            List of ``Document`` objects produced by the ingestion layer.

        Returns
        -------
        List[Chunk]
            Flat list of Chunk objects in document order.
        """
        result: List[Chunk] = []
        for doc in documents:
            raw_chunks = self._sliding_window(doc.text)
            total = len(raw_chunks)
            for idx, text in enumerate(raw_chunks):
                result.append(
                    Chunk(
                        text=text,
                        metadata={
                            **doc.metadata,
                            "chunk_index": idx,
                            "total_chunks": total,
                        },
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sliding_window(self, text: str) -> List[str]:
        """Produce overlapping chunks from *text*.

        For each window the method tries to snap the end position to the last
        sentence boundary within a look-back region of
        ``min(200, chunk_size // 4)`` characters.  If no boundary is found the
        hard character limit is used as-is.
        """
        if not text:
            return []

        # How far back from the hard cut to search for a sentence boundary.
        snap_back = min(200, self.chunk_size // 4)
        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            hard_end = min(start + self.chunk_size, text_len)

            # Prefer ending at a sentence boundary (only look if not at EOF).
            if hard_end < text_len:
                end = self._last_sentence_end(text, hard_end, snap_back)
            else:
                end = hard_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= text_len:
                break

            # Slide forward, keeping `chunk_overlap` chars from the current
            # chunk.  Guard against degenerate cases where the snapped end is
            # so early that subtracting the overlap would not advance start.
            next_start = end - self.chunk_overlap
            start = next_start if next_start > start else end

        return chunks

    @staticmethod
    def _last_sentence_end(text: str, pos: int, look_back: int) -> int:
        """Return the index just *after* the last sentence-ending punctuation
        found in ``text[pos - look_back : pos]``.

        Returns *pos* unchanged when no sentence boundary is found, so the
        caller can fall back to the hard character cut.
        """
        search_start = max(0, pos - look_back)
        segment = text[search_start:pos]
        matches = list(_SENTENCE_END_RE.finditer(segment))
        if matches:
            last = matches[-1]
            # +1 to include the [.!?] but exclude the trailing whitespace.
            return search_start + last.start() + 1
        return pos


# ---------------------------------------------------------------------------
# Legacy functional API (used by streamlit_app.py)
# ---------------------------------------------------------------------------


def chunk_text(
    pages: List[str],
    chunk_size: int = _CHUNK_SIZE_DEFAULT,
    overlap: int = _CHUNK_OVERLAP_DEFAULT,
) -> List[str]:
    """Split joined page text into fixed-size overlapping chunks.

    .. deprecated::
        Prefer :class:`TextChunker` which works with ``Document`` objects and
        returns typed ``Chunk`` instances with metadata.

    Parameters
    ----------
    pages:      List of page/section text strings from the ingestion layer.
    chunk_size: Maximum number of characters per chunk.
    overlap:    Number of characters to overlap between consecutive chunks.

    Returns
    -------
    List[str]
        List of text chunk strings.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    from app.ingestion.loader import Document  # lazy import

    docs = [Document(text=p, metadata={}) for p in pages]
    return [c.text for c in chunker.chunk(docs)]
