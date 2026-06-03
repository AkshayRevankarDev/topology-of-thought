"""
chunker.py — Split long-form text into overlapping windows for retrieval.

Two strategies, used in order:
  1. Try to split on sentence boundaries (regex on .!?\\n) so chunks read as
     coherent passages — the LLM gets better synthesis input.
  2. Fall back to a hard character window if no sentence break is near the
     target length.

Each emitted Chunk carries:
  text         the chunk text
  source_id    stable identifier of the originating document (e.g. PDF path)
  source_meta  free-form dict — typically {'page': int, 'paper': str}
  chunk_index  position of the chunk within the document
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional


_SENTENCE_BREAK = re.compile(r'(?<=[\.!?])\s+(?=[A-Z0-9])')


@dataclass
class Chunk:
    """One indexable text window with provenance."""
    text: str
    source_id: str
    chunk_index: int
    source_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic id used as the primary key in the vector store."""
        return f'{self.source_id}::{self.chunk_index}'


class Chunker:
    """Splits raw text into overlapping retrieval-friendly Chunks.

    Args:
        target_chars: Approximate chunk length in characters (default 1200,
            roughly 250–300 tokens with the default tokenizer).
        overlap_chars: Number of overlap characters between consecutive chunks
            so context isn't lost at boundaries.
    """

    def __init__(self, target_chars: int = 1200, overlap_chars: int = 200) -> None:
        if overlap_chars >= target_chars:
            raise ValueError('overlap_chars must be smaller than target_chars')
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(
        self,
        text: str,
        source_id: str,
        source_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Return all chunks for one document.

        Args:
            text: The full document text.
            source_id: Stable identifier for the document.
            source_meta: Extra provenance (e.g. {'paper': 'attention.pdf'}).
        """
        text = (text or '').strip()
        if not text:
            return []

        chunks: List[Chunk] = []
        meta = source_meta or {}
        cursor = 0
        idx = 0
        n = len(text)
        while cursor < n:
            window_end = min(cursor + self.target_chars, n)
            # Try to back off to a sentence break near the end of the window.
            slice_text = text[cursor:window_end]
            if window_end < n:
                breaks = list(_SENTENCE_BREAK.finditer(slice_text))
                if breaks:
                    # Keep the latest break that leaves at least half the window.
                    cutoff = breaks[-1].end()
                    if cutoff >= self.target_chars * 0.5:
                        slice_text = slice_text[:cutoff]
            slice_text = slice_text.strip()
            if slice_text:
                chunks.append(Chunk(
                    text=slice_text,
                    source_id=source_id,
                    chunk_index=idx,
                    source_meta=dict(meta),
                ))
                idx += 1
            cursor += max(len(slice_text) - self.overlap_chars, 1)
        return chunks

    def chunk_many(self, documents: Iterable[Dict[str, Any]]) -> Iterator[Chunk]:
        """Chunk a stream of {'text', 'source_id', 'source_meta'} dicts."""
        for doc in documents:
            for c in self.chunk(
                text=doc.get('text', ''),
                source_id=doc['source_id'],
                source_meta=doc.get('source_meta'),
            ):
                yield c
