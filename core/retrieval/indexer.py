"""
indexer.py — Glue layer: documents -> chunks -> vector store.

A document is any dict with at least:
    {'text': str, 'source_id': str, 'source_meta': dict}

The same shape is produced by core.ingestion adapters (PDF, mbox, notes …).
Indexer keeps the wiring trivial:

    idx = Indexer.default(session='default')
    idx.add_documents(docs)
    hits = idx.search('what is multi-head attention?')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.retrieval.chunker import Chunker
from core.retrieval.vector_index import VectorIndex, Retrieved

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_DIR = PROJECT_ROOT / 'data' / 'vector_store'


class Indexer:
    """High-level RAG ingestion + retrieval entry point."""

    def __init__(self, vector_index: VectorIndex, chunker: Optional[Chunker] = None):
        self.vector_index = vector_index
        self.chunker = chunker or Chunker()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def default(cls, session: str = 'default') -> 'Indexer':
        """Convenience constructor — file-backed store under data/vector_store."""
        return cls(
            vector_index=VectorIndex(
                persist_dir=DEFAULT_VECTOR_DIR,
                collection=session,
            ),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_documents(self, documents: Iterable[Dict[str, Any]]) -> int:
        """Chunk and embed all documents. Returns total chunks indexed."""
        chunks = list(self.chunker.chunk_many(documents))
        if not chunks:
            return 0
        n = self.vector_index.add(chunks)
        logger.info('Indexed %d chunks from %d docs', n, sum(1 for _ in chunks))
        return n

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 6,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Retrieved]:
        """Return the top-k chunks most relevant to *query*."""
        return self.vector_index.query(query, k=k, where=where)

    def count(self) -> int:
        return self.vector_index.count()
