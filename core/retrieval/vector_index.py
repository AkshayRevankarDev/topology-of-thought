"""
vector_index.py — Persistent local vector store backed by Chroma.

Chroma chosen because:
  * Local, file-backed (PersistentClient), no server to manage.
  * Cosine similarity out of the box.
  * Metadata filters, which we use to scope retrieval to a session or source.

If chromadb is not installed, the class raises at construction time with a
clear hint to `pip install chromadb`.

The embedding model is reused from `core.deduplicator` so chunk vectors live
in the same space as the GraphState node embeddings — that lets the chat layer
do "which graph nodes are nearest to my retrieved chunks?" trivially.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Retrieved:
    """One retrieval hit returned by :meth:`VectorIndex.query`."""
    chunk_id: str
    text: str
    source_id: str
    source_meta: Dict[str, Any]
    score: float  # cosine similarity in [-1, 1]; higher = closer


class VectorIndex:
    """File-backed Chroma vector store for retrieval chunks.

    Args:
        persist_dir: Filesystem directory to write the store to.
        collection: Logical collection name (e.g. session id).
        embed_model: Name of the SentenceTransformer model. MUST match the
            model used by core.deduplicator so chunks and graph nodes share
            an embedding space.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection: str = 'default',
        embed_model: str = 'all-MiniLM-L6-v2',
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                'chromadb is required for the retrieval layer. '
                'Install it with: pip install chromadb'
            ) from exc

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={'hnsw:space': 'cosine'},
        )
        self.embed_model_name = embed_model
        self._embedder = None  # lazy

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        # Reuse the singleton from core.deduplicator if it's already loaded
        # so we don't double-allocate the model on disk / VRAM.
        try:
            from core import deduplicator
            self._embedder = deduplicator._get_embedder()  # noqa: SLF001
            return self._embedder
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        vecs = embedder.encode(list(texts), show_progress_bar=False)
        if isinstance(vecs, np.ndarray):
            return vecs.astype(float).tolist()
        return [list(map(float, v)) for v in vecs]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def add(self, chunks: Iterable[Chunk], batch_size: int = 64) -> int:
        """Add chunks to the store. Returns the number of chunks added."""
        ids: List[str] = []
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        added = 0

        def _flush() -> None:
            nonlocal added
            if not ids:
                return
            embs = self._embed(texts)
            self._collection.upsert(
                ids=list(ids),
                documents=list(texts),
                metadatas=list(metas),
                embeddings=embs,
            )
            added += len(ids)
            ids.clear()
            texts.clear()
            metas.clear()

        for c in chunks:
            meta = {'source_id': c.source_id, 'chunk_index': c.chunk_index}
            # Chroma metadata values must be primitive types — flatten.
            for k, v in c.source_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            ids.append(c.chunk_id)
            texts.append(c.text)
            metas.append(meta)
            if len(ids) >= batch_size:
                _flush()
        _flush()
        return added

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        k: int = 6,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Retrieved]:
        """Return the top-k chunks closest to *text*.

        Args:
            text: Query string.
            k: Maximum number of hits to return.
            where: Optional Chroma metadata filter (e.g. {'paper': 'a.pdf'}).
        """
        if not text.strip():
            return []
        emb = self._embed([text])[0]
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=k,
            where=where or None,
            include=['documents', 'metadatas', 'distances'],
        )
        out: List[Retrieved] = []
        ids   = res.get('ids',        [[]])[0]
        docs  = res.get('documents',  [[]])[0]
        metas = res.get('metadatas',  [[]])[0]
        dists = res.get('distances',  [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)  # chroma returns cosine distance
            out.append(Retrieved(
                chunk_id=cid,
                text=doc or '',
                source_id=str(meta.get('source_id', '')),
                source_meta=dict(meta or {}),
                score=score,
            ))
        return out

    def count(self) -> int:
        """Return the number of chunks currently in the collection."""
        return int(self._collection.count())
