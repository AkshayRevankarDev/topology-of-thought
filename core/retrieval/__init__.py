"""
core.retrieval — vector store + chunking for the RAG layer.

Exposes:
    Chunker        chunk long-form text into overlapping windows
    VectorIndex    persistent local vector store backed by Chroma
    Indexer        glue: documents -> chunks -> vectors -> store
"""

from core.retrieval.chunker import Chunker, Chunk
from core.retrieval.vector_index import VectorIndex
from core.retrieval.indexer import Indexer

__all__ = ['Chunker', 'Chunk', 'VectorIndex', 'Indexer']
