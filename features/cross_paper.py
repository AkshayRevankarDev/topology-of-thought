"""
cross_paper.py — Dual-PDF comparison and cross-paper edge discovery.

Loads two independently-ingested GraphState objects and finds concept pairs
that are semantically similar across the two papers.  Bridge edges are added
to a merged graph with a ``"cross_paper"`` relation tag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from core.graph_state import Edge, GraphState, Node
from core.pdf_ingestion import ingest_pdf
from core.concept_extractor import build_graph_from_chunks
from core.deduplicator import deduplicate_graph, embed_texts

logger = logging.getLogger(__name__)


def ingest_and_extract(
    pdf_path: str | Path,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    chunk_size: int = 400,
    overlap: int = 80,
) -> GraphState:
    """Full pipeline: PDF → chunks → concepts → deduplicated GraphState.

    Args:
        pdf_path: Path to the PDF file.
        model: Ollama model tag for concept extraction.
        ollama_url: Ollama server URL.
        chunk_size: Word count per chunk.
        overlap: Word overlap between chunks.

    Returns:
        Populated and deduplicated :class:`~core.graph_state.GraphState`.
    """
    chunks = ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    graph = build_graph_from_chunks(chunks, model=model, base_url=ollama_url)
    deduplicate_graph(graph)
    graph.source_papers = [Path(pdf_path).name]
    return graph


def find_cross_paper_pairs(
    graph_a: GraphState,
    graph_b: GraphState,
    threshold: float = 0.80,
) -> List[Tuple[Node, Node, float]]:
    """Find concept pairs across two graphs above a similarity threshold.

    Only node pairs from *different* source papers are compared to avoid
    intra-paper duplicates masquerading as cross-paper links.

    Args:
        graph_a: First paper's GraphState (all nodes tagged ``source_paper``).
        graph_b: Second paper's GraphState.
        threshold: Cosine-similarity threshold for a cross-paper bridge.

    Returns:
        List of ``(node_from_a, node_from_b, similarity)`` triples, sorted
        descending by similarity.
    """
    nodes_a = graph_a.nodes
    nodes_b = graph_b.nodes

    if not nodes_a or not nodes_b:
        return []

    labels_a = [n.label for n in nodes_a]
    labels_b = [n.label for n in nodes_b]

    emb_a = embed_texts(labels_a)
    emb_b = embed_texts(labels_b)

    # Update embeddings on node objects.
    for node, emb in zip(nodes_a, emb_a):
        node.embedding = emb.tolist()
    for node, emb in zip(nodes_b, emb_b):
        node.embedding = emb.tolist()

    # Cross similarity matrix: (len_a, len_b)
    sim_matrix = (emb_a @ emb_b.T).astype(np.float32)

    pairs: List[Tuple[Node, Node, float]] = []
    for i, node_a in enumerate(nodes_a):
        for j, node_b in enumerate(nodes_b):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                pairs.append((node_a, node_b, sim))

    pairs.sort(key=lambda t: t[2], reverse=True)
    logger.info(
        "Found %d cross-paper pairs above threshold %.2f.",
        len(pairs), threshold,
    )
    return pairs


def merge_graphs(
    graph_a: GraphState,
    graph_b: GraphState,
    cross_pairs: List[Tuple[Node, Node, float]],
    session_name: str = "cross_paper",
) -> GraphState:
    """Combine two graphs into one merged GraphState with bridge edges.

    Both node sets are inserted as-is (no intra-merge deduplication here —
    run :func:`~core.deduplicator.deduplicate_graph` afterwards if desired).
    Bridge edges use the ``"cross_paper_bridge"`` relation.

    Args:
        graph_a: First paper's graph.
        graph_b: Second paper's graph.
        cross_pairs: Output of :func:`find_cross_paper_pairs`.
        session_name: Label for the merged session.

    Returns:
        New merged :class:`~core.graph_state.GraphState`.
    """
    merged = GraphState()
    merged.session_name = session_name
    merged.source_papers = list(
        set(graph_a.source_papers + graph_b.source_papers)
    )

    for node in graph_a.nodes:
        merged.add_node(node)
    for edge in graph_a.edges:
        try:
            merged.add_edge(edge)
        except ValueError:
            pass

    for node in graph_b.nodes:
        merged.add_node(node)
    for edge in graph_b.edges:
        try:
            merged.add_edge(edge)
        except ValueError:
            pass

    # Add bridge edges.
    for node_a, node_b, sim in cross_pairs:
        try:
            bridge = Edge(
                source_id=node_a.id,
                target_id=node_b.id,
                relation="cross_paper_bridge",
                weight=sim,
                confidence=sim,
                source_paper="cross_paper",
            )
            merged.add_edge(bridge)
        except ValueError as exc:
            logger.debug("Bridge edge skipped: %s", exc)

    logger.info(
        "Merged graph: %d nodes, %d edges (incl. %d bridges).",
        len(merged.nodes),
        len(merged.edges),
        len(cross_pairs),
    )
    return merged


def compare_pdfs(
    pdf_path_a: str | Path,
    pdf_path_b: str | Path,
    threshold: float = 0.80,
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> GraphState:
    """End-to-end dual-PDF comparison pipeline.

    Args:
        pdf_path_a: First PDF path.
        pdf_path_b: Second PDF path.
        threshold: Cross-paper similarity threshold.
        model: Ollama model tag.
        ollama_url: Ollama server URL.

    Returns:
        Merged :class:`~core.graph_state.GraphState` with bridge edges.
    """
    logger.info("Ingesting PDF A: %s", pdf_path_a)
    graph_a = ingest_and_extract(pdf_path_a, model=model, ollama_url=ollama_url)
    logger.info("Ingesting PDF B: %s", pdf_path_b)
    graph_b = ingest_and_extract(pdf_path_b, model=model, ollama_url=ollama_url)

    pairs = find_cross_paper_pairs(graph_a, graph_b, threshold=threshold)
    merged = merge_graphs(
        graph_a,
        graph_b,
        pairs,
        session_name=f"{Path(pdf_path_a).stem}_vs_{Path(pdf_path_b).stem}",
    )
    return merged
