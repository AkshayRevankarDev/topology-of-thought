"""
deduplicator.py — Semantic deduplication of graph nodes using dense embeddings.

Two nodes are considered duplicates when their label embeddings exceed a
cosine-similarity threshold.  The surviving node merges the description,
page references and confidence of the removed one.  Edges are rewired to
point to the canonical node.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.graph_state import Edge, GraphState, Node

logger = logging.getLogger(__name__)

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = None  # module-level lazy singleton


def _get_embedder():
    """Lazily load the SentenceTransformer model (singleton).

    Returns:
        A loaded :class:`sentence_transformers.SentenceTransformer` instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        RuntimeError: If the model cannot be loaded.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required.  "
            "Install with: pip install sentence-transformers"
        ) from exc
    try:
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
        logger.info("Loaded embedding model '%s'.", _EMBED_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SentenceTransformer model '{_EMBED_MODEL_NAME}': {exc}"
        ) from exc
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """Compute L2-normalised dense embeddings for a list of strings.

    Args:
        texts: Input strings.  Empty strings are replaced with a single space
            to avoid model errors.

    Returns:
        Float32 numpy array of shape ``(len(texts), embedding_dim)`` with each
        row unit-normalised.
    """
    cleaned = [t if t.strip() else " " for t in texts]
    model = _get_embedder()
    embeddings: np.ndarray = model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine-similarity matrix for *embeddings*.

    Because embeddings are already L2-normalised, this is simply the dot
    product matrix.

    Args:
        embeddings: Shape ``(n, d)`` float32 array of unit-norm vectors.

    Returns:
        Shape ``(n, n)`` float32 similarity matrix.
    """
    return (embeddings @ embeddings.T).astype(np.float32)


def find_duplicate_pairs(
    nodes: List[Node],
    threshold: float = 0.92,
) -> List[Tuple[str, str]]:
    """Return pairs of node ids whose label embeddings exceed *threshold*.

    Uses greedy clustering: once a node is marked as a duplicate of an earlier
    one it is excluded from being a canonical representative.

    Args:
        nodes: List of Node objects to compare.
        threshold: Cosine-similarity threshold; nodes above this are merged.

    Returns:
        List of ``(keep_id, remove_id)`` pairs.  The first element in each
        pair is the canonical (surviving) node.
    """
    if len(nodes) < 2:
        return []

    labels = [n.label for n in nodes]
    embeddings = embed_texts(labels)
    # Store embeddings on the node objects for later reuse.
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb.tolist()

    sim = cosine_similarity_matrix(embeddings)
    n = len(nodes)
    merged: set[int] = set()
    pairs: List[Tuple[str, str]] = []

    for i in range(n):
        if i in merged:
            continue
        for j in range(i + 1, n):
            if j in merged:
                continue
            if sim[i, j] >= threshold:
                pairs.append((nodes[i].id, nodes[j].id))
                merged.add(j)

    return pairs


def _merge_nodes(canonical: Node, duplicate: Node) -> Node:
    """Merge *duplicate* metadata into *canonical* in-place.

    Args:
        canonical: The node that will survive.
        duplicate: The node to be absorbed.

    Returns:
        The updated *canonical* node.
    """
    # Keep the longer description.
    if len(duplicate.description) > len(canonical.description):
        canonical.description = duplicate.description
    # Union page references.
    existing_refs = set(canonical.page_refs)
    for ref in duplicate.page_refs:
        if ref not in existing_refs:
            canonical.page_refs.append(ref)
    # Boost confidence to the maximum observed.
    canonical.confidence = max(canonical.confidence, duplicate.confidence)
    # Merge metadata dicts (canonical takes precedence on key conflicts).
    merged_meta = {**duplicate.metadata, **canonical.metadata}
    canonical.metadata = merged_meta
    return canonical


def deduplicate_graph(graph: GraphState, threshold: float = 0.92) -> GraphState:
    """Remove semantically duplicate nodes from *graph* in-place.

    Steps:
    1. Compute label embeddings for all nodes.
    2. Identify duplicate pairs above *threshold*.
    3. For each pair, merge the duplicate into the canonical node and rewire
       all edges that referenced the duplicate.

    Args:
        graph: The :class:`~core.graph_state.GraphState` to deduplicate.
        threshold: Cosine-similarity threshold for merging (0–1).

    Returns:
        The same *graph* instance (mutated in-place) for convenient chaining.
    """
    nodes = graph.nodes
    if not nodes:
        return graph

    logger.info("Deduplicating %d nodes at threshold %.2f…", len(nodes), threshold)
    pairs = find_duplicate_pairs(nodes, threshold=threshold)

    if not pairs:
        logger.info("No duplicates found.")
        return graph

    logger.info("Found %d duplicate pair(s) to merge.", len(pairs))

    for keep_id, remove_id in pairs:
        canonical = graph.get_node(keep_id)
        duplicate = graph.get_node(remove_id)
        if canonical is None or duplicate is None:
            # Already removed by a prior merge cascade.
            continue

        _merge_nodes(canonical, duplicate)

        # Rewire edges: replace remove_id references with keep_id.
        for edge in graph.edges:
            if edge.source_id == remove_id:
                edge.source_id = keep_id
            if edge.target_id == remove_id:
                edge.target_id = keep_id

        # Remove self-loops that may have been created by the rewire.
        self_loops = [e.id for e in graph.edges if e.source_id == e.target_id]
        for edge_id in self_loops:
            graph.remove_edge(edge_id)

        # Remove the duplicate node (edges are already rewired).
        try:
            graph._nodes.pop(remove_id, None)
        except Exception as exc:
            logger.warning("Could not remove duplicate node '%s': %s", remove_id, exc)

    logger.info(
        "After deduplication: %d nodes, %d edges.",
        len(graph.nodes),
        len(graph.edges),
    )
    return graph


def embed_all_nodes(graph: GraphState) -> None:
    """Populate the ``embedding`` field for every node that lacks one.

    This is useful as a standalone pre-computation step before running
    nearest-neighbour queries at runtime.

    Args:
        graph: GraphState whose nodes will be embedded in-place.
    """
    unembedded = [n for n in graph.nodes if n.embedding is None]
    if not unembedded:
        return
    labels = [n.label for n in unembedded]
    embeddings = embed_texts(labels)
    for node, emb in zip(unembedded, embeddings):
        node.embedding = emb.tolist()
    logger.info("Embedded %d nodes.", len(unembedded))
