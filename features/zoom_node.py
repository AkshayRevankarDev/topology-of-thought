"""
zoom_node.py — Node explosion: expand a node into a sub-graph view.

When a user pinches and holds on a node, this module queries Ollama to
generate a more detailed decomposition of that concept, then creates child
nodes arranged in a radial pattern around the parent.  The parent node is
marked ``expanded=True`` and child nodes carry a ``parent_id`` metadata key.

Collapsing the sub-graph removes the child nodes and resets ``expanded``.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import requests

from core.graph_state import Edge, GraphState, Node

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """\
You are a concept decomposition assistant.
Break down the concept "{concept}" into 4 to 6 more specific sub-concepts or
components.  For each sub-concept provide a one-sentence description.

Return ONLY valid JSON with no markdown fences:
{{
  "sub_concepts": [
    {{"label": "<short name>", "description": "<one sentence>"}},
    ...
  ]
}}
"""


def _call_ollama_decompose(
    concept: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> List[dict]:
    """Ask Ollama to decompose a concept into sub-concepts.

    Args:
        concept: The concept label to decompose.
        model: Ollama model tag.
        base_url: Ollama server base URL.
        timeout: Request timeout in seconds.

    Returns:
        List of ``{"label": str, "description": str}`` dicts.

    Raises:
        ConnectionError: If Ollama is unreachable.
        RuntimeError: If the server returns an error or unparseable output.
    """
    import json, re

    prompt = _DECOMPOSE_PROMPT.format(concept=concept)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json().get("response", "")
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise RuntimeError(f"No JSON in Ollama response for '{concept}'.")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON parse error for '{concept}': {exc}") from exc

    return data.get("sub_concepts", [])


def _radial_positions(
    cx: float,
    cy: float,
    n: int,
    radius: float = 250.0,
    start_angle_deg: float = 0.0,
) -> List[tuple[float, float]]:
    """Compute evenly-spaced positions around a circle.

    Args:
        cx: X coordinate of the circle centre.
        cy: Y coordinate of the circle centre.
        n: Number of positions.
        radius: Circle radius in pixels.
        start_angle_deg: Starting angle in degrees (0 = right / 3 o'clock).

    Returns:
        List of ``(x, y)`` tuples.
    """
    positions = []
    for i in range(n):
        angle = math.radians(start_angle_deg + 360.0 * i / n)
        positions.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return positions


def expand_node(
    graph: GraphState,
    node_id: str,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    sub_radius: float = 250.0,
) -> List[Node]:
    """Expand a node into a sub-graph of decomposed concepts.

    If the node is already expanded, this is a no-op.

    Args:
        graph: The active GraphState to mutate.
        node_id: ID of the node to expand.
        model: Ollama model tag.
        base_url: Ollama server URL.
        sub_radius: Radial distance for child node placement.

    Returns:
        List of newly created child Nodes.

    Raises:
        KeyError: If *node_id* does not exist in the graph.
        ConnectionError: If Ollama is unreachable.
    """
    parent = graph.get_node(node_id)
    if parent is None:
        raise KeyError(f"Node '{node_id}' not found in graph.")
    if parent.expanded:
        logger.debug("Node '%s' is already expanded.", parent.label)
        return []

    sub_concepts = _call_ollama_decompose(parent.label, model=model, base_url=base_url)
    if not sub_concepts:
        logger.warning("No sub-concepts returned for '%s'.", parent.label)
        return []

    positions = _radial_positions(
        parent.x, parent.y, n=len(sub_concepts), radius=sub_radius
    )

    child_nodes: List[Node] = []
    for sc, (px, py) in zip(sub_concepts, positions):
        label = str(sc.get("label", "")).strip()
        if not label:
            continue
        child = Node(
            label=label,
            description=str(sc.get("description", "")),
            source_paper=parent.source_paper,
            confidence=parent.confidence * 0.9,  # children inherit slightly lower confidence
            x=px,
            y=py,
            metadata={"parent_id": parent.id, "is_child_node": True},
        )
        graph.add_node(child)
        child_nodes.append(child)

        try:
            graph.add_edge(
                Edge(
                    source_id=parent.id,
                    target_id=child.id,
                    relation="decomposes_to",
                    weight=0.8,
                    confidence=parent.confidence,
                    source_paper=parent.source_paper,
                )
            )
        except ValueError as exc:
            logger.warning("Could not add child edge: %s", exc)

    parent.expanded = True
    logger.info(
        "Expanded node '%s' into %d sub-concepts.", parent.label, len(child_nodes)
    )
    return child_nodes


def collapse_node(graph: GraphState, node_id: str) -> int:
    """Remove all child nodes and edges created by :func:`expand_node`.

    Args:
        graph: The GraphState to mutate.
        node_id: ID of the expanded parent node.

    Returns:
        Number of child nodes removed.

    Raises:
        KeyError: If *node_id* does not exist.
    """
    parent = graph.get_node(node_id)
    if parent is None:
        raise KeyError(f"Node '{node_id}' not found.")

    if not parent.expanded:
        return 0

    children = [
        n for n in graph.nodes
        if n.metadata.get("parent_id") == node_id
        and n.metadata.get("is_child_node") is True
    ]

    count = 0
    for child in children:
        try:
            graph.remove_node(child.id)  # also removes attached edges
            count += 1
        except KeyError:
            pass

    parent.expanded = False
    logger.info("Collapsed node '%s', removed %d children.", parent.label, count)
    return count


def toggle_expansion(
    graph: GraphState,
    node_id: str,
    **expand_kwargs,
) -> List[Node]:
    """Toggle between expanded and collapsed state for a node.

    Args:
        graph: The active GraphState.
        node_id: Target node ID.
        **expand_kwargs: Forwarded to :func:`expand_node` when expanding.

    Returns:
        Newly created child nodes (empty list if collapsing).
    """
    parent = graph.get_node(node_id)
    if parent is None:
        raise KeyError(f"Node '{node_id}' not found.")

    if parent.expanded:
        collapse_node(graph, node_id)
        return []
    else:
        return expand_node(graph, node_id, **expand_kwargs)
