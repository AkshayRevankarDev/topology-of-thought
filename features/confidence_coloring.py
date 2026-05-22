"""
confidence_coloring.py — Visual confidence encoding for graph nodes.

Maps each node's ``confidence`` scalar to a colour and normalises confidence
scores across the whole graph.  Designed to be queried by the renderer each
frame to produce live visual feedback.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from core.graph_state import GraphState, Node


# ---------------------------------------------------------------------------
# Colour palettes — all colours in normalised RGB float tuples (r, g, b) ∈ [0,1]
# ---------------------------------------------------------------------------

def lerp_color(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    """Linearly interpolate between two RGB colour tuples.

    Args:
        a: Start colour ``(r, g, b)`` in [0, 1].
        b: End colour ``(r, g, b)`` in [0, 1].
        t: Interpolation factor in [0, 1].

    Returns:
        Interpolated colour ``(r, g, b)``.
    """
    t = max(0.0, min(1.0, t))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def confidence_to_rgb(
    confidence: float,
    low_color: Tuple[float, float, float] = (0.85, 0.15, 0.10),   # red
    mid_color: Tuple[float, float, float] = (0.95, 0.75, 0.10),   # amber
    high_color: Tuple[float, float, float] = (0.15, 0.75, 0.95),  # cyan-blue
) -> Tuple[float, float, float]:
    """Map a confidence scalar to an RGB colour using a three-stop gradient.

    The gradient is:
      - 0.0 → *low_color*  (typically red)
      - 0.5 → *mid_color*  (typically amber)
      - 1.0 → *high_color* (typically cyan)

    Args:
        confidence: Float in [0, 1].
        low_color: RGB colour for minimum confidence.
        mid_color: RGB colour at the midpoint.
        high_color: RGB colour for maximum confidence.

    Returns:
        Interpolated RGB tuple with each channel in [0, 1].
    """
    c = max(0.0, min(1.0, confidence))
    if c <= 0.5:
        return lerp_color(low_color, mid_color, c * 2.0)
    else:
        return lerp_color(mid_color, high_color, (c - 0.5) * 2.0)


def confidence_to_bgr_uint8(confidence: float) -> Tuple[int, int, int]:
    """Return an OpenCV-compatible BGR uint8 colour for *confidence*.

    Args:
        confidence: Float in [0, 1].

    Returns:
        BGR tuple of uint8 values in [0, 255].
    """
    r, g, b = confidence_to_rgb(confidence)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Graph-wide normalisation
# ---------------------------------------------------------------------------

def normalise_confidences(graph: GraphState) -> Dict[str, float]:
    """Normalise all node confidences to [0, 1] relative to the graph range.

    If all nodes have identical confidence scores, all normalised values are
    set to 1.0.

    Args:
        graph: The GraphState to inspect.

    Returns:
        Mapping from ``node.id`` to its normalised confidence score.
    """
    nodes = graph.nodes
    if not nodes:
        return {}

    raw = [n.confidence for n in nodes]
    min_c = min(raw)
    max_c = max(raw)
    span = max_c - min_c

    if span == 0.0:
        return {n.id: 1.0 for n in nodes}

    return {n.id: (n.confidence - min_c) / span for n in nodes}


def colour_map_for_graph(
    graph: GraphState,
    normalise: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Build a complete ``node_id → RGB`` colour map for the whole graph.

    Args:
        graph: The source GraphState.
        normalise: If True, rescale confidences to [0, 1] before mapping.

    Returns:
        Dict mapping each node id to its RGB float colour tuple.
    """
    if normalise:
        conf_map = normalise_confidences(graph)
    else:
        conf_map = {n.id: n.confidence for n in graph.nodes}

    return {node_id: confidence_to_rgb(c) for node_id, c in conf_map.items()}


def apply_colours_to_nodes(graph: GraphState, normalise: bool = True) -> None:
    """Store the computed colour in each node's ``metadata["color_rgb"]`` field.

    Useful for serialising colour state into the session JSON so the renderer
    can read it without recomputing every frame.

    Args:
        graph: The GraphState to annotate.
        normalise: Whether to normalise confidences first.
    """
    colour_map = colour_map_for_graph(graph, normalise=normalise)
    for node in graph.nodes:
        rgb = colour_map.get(node.id, (0.5, 0.5, 0.5))
        node.metadata["color_rgb"] = list(rgb)


# ---------------------------------------------------------------------------
# Pulse / animation helpers
# ---------------------------------------------------------------------------

def pulsed_alpha(
    base_alpha: float,
    amplitude: float,
    frequency: float,
    t: float,
) -> float:
    """Compute a sinusoidally pulsing alpha value for animated highlighting.

    Args:
        base_alpha: Centre alpha value in [0, 1].
        amplitude: Half-amplitude of the pulse (e.g. 0.15).
        frequency: Pulses per second (e.g. 2.0).
        t: Current time in seconds (e.g. ``time.time()``).

    Returns:
        Alpha value clamped to [0, 1].
    """
    import math
    alpha = base_alpha + amplitude * math.sin(2.0 * math.pi * frequency * t)
    return max(0.0, min(1.0, alpha))
