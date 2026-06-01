"""
graph_renderer.py — Graph rendering for TouchDesigner Script TOP + standalone use.

Aesthetic: white-on-dark, minimal, monospace labels, corner brackets, confidence
coloring (dim blue-grey → bright white).  No glow, no bloom.

AR mode (ar=True)
-----------------
Background is fully transparent (alpha=0).  Only nodes, edges, and labels are
drawn.  Intended for compositing over a live webcam feed so nodes appear to
float in physical space.  Each node is depth-scaled by its distance from the
frame centre, giving a perspective-like depth illusion.

TouchDesigner usage
-------------------
The ``cook(scriptOp)`` function writes an RGBA float32 numpy array to the
Script TOP output each frame.  The GraphState must be stored on
``scriptOp.storage["graph"]`` before the first cook.

Standalone usage
----------------
Call :func:`render_frame` directly to get a BGR uint8 numpy array for display
with OpenCV, or :func:`render_to_rgba` for a float32 RGBA array.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("pip install opencv-python") from exc

from core.graph_state import GraphState, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (all BGR uint8)
# ---------------------------------------------------------------------------
BG_COLOR          = (12, 10, 10)          # #0a0a0c
NODE_COLOR_LO     = (55, 50, 65)          # low-confidence node
NODE_COLOR_HI     = (232, 228, 238)       # high-confidence node
EDGE_COLOR        = (70, 68, 80)          # default edge
LABEL_COLOR       = (148, 143, 158)       # label text
LABEL_SHADOW      = (5,  4,  6)           # label drop-shadow
BRACKET_COLOR     = (52, 50, 65)          # idle bracket
BRACKET_SEL_COLOR = (185, 180, 215)       # selected bracket / ring
SEL_RING_COLOR    = (195, 190, 220)

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
NODE_R_MIN    = 5
NODE_R_MAX    = 16
BRACKET_LEN   = 9
BRACKET_PAD   = 5       # gap between node edge and bracket inner corner
BRACKET_THICK = 1

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT       = cv2.FONT_HERSHEY_PLAIN   # closest to monospace in OpenCV
FONT_SCALE = 0.95
FONT_THICK = 1

# ---------------------------------------------------------------------------
# Edge rendering
# ---------------------------------------------------------------------------
EDGE_ALPHA_MIN = 0.35


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_bgr(
    lo: Tuple[int, int, int],
    hi: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linearly interpolate between two BGR colours.

    Args:
        lo: Start colour.
        hi: End colour.
        t: Factor in [0, 1].

    Returns:
        Interpolated BGR tuple.
    """
    t = max(0.0, min(1.0, t))
    return (
        int(lo[0] + (hi[0] - lo[0]) * t),
        int(lo[1] + (hi[1] - lo[1]) * t),
        int(lo[2] + (hi[2] - lo[2]) * t),
    )


def _conf_color(confidence: float) -> Tuple[int, int, int]:
    """Map node confidence ∈ [0, 1] to a BGR fill colour.

    Args:
        confidence: Confidence score.

    Returns:
        BGR uint8 tuple.
    """
    return _lerp_bgr(NODE_COLOR_LO, NODE_COLOR_HI, confidence)


def _degree_map(graph: GraphState) -> Dict[str, int]:
    """Compute per-node edge count.

    Args:
        graph: The GraphState to analyse.

    Returns:
        Dict mapping node id → degree.
    """
    deg: Dict[str, int] = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        deg[e.source_id] = deg.get(e.source_id, 0) + 1
        deg[e.target_id] = deg.get(e.target_id, 0) + 1
    return deg


def _node_radius(degree: int, max_degree: int) -> int:
    """Return pixel radius scaled by node degree.

    Args:
        degree: This node's edge count.
        max_degree: Maximum degree in the graph.

    Returns:
        Radius clamped to [NODE_R_MIN, NODE_R_MAX].
    """
    t = degree / max(max_degree, 1)
    return NODE_R_MIN + int(t * (NODE_R_MAX - NODE_R_MIN))


def _draw_brackets(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw four L-shaped corner brackets (BGR canvas)."""
    _draw_brackets_bgra(canvas, cx, cy, r, color, alpha=255)


def _draw_brackets_bgra(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: Tuple[int, int, int],
    alpha: int = 255,
) -> None:
    """Draw four L-shaped corner brackets around a node on a BGRA canvas.

    Args:
        canvas: BGRA image to draw on (mutated in-place).
        cx: Node centre x.
        cy: Node centre y.
        r: Node circle radius.
        color: BGR bracket colour (alpha applied separately).
        alpha: Alpha value 0-255 for the brackets.
    """
    p = r + BRACKET_PAD
    L = BRACKET_LEN
    t = BRACKET_THICK
    aa = cv2.LINE_AA
    c4 = (*color, alpha)  # BGRA tuple

    cv2.line(canvas, (cx-p, cy-p), (cx-p+L, cy-p  ), c4, t, aa)
    cv2.line(canvas, (cx-p, cy-p), (cx-p,   cy-p+L), c4, t, aa)
    cv2.line(canvas, (cx+p, cy-p), (cx+p-L, cy-p  ), c4, t, aa)
    cv2.line(canvas, (cx+p, cy-p), (cx+p,   cy-p+L), c4, t, aa)
    cv2.line(canvas, (cx-p, cy+p), (cx-p+L, cy+p  ), c4, t, aa)
    cv2.line(canvas, (cx-p, cy+p), (cx-p,   cy+p-L), c4, t, aa)
    cv2.line(canvas, (cx+p, cy+p), (cx+p-L, cy+p  ), c4, t, aa)
    cv2.line(canvas, (cx+p, cy+p), (cx+p,   cy+p-L), c4, t, aa)


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def render_frame(
    graph: GraphState,
    width: int = 1280,
    height: int = 720,
    selected_id: Optional[str] = None,
    show_labels: bool = True,
    ar: bool = False,
) -> np.ndarray:
    """Render the knowledge graph to a BGRA uint8 numpy array.

    Args:
        graph: The GraphState to draw.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        selected_id: ID of the currently selected/dragged node (gets a
            highlight ring), or None.
        show_labels: Whether to render text labels below nodes.
        ar: If True, background is fully transparent (alpha=0) so the frame
            can be composited over a webcam feed.  Nodes near the frame edges
            are depth-scaled slightly smaller to give a 3D depth illusion.

    Returns:
        ``(height, width, 4)`` uint8 BGRA numpy array.
    """
    # Always work in BGRA so we can set per-pixel alpha for AR mode.
    if ar:
        canvas = np.zeros((height, width, 4), dtype=np.uint8)  # transparent bg
    else:
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        canvas[:, :, :3] = BG_COLOR
        canvas[:, :,  3] = 255  # fully opaque background

    nodes = graph.nodes
    edges = graph.edges

    deg = _degree_map(graph)
    max_deg = max(deg.values()) if deg else 1

    # Physics engine runs in a 1920×1080 coordinate space.
    # Scale positions to whatever canvas size is requested so the graph
    # fills the frame correctly at 128×128, 1280×720, 1920×1080, etc.
    PHYS_W = 1920.0
    PHYS_H = 1080.0

    # Centre of frame — used for AR depth illusion
    cx_frame = width  / 2.0
    cy_frame = height / 2.0
    max_dist = (cx_frame ** 2 + cy_frame ** 2) ** 0.5

    pos: Dict[str, Tuple[int, int]] = {}
    depth_scale: Dict[str, float] = {}
    for node in nodes:
        px = int(np.clip(node.x / PHYS_W * width,  0, width  - 1))
        py = int(np.clip(node.y / PHYS_H * height, 0, height - 1))
        pos[node.id] = (px, py)
        if ar:
            # Nodes near the centre appear "closer" (larger); edge nodes appear
            # further (smaller).  Scale range: 0.65 (corner) → 1.15 (centre).
            dist = ((px - cx_frame) ** 2 + (py - cy_frame) ** 2) ** 0.5
            t = dist / max_dist          # 0 = centre, 1 = corner
            depth_scale[node.id] = 1.15 - t * 0.50
        else:
            depth_scale[node.id] = 1.0

    # Helper: draw on BGRA canvas with full alpha on drawn pixels
    def _line(p1, p2, color_bgr, alpha_val, thick=1):
        # Draw on a scratch BGR image then stamp onto BGRA canvas
        color_bgra = (*color_bgr, alpha_val)
        cv2.line(canvas, p1, p2, color_bgra, thick, cv2.LINE_AA)

    def _circle(center, radius, color_bgr, alpha_val, filled):
        color_bgra = (*color_bgr, alpha_val)
        thickness = -1 if filled else 1
        cv2.circle(canvas, center, radius, color_bgra, thickness, cv2.LINE_AA)

    def _text(txt, org, color_bgr, alpha_val, scale, thick):
        color_bgra = (*color_bgr, alpha_val)
        cv2.putText(canvas, txt, org, FONT, scale, color_bgra, thick, cv2.LINE_AA)

    # --- Edges ---
    for edge in edges:
        p1 = pos.get(edge.source_id)
        p2 = pos.get(edge.target_id)
        if p1 is None or p2 is None:
            continue
        alpha_f = max(EDGE_ALPHA_MIN, edge.confidence * 0.5)
        ec = tuple(int(c * alpha_f) for c in EDGE_COLOR)
        edge_alpha = int(180 * alpha_f) if ar else 255
        _line(p1, p2, ec, edge_alpha)

    # --- Nodes ---
    for node in nodes:
        p = pos.get(node.id)
        if p is None:
            continue
        cx, cy = p
        ds = depth_scale.get(node.id, 1.0)
        r = max(2, int(_node_radius(deg.get(node.id, 0), max_deg) * ds))
        color = _conf_color(node.confidence)
        is_sel = (node.id == selected_id) or node.selected
        node_alpha = 255

        # Filled circle
        _circle((cx, cy), r, color, node_alpha, filled=True)

        # Selection ring
        if is_sel:
            _circle((cx, cy), r + 3, SEL_RING_COLOR, node_alpha, filled=False)

        # Corner brackets
        _draw_brackets_bgra(canvas, cx, cy, r,
                             BRACKET_SEL_COLOR if is_sel else BRACKET_COLOR,
                             alpha=200 if ar else 255)

        # Label — drawn to the RIGHT of the dot, vertically centred
        if show_labels and node.label:
            raw = node.label
            label = raw[:18] + "..." if len(raw) > 18 else raw
            font_scale = max(0.6, FONT_SCALE * ds)
            (tw, th), _ = cv2.getTextSize(label, FONT, font_scale, FONT_THICK)
            lx = cx + r + 8
            ly = cy + th // 2
            # Shadow
            _text(label, (lx + 1, ly + 1), LABEL_SHADOW,
                  150 if ar else 255, font_scale, FONT_THICK + 1)
            # Text
            label_alpha = 230 if ar else 255
            _text(label, (lx, ly), LABEL_COLOR,
                  label_alpha, font_scale, FONT_THICK)

    return canvas


def render_to_rgba(
    graph: GraphState,
    width: int = 1920,
    height: int = 1080,
    ar: bool = True,
    **kwargs,
) -> np.ndarray:
    """Render the graph and return a float32 RGBA array for TouchDesigner.

    TouchDesigner Script TOPs expect RGBA float32 in [0, 1].

    Args:
        graph: The GraphState to render.
        width: Canvas width.
        height: Canvas height.
        ar: If True (default) the background is transparent so the graph can
            be composited over a webcam feed with additive/over blending.
            Set False for opaque dark-background standalone rendering.
        **kwargs: Forwarded to :func:`render_frame`.

    Returns:
        ``(height, width, 4)`` float32 numpy array with RGBA channels.
    """
    bgra = render_frame(graph, width=width, height=height, ar=ar, **kwargs)
    # render_frame returns BGRA; swap B↔R to get RGBA for TD.
    rgba_u8 = bgra[:, :, [2, 1, 0, 3]]
    # TD Script TOP expects row 0 at the BOTTOM (OpenGL convention).
    # OpenCV/NumPy have row 0 at the TOP, so flip vertically.
    rgba_u8 = np.flipud(rgba_u8)
    return (rgba_u8 / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# TouchDesigner cook callback
# ---------------------------------------------------------------------------

def cook(scriptOp) -> None:  # noqa: ANN001
    """TouchDesigner Script TOP cook callback.

    Reads ``scriptOp.storage["graph"]`` and writes the rendered RGBA image.

    Args:
        scriptOp: The TD Script TOP operator (injected by TD at runtime).
    """
    graph: Optional[GraphState] = scriptOp.storage.get("graph")
    if graph is None:
        logger.warning("graph_renderer cook: no 'graph' in scriptOp.storage.")
        return
    rgba = render_to_rgba(graph, width=scriptOp.width, height=scriptOp.height)
    scriptOp.copyNumpyArray(rgba)
