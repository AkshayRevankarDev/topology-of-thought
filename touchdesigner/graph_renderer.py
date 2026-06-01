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
# IRON MAN HUD  —  Neon electric-blue palette (all BGR uint8)
# ---------------------------------------------------------------------------
BG_COLOR           = (5,   3,   8)         # near-black (opaque mode only)

# Node neon layers  (BGR)
NEON_FAR_GLOW      = ( 80,  30,   0)       # outermost dim blue haze
NEON_MID_GLOW      = (180,  80,   5)       # mid glow ring
NEON_RING          = (255, 160,  20)       # main bright ring  (sky-blue)
NEON_HIGHLIGHT     = (255, 230,  80)       # inner highlight arc (cyan)
NEON_CORE          = (255, 255, 200)       # hot white-blue core

# Confidence gradient: low → high
NODE_COLOR_LO      = ( 60,  20,   0)       # dim blue — low confidence
NODE_COLOR_HI      = (255, 180,  30)       # electric cyan — high confidence

# Edges
EDGE_NEON          = (160,  70,   5)       # dim cyan edge line
EDGE_GLOW          = ( 60,  20,   0)       # edge glow halo

# Labels
LABEL_COLOR        = (255, 210,  70)       # bright cyan-white text
LABEL_GLOW         = ( 80,  30,   0)       # label outer glow

# Brackets / HUD corners
BRACKET_COLOR      = (140,  60,   5)       # idle bracket (dim blue)
BRACKET_SEL_COLOR  = (255, 240, 100)       # selected (hot white-cyan)
SEL_RING_COLOR     = (255, 255, 180)       # selection pulse ring

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
NODE_R_MIN    = 6
NODE_R_MAX    = 20
BRACKET_LEN   = 10
BRACKET_PAD   = 6
BRACKET_THICK = 1

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT       = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.95
FONT_THICK = 1

# ---------------------------------------------------------------------------
# Edge rendering
# ---------------------------------------------------------------------------
EDGE_ALPHA_MIN = 0.30


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
    # Always BGRA canvas
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    if not ar:
        canvas[:, :, :3] = BG_COLOR
        canvas[:, :,  3] = 255

    nodes = graph.nodes
    edges = graph.edges
    deg   = _degree_map(graph)
    max_deg = max(deg.values()) if deg else 1

    PHYS_W, PHYS_H = 1920.0, 1080.0
    cx_frame  = width  / 2.0
    cy_frame  = height / 2.0
    max_dist  = (cx_frame ** 2 + cy_frame ** 2) ** 0.5

    # Build pixel positions + depth scale (centre = closer/brighter)
    pos: Dict[str, Tuple[int, int]] = {}
    depth_s: Dict[str, float] = {}
    for node in nodes:
        px = int(np.clip(node.x / PHYS_W * width,  0, width  - 1))
        py = int(np.clip(node.y / PHYS_H * height, 0, height - 1))
        pos[node.id] = (px, py)
        dist = ((px - cx_frame) ** 2 + (py - cy_frame) ** 2) ** 0.5
        t = dist / max(max_dist, 1.0)
        # centre nodes 1.2×, corner nodes 0.6×
        depth_s[node.id] = 1.20 - t * 0.60

    # -----------------------------------------------------------------------
    # Neon confidence colour  (BGR, blended by confidence level)
    # -----------------------------------------------------------------------
    def _neon_color(confidence: float) -> Tuple[int, int, int]:
        return _lerp_bgr(NODE_COLOR_LO, NODE_COLOR_HI, confidence)

    # -----------------------------------------------------------------------
    # Draw neon-glowing circle (the key AR visual element)
    # -----------------------------------------------------------------------
    def _neon_circle(cx: int, cy: int, r: int, base: Tuple[int,int,int],
                     ds: float, selected: bool) -> None:
        """Multi-layer neon glow ring. Outer haze → bright ring → hot core."""
        # Outer haze layers (big, dim, semi-transparent)
        haze_layers = [
            (r + int(18*ds), NEON_FAR_GLOW,  25),
            (r + int(12*ds), NEON_FAR_GLOW,  45),
            (r + int( 7*ds), NEON_MID_GLOW,  80),
            (r + int( 4*ds), NEON_MID_GLOW, 120),
        ]
        for hr, hc, ha in haze_layers:
            if hr > 0:
                cv2.circle(canvas, (cx, cy), hr,
                           (*hc, ha), 2, cv2.LINE_AA)

        # Semi-transparent dark fill (gives depth to the circle interior)
        fill_a = int(55 * ds)
        fill_c = (base[0]//6, base[1]//6, base[2]//8)
        cv2.circle(canvas, (cx, cy), r, (*fill_c, fill_a), -1, cv2.LINE_AA)

        # Main neon ring (1-2 px bright)
        ring_a = int(220 * ds)
        cv2.circle(canvas, (cx, cy), r, (*NEON_RING, ring_a), 2, cv2.LINE_AA)

        # Inner highlight arc (makes it look rounded / 3-D)
        if r > 5:
            hi_r = max(1, r - 2)
            hi_a = int(100 * ds)
            cv2.circle(canvas, (cx, cy), hi_r,
                       (*NEON_HIGHLIGHT, hi_a), 1, cv2.LINE_AA)

        # Hot white core dot
        core_r = max(2, r // 4)
        core_a = int(200 * ds)
        cv2.circle(canvas, (cx, cy), core_r,
                   (*NEON_CORE, core_a), -1, cv2.LINE_AA)

        # Selection: expanding pulse rings
        if selected:
            for i, pr in enumerate(range(r + 6, r + 24, 5)):
                pa = max(0, int(200 - i * 55))
                cv2.circle(canvas, (cx, cy), pr,
                           (*SEL_RING_COLOR, pa), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # EDGES — thin neon lines with soft outer glow
    # -----------------------------------------------------------------------
    for edge in edges:
        p1 = pos.get(edge.source_id)
        p2 = pos.get(edge.target_id)
        if p1 is None or p2 is None:
            continue
        af = max(EDGE_ALPHA_MIN, edge.confidence * 0.5)
        # Glow pass (thick, dim)
        ga = int(40 * af)
        cv2.line(canvas, p1, p2, (*EDGE_GLOW, ga), 3, cv2.LINE_AA)
        # Core line (thin, bright)
        la = int(160 * af)
        cv2.line(canvas, p1, p2, (*EDGE_NEON, la), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # NODES — neon glowing circles
    # -----------------------------------------------------------------------
    for node in nodes:
        p = pos.get(node.id)
        if p is None:
            continue
        cx, cy = p
        ds  = depth_s.get(node.id, 1.0)
        r   = max(4, int(_node_radius(deg.get(node.id, 0), max_deg) * ds))
        col = _neon_color(node.confidence)
        is_sel = (node.id == selected_id) or node.selected

        _neon_circle(cx, cy, r, col, ds, is_sel)

        # HUD corner brackets
        _draw_brackets_bgra(
            canvas, cx, cy, r,
            BRACKET_SEL_COLOR if is_sel else BRACKET_COLOR,
            alpha=int((200 if is_sel else 140) * ds),
        )

        # Label — right of node, depth-scaled font
        if show_labels and node.label:
            raw   = node.label
            label = (raw[:16] + "...") if len(raw) > 16 else raw
            fs    = max(0.55, FONT_SCALE * ds)
            (tw, th), _ = cv2.getTextSize(label, FONT, fs, FONT_THICK)
            lx = cx + r + 7
            ly = cy + th // 2
            # Glow pass
            la_glow = int(80 * ds)
            cv2.putText(canvas, label, (lx + 1, ly + 1),
                        FONT, fs, (*LABEL_GLOW, la_glow),
                        FONT_THICK + 2, cv2.LINE_AA)
            # Bright text
            la_txt = int(230 * ds)
            cv2.putText(canvas, label, (lx, ly),
                        FONT, fs, (*LABEL_COLOR, la_txt),
                        FONT_THICK, cv2.LINE_AA)

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
