"""
standalone_preview.py — Terminal-runnable graph visualiser (no TouchDesigner needed).

Aesthetic: white-on-dark, minimal, monospace labels, corner brackets, confidence
coloring. Matches the Instagram reference look — no glow, no bloom.

Controls:
  Mouse drag  — grab and move any node (pins it; releases on mouse-up)
  Pinch       — grab the node under the pinch midpoint (webcam)
  Swipe down  — scatter all nodes
  Open palm   — release any pinned/dragged node
  Q           — quit
  S           — screenshot → data/exports/screenshot_<timestamp>.png
  R           — re-scatter nodes randomly
  Space       — pause / resume physics
  G           — toggle gesture overlay
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is on the path when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python required: pip install opencv-python") from exc

import numpy as np

from core.graph_state import GraphState, Node
from features.session_memory import load_session
from touchdesigner.physics import PhysicsEngine

# Hand tracking + gestures are optional — preview still works without them.
try:
    from touchdesigner import hand_tracking
    from touchdesigner.gesture_engine import GestureEngine
    _GESTURES_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    hand_tracking = None  # type: ignore[assignment]
    GestureEngine = None  # type: ignore[assignment]
    _GESTURES_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canvas & aesthetic constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720
FPS_TARGET = 30
FRAME_MS = int(1000 / FPS_TARGET)

BG_COLOR = (12, 10, 10)               # #0a0a0c in BGR

# Node
NODE_R_MIN = 5
NODE_R_MAX = 16
NODE_COLOR_LO = (55, 52, 65)          # low-confidence BGR  ≈ #41 34 37
NODE_COLOR_HI = (230, 225, 235)       # high-confidence BGR ≈ #eb e1 e6

# Edges
EDGE_COLOR_BASE = (70, 68, 80)
EDGE_ALPHA_MIN = 0.35

# Labels
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.95
FONT_THICK = 1
LABEL_COLOR = (145, 140, 155)
LABEL_SHADOW = (5, 4, 6)

# Corner brackets
BRACKET_LEN = 9
BRACKET_PAD = 5
BRACKET_COLOR_IDLE = (55, 52, 68)
BRACKET_COLOR_SEL = (180, 175, 210)
BRACKET_THICK = 1

# Selected node
SEL_RING_COLOR = (200, 195, 225)

# HUD
HUD_COLOR = (80, 78, 95)
HUD_FONT_SCALE = 0.75
HUD_FONT = cv2.FONT_HERSHEY_PLAIN

# Gesture cursor
CURSOR_COLOR_IDLE = (160, 150, 200)
CURSOR_COLOR_PINCH = (90, 230, 255)     # bright cyan when pinching
CURSOR_COLOR_HOVER = (110, 200, 230)    # warm hover indicator
CURSOR_R = 18

# Picker — how close (in pixels) the cursor / mouse must be to a node to grab it
PICK_RADIUS = 90.0       # mouse picker
PICK_RADIUS_GESTURE = 140.0   # generous radius for hand pinches (jittery)

# Webcam overlay
CAM_OVERLAY_W = 240
CAM_OVERLAY_H = 180
CAM_OVERLAY_PAD = 14


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp_color(
    lo: Tuple[int, int, int],
    hi: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linearly interpolate between two BGR colours."""
    t = max(0.0, min(1.0, t))
    return (
        int(lo[0] + (hi[0] - lo[0]) * t),
        int(lo[1] + (hi[1] - lo[1]) * t),
        int(lo[2] + (hi[2] - lo[2]) * t),
    )


def _conf_color(confidence: float) -> Tuple[int, int, int]:
    """Map confidence ∈ [0,1] to a BGR node colour."""
    return _lerp_color(NODE_COLOR_LO, NODE_COLOR_HI, confidence)


def _degree_map(graph: GraphState) -> dict:
    """Return {node_id: edge_count}."""
    deg: dict = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        deg[e.source_id] = deg.get(e.source_id, 0) + 1
        deg[e.target_id] = deg.get(e.target_id, 0) + 1
    return deg


def _node_radius(degree: int, max_degree: int) -> int:
    """Scale node radius by degree."""
    t = degree / max(max_degree, 1)
    return NODE_R_MIN + int(t * (NODE_R_MAX - NODE_R_MIN))


def _draw_brackets(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw four corner bracket marks around a node circle."""
    p = r + BRACKET_PAD
    L = BRACKET_LEN
    t = BRACKET_THICK
    aa = cv2.LINE_AA

    cv2.line(canvas, (cx - p, cy - p), (cx - p + L, cy - p    ), color, t, aa)
    cv2.line(canvas, (cx - p, cy - p), (cx - p,     cy - p + L), color, t, aa)
    cv2.line(canvas, (cx + p, cy - p), (cx + p - L, cy - p    ), color, t, aa)
    cv2.line(canvas, (cx + p, cy - p), (cx + p,     cy - p + L), color, t, aa)
    cv2.line(canvas, (cx - p, cy + p), (cx - p + L, cy + p    ), color, t, aa)
    cv2.line(canvas, (cx - p, cy + p), (cx - p,     cy + p - L), color, t, aa)
    cv2.line(canvas, (cx + p, cy + p), (cx + p - L, cy + p    ), color, t, aa)
    cv2.line(canvas, (cx + p, cy + p), (cx + p,     cy + p - L), color, t, aa)


def _nearest_node(
    graph: GraphState,
    x: float,
    y: float,
    max_dist: float = PICK_RADIUS,
) -> Optional[str]:
    """Return the id of the node nearest to (x, y) within *max_dist*, or None."""
    best_d = max_dist
    best_id: Optional[str] = None
    for node in graph.nodes:
        d = math.hypot(node.x - x, node.y - y)
        if d < best_d:
            best_d = d
            best_id = node.id
    return best_id


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_frame(
    canvas: np.ndarray,
    graph: GraphState,
    deg: dict,
    max_deg: int,
    selected_id: Optional[str],
) -> None:
    """Draw the full graph onto *canvas* in-place."""
    canvas[:] = BG_COLOR

    # Edges first, under nodes
    for edge in graph.edges:
        src = graph.get_node(edge.source_id)
        tgt = graph.get_node(edge.target_id)
        if src is None or tgt is None:
            continue
        x1 = int(np.clip(src.x, 0, WIDTH - 1))
        y1 = int(np.clip(src.y, 0, HEIGHT - 1))
        x2 = int(np.clip(tgt.x, 0, WIDTH - 1))
        y2 = int(np.clip(tgt.y, 0, HEIGHT - 1))
        alpha = max(EDGE_ALPHA_MIN, edge.confidence * 0.55)
        ec = tuple(int(c * alpha) for c in EDGE_COLOR_BASE)
        cv2.line(canvas, (x1, y1), (x2, y2), ec, 1, cv2.LINE_AA)

    # Nodes
    for node in graph.nodes:
        px = int(np.clip(node.x, 0, WIDTH - 1))
        py = int(np.clip(node.y, 0, HEIGHT - 1))
        r = _node_radius(deg.get(node.id, 0), max_deg)
        color = _conf_color(node.confidence)
        is_sel = node.id == selected_id

        cv2.circle(canvas, (px, py), r, color, -1, cv2.LINE_AA)
        if is_sel:
            cv2.circle(canvas, (px, py), r + 3, SEL_RING_COLOR, 1, cv2.LINE_AA)
        _draw_brackets(
            canvas, px, py, r,
            BRACKET_COLOR_SEL if is_sel else BRACKET_COLOR_IDLE,
        )

        raw = node.label
        label = raw[:18] + "..." if len(raw) > 18 else raw
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        lx = px + r + 8
        ly = py + th // 2

        cv2.putText(
            canvas, label, (lx + 1, ly + 1),
            FONT, FONT_SCALE, LABEL_SHADOW, FONT_THICK + 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, label, (lx, ly),
            FONT, FONT_SCALE, LABEL_COLOR, FONT_THICK, cv2.LINE_AA,
        )


def _draw_cursor(
    canvas: np.ndarray,
    x: int,
    y: int,
    pinching: bool,
    pinch_dist: float = 0.0,
) -> None:
    """Draw a ring + crosshair representing the gesture cursor position.

    The ring shrinks as the pinch distance closes — gives a live preview
    of how close you are to a pinch trigger.
    """
    color = CURSOR_COLOR_PINCH if pinching else CURSOR_COLOR_IDLE
    aa = cv2.LINE_AA

    if pinching:
        # Solid filled disc + outer ring for crisp "grabbed" feedback.
        cv2.circle(canvas, (x, y), CURSOR_R + 4, color, 2, aa)
        cv2.circle(canvas, (x, y), 5, color, -1, aa)
    else:
        # Outer ring + animated inner ring that shrinks as pinch closes.
        inner = max(3, int(CURSOR_R * min(1.0, pinch_dist / 0.12)))
        cv2.circle(canvas, (x, y), CURSOR_R, color, 1, aa)
        cv2.circle(canvas, (x, y), inner, color, 1, aa)

    cv2.line(canvas, (x - 5, y), (x + 5, y), color, 1, aa)
    cv2.line(canvas, (x, y - 5), (x, y + 5), color, 1, aa)


def _draw_hover(
    canvas: np.ndarray,
    graph: GraphState,
    hover_id: Optional[str],
    deg: dict,
    max_deg: int,
) -> None:
    """Draw a soft ring around the node the cursor is currently hovering over."""
    if not hover_id:
        return
    node = graph.get_node(hover_id)
    if node is None:
        return
    r = _node_radius(deg.get(node.id, 0), max_deg) + 8
    cv2.circle(
        canvas,
        (int(node.x), int(node.y)),
        r,
        CURSOR_COLOR_HOVER,
        1,
        cv2.LINE_AA,
    )


def _draw_cam_overlay(
    canvas: np.ndarray,
    cam_bgr: Optional[np.ndarray],
    hands: list,
) -> None:
    """Composite a small webcam preview with hand-landmark skeleton into the top-right corner."""
    if cam_bgr is None:
        return
    h, w = canvas.shape[:2]
    thumb = cv2.resize(cam_bgr, (CAM_OVERLAY_W, CAM_OVERLAY_H), interpolation=cv2.INTER_AREA)

    # Draw skeleton in the thumbnail's coordinate space
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for hand in hands:
        lms = hand.get("landmarks", [])
        if len(lms) < 21:
            continue
        pts = [
            (int((1.0 - lm[0]) * CAM_OVERLAY_W), int(lm[1] * CAM_OVERLAY_H))
            for lm in lms
        ]
        for a, b in connections:
            cv2.line(thumb, pts[a], pts[b], (90, 220, 110), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(thumb, pt, 2, (140, 240, 140), -1, cv2.LINE_AA)

    # mirror horizontally for natural feedback
    thumb = cv2.flip(thumb, 1)

    x0 = w - CAM_OVERLAY_W - CAM_OVERLAY_PAD
    y0 = CAM_OVERLAY_PAD
    canvas[y0:y0 + CAM_OVERLAY_H, x0:x0 + CAM_OVERLAY_W] = thumb
    cv2.rectangle(
        canvas, (x0 - 1, y0 - 1),
        (x0 + CAM_OVERLAY_W, y0 + CAM_OVERLAY_H),
        (75, 73, 88), 1, cv2.LINE_AA,
    )


def _draw_hud(
    canvas: np.ndarray,
    graph: GraphState,
    fps: float,
    paused: bool,
    gesture_ctrl: Optional["GestureController"] = None,
) -> None:
    """Overlay a minimal HUD with live gesture telemetry in the top-left corner."""
    lines = [
        f"nodes:{len(graph.nodes)}  edges:{len(graph.edges)}",
        f"fps:{fps:.0f}  {'[PAUSED]' if paused else ''}",
    ]
    if gesture_ctrl is not None:
        # Live diagnostic line — tells the user instantly why a pinch isn't firing.
        pd = gesture_ctrl.last_pinch_dist
        bar_filled = int(min(1.0, pd / 0.15) * 20)
        bar = "[" + "#" * bar_filled + "." * (20 - bar_filled) + "]"
        threshold = gesture_ctrl.engine._trackers[0].close_threshold if gesture_ctrl.engine else 0
        lines.append(
            f"hands:{gesture_ctrl.hand_count}  pinch-dist:{pd:.3f} {bar}  "
            f"thresh:{threshold:.3f}"
        )
        lines.append(
            f"last:{gesture_ctrl.last_event_name}  "
            f"holding:{gesture_ctrl.dragging_id[:6] if gesture_ctrl.dragging_id else '—'}  "
            f"hover:{gesture_ctrl.hover_id[:6] if gesture_ctrl.hover_id else '—'}"
        )
    else:
        lines.append("gestures: off (mouse only)")
    lines.append("pinch:grab  swipe-down:scatter  open-palm:release")
    lines.append("Q:quit  S:save  R:scatter  SPC:pause  G:toggle-cam")
    y = 18
    for line in lines:
        cv2.putText(
            canvas, line, (10, y),
            HUD_FONT, HUD_FONT_SCALE, HUD_COLOR, 1, cv2.LINE_AA,
        )
        y += 16


# ---------------------------------------------------------------------------
# Mouse interaction
# ---------------------------------------------------------------------------

class MouseHandler:
    """Tracks click-drag state for node interaction."""

    def __init__(self, graph: GraphState) -> None:
        self.graph = graph
        self.dragging_id: Optional[str] = None
        self._offset_x = 0.0
        self._offset_y = 0.0

    def callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            best_id = _nearest_node(self.graph, x, y, max_dist=PICK_RADIUS)
            if best_id:
                self.dragging_id = best_id
                node = self.graph.get_node(best_id)
                node.pinned = True
                node.selected = True
                self._offset_x = node.x - x
                self._offset_y = node.y - y

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_id:
            node = self.graph.get_node(self.dragging_id)
            if node:
                node.x = float(x) + self._offset_x
                node.y = float(y) + self._offset_y
                node.vx = 0.0
                node.vy = 0.0

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_id:
                node = self.graph.get_node(self.dragging_id)
                if node:
                    node.pinned = False
                    node.selected = False
            self.dragging_id = None


# ---------------------------------------------------------------------------
# Gesture-to-graph adapter
# ---------------------------------------------------------------------------

class GestureController:
    """Translate MediaPipe hand landmarks into graph drag/scatter/release actions.

    The pinch midpoint of hand 0 acts as a 2-D cursor in canvas coordinates.
    A pinch_start grabs the nearest node; pinching frames drag it; pinch_end
    releases.  Swipes and open palm trigger one-shot actions.

    Args:
        graph: The live GraphState.
        canvas_width: Render canvas width in pixels.
        canvas_height: Render canvas height in pixels.
        engine: Optional preconstructed GestureEngine (mostly for tests).
    """

    def __init__(
        self,
        graph: GraphState,
        canvas_width: int = WIDTH,
        canvas_height: int = HEIGHT,
        engine: Optional["GestureEngine"] = None,  # type: ignore[name-defined]
    ) -> None:
        self.graph = graph
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        # Forgiving defaults: wider close/open thresholds and shorter hold.
        # Real-world pinch landmark distances vary 0.02-0.10 depending on
        # hand size, distance from camera and lighting.
        self.engine = engine or (
            GestureEngine(
                close_threshold=0.07,
                open_threshold=0.12,
                min_hold_frames=2,
                swipe_threshold=0.18,
            ) if _GESTURES_AVAILABLE else None
        )
        self.dragging_id: Optional[str] = None
        self.cursor_xy: Optional[Tuple[int, int]] = None
        self._smoothed_xy: Optional[Tuple[float, float]] = None
        self.pinching = False
        self.last_event_name = "—"
        self.last_event_t = 0.0
        self.last_pinch_dist = 0.0
        self.hand_count = 0
        self.hover_id: Optional[str] = None
        self.enabled = self.engine is not None

    def _to_canvas(self, nx: float, ny: float) -> Tuple[int, int]:
        """Map normalised (0..1) MediaPipe coords (selfie-mirrored) to canvas px."""
        # MediaPipe x increases left-to-right of the input image; webcam is not
        # mirrored on our side, so flip x for natural "follow my hand" feel.
        cx = int((1.0 - nx) * self.canvas_width)
        cy = int(ny * self.canvas_height)
        cx = max(0, min(self.canvas_width - 1, cx))
        cy = max(0, min(self.canvas_height - 1, cy))
        return cx, cy

    def update(self, hands: list) -> None:
        """Consume one frame of hand-landmark data and apply effects to the graph."""
        if not self.enabled or self.engine is None:
            return

        self.hand_count = len(hands)

        # Update cursor from the first detected hand's pinch midpoint
        if hands:
            lms = hands[0].get("landmarks", [])
            if len(lms) >= 9:
                thumb = lms[4]
                index = lms[8]
                mx = (thumb[0] + index[0]) / 2.0
                my = (thumb[1] + index[1]) / 2.0
                raw_cx, raw_cy = self._to_canvas(mx, my)

                # Track raw pinch distance for the debug HUD.
                self.last_pinch_dist = (
                    (thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2
                ) ** 0.5

                # Exponential-moving-average smoothing to kill jitter.
                if self._smoothed_xy is None:
                    self._smoothed_xy = (float(raw_cx), float(raw_cy))
                else:
                    sx, sy = self._smoothed_xy
                    a = 0.45
                    self._smoothed_xy = (
                        sx * (1 - a) + raw_cx * a,
                        sy * (1 - a) + raw_cy * a,
                    )
                self.cursor_xy = (int(self._smoothed_xy[0]), int(self._smoothed_xy[1]))
        else:
            self.cursor_xy = None
            self._smoothed_xy = None
            self.last_pinch_dist = 0.0

        # Hover indicator — which node would I grab if I pinched right now?
        if self.cursor_xy is not None and not self.pinching:
            self.hover_id = _nearest_node(
                self.graph, self.cursor_xy[0], self.cursor_xy[1],
                max_dist=PICK_RADIUS_GESTURE,
            )
        elif not self.pinching:
            self.hover_id = None

        events = self.engine.update(hands)
        for ev in events:
            self.last_event_name = ev.name
            self.last_event_t = time.time()

            if ev.name == "pinch_start" and self.cursor_xy is not None:
                cx, cy = self.cursor_xy
                # Prefer the hover target if any, else nearest within wide radius.
                node_id = self.hover_id or _nearest_node(
                    self.graph, cx, cy, max_dist=PICK_RADIUS_GESTURE,
                )
                if node_id:
                    self.dragging_id = node_id
                    node = self.graph.get_node(node_id)
                    if node:
                        node.pinned = True
                        node.selected = True
                self.pinching = True

            elif ev.name == "pinching":
                self.pinching = True
                if self.dragging_id and self.cursor_xy is not None:
                    node = self.graph.get_node(self.dragging_id)
                    if node:
                        cx, cy = self.cursor_xy
                        node.x = float(cx)
                        node.y = float(cy)
                        node.vx = 0.0
                        node.vy = 0.0

            elif ev.name == "pinch_end":
                if self.dragging_id:
                    node = self.graph.get_node(self.dragging_id)
                    if node:
                        node.pinned = False
                        node.selected = False
                self.dragging_id = None
                self.pinching = False

            elif ev.name == "swipe_down":
                _scatter(self.graph)

            elif ev.name == "open_palm":
                for node in self.graph.nodes:
                    node.pinned = False
                    node.selected = False
                self.dragging_id = None
                self.pinching = False


# ---------------------------------------------------------------------------
# Webcam setup with graceful failure
# ---------------------------------------------------------------------------

def _start_hand_tracking() -> Tuple[bool, str]:
    """Attempt to start the webcam + MediaPipe Tasks HandLandmarker.

    Returns:
        ``(ok, message)`` — ``ok`` False means the user-facing preview should
        still run but without gesture input.
    """
    if not _GESTURES_AVAILABLE or hand_tracking is None:
        return False, "mediapipe/hand_tracking not importable"
    try:
        hand_tracking.startup(camera_index=0)
    except Exception as exc:
        return False, str(exc)
    return True, ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _scatter(graph: GraphState, margin: int = 80) -> None:
    """Randomise all node positions within the canvas bounds."""
    rng = random.Random()
    for node in graph.nodes:
        node.x = float(rng.randint(margin, WIDTH - margin))
        node.y = float(rng.randint(margin, HEIGHT - margin))
        node.vx = 0.0
        node.vy = 0.0


def run(
    session_name: str = "attention_is_all_you_need",
    enable_gestures: bool = True,
) -> None:
    """Load a session and open the interactive preview window."""
    print(f"Loading session '{session_name}'...")
    graph = load_session(session_name)
    print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    _scatter(graph)

    engine = PhysicsEngine(
        graph,
        canvas_width=float(WIDTH),
        canvas_height=float(HEIGHT),
        repulsion=4000.0,
        spring_k=0.04,
        spring_l=160.0,
        damping=0.82,
        gravity=0.008,
        ticks_per_second=60.0,
    )
    engine.start()

    # --- Gesture / webcam setup ------------------------------------------------
    gesture_ok = False
    gesture_err = ""
    if enable_gestures and _GESTURES_AVAILABLE:
        gesture_ok, gesture_err = _start_hand_tracking()
        if gesture_ok:
            print("Hand gesture control: enabled (MediaPipe Tasks HandLandmarker)")
        else:
            print(f"Hand gesture control: disabled — {gesture_err}")
            print("  (Grant camera permission in System Settings → Privacy & "
                  "Security → Camera, then restart.)")

    gesture_ctrl: Optional[GestureController] = (
        GestureController(graph, canvas_width=WIDTH, canvas_height=HEIGHT)
        if gesture_ok else None
    )

    show_cam = True

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    mouse = MouseHandler(graph)

    cv2.namedWindow("Topology of Thought", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Topology of Thought", WIDTH, HEIGHT)
    cv2.setMouseCallback("Topology of Thought", mouse.callback)

    exports_dir = _ROOT / "data" / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    paused = False
    fps_display = 30.0

    print("Window open. Drag with mouse OR pinch with hand. "
          "Q=quit, S=screenshot, R=scatter, Space=pause, G=toggle-cam.")

    while True:
        t0 = time.perf_counter()

        cam_bgr_overlay: Optional[np.ndarray] = None
        hands_for_overlay: list = []

        if gesture_ctrl is not None and gesture_ok:
            rgb = hand_tracking.read_frame()
            if rgb is not None:
                hands = hand_tracking.process_frame(rgb)
                gesture_ctrl.update(hands)
                hands_for_overlay = hands
                if show_cam:
                    cam_bgr_overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        deg = _degree_map(graph)
        max_deg = max(deg.values()) if deg else 1

        selected_id = (
            (gesture_ctrl.dragging_id if gesture_ctrl else None)
            or mouse.dragging_id
        )
        render_frame(canvas, graph, deg, max_deg, selected_id)

        if cam_bgr_overlay is not None:
            _draw_cam_overlay(canvas, cam_bgr_overlay, hands_for_overlay)

        if gesture_ctrl is not None:
            _draw_hover(canvas, graph, gesture_ctrl.hover_id, deg, max_deg)
            if gesture_ctrl.cursor_xy is not None:
                cx, cy = gesture_ctrl.cursor_xy
                _draw_cursor(
                    canvas, cx, cy,
                    gesture_ctrl.pinching,
                    gesture_ctrl.last_pinch_dist,
                )

        _draw_hud(canvas, graph, fps_display, paused, gesture_ctrl)

        cv2.imshow("Topology of Thought", canvas)
        key = cv2.waitKey(FRAME_MS) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = int(time.time())
            out_path = exports_dir / f"screenshot_{ts}.png"
            cv2.imwrite(str(out_path), canvas)
            print(f"Screenshot saved: {out_path}")
        elif key == ord("r"):
            _scatter(graph)
        elif key == ord(" "):
            paused = not paused
            if paused:
                engine.stop()
                print("Physics paused.")
            else:
                engine.start()
                print("Physics resumed.")
        elif key == ord("g"):
            show_cam = not show_cam

        elapsed = time.perf_counter() - t0
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(elapsed, 1e-6))

    engine.stop()
    if gesture_ok and hand_tracking is not None:
        hand_tracking.shutdown()
    cv2.destroyAllWindows()
    print("Preview closed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topology of Thought — standalone preview")
    parser.add_argument(
        "--session",
        default="attention_is_all_you_need",
        help="Session name to load (default: attention_is_all_you_need)",
    )
    parser.add_argument(
        "--no-gestures",
        action="store_true",
        help="Disable webcam + hand gesture control (mouse only).",
    )
    args = parser.parse_args()
    run(session_name=args.session, enable_gestures=not args.no_gestures)
