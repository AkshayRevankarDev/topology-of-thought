"""
gesture_engine.py — Pinch gesture state machine for hand-tracked interaction.

Reads MediaPipe Hand Landmark data and emits discrete :class:`GestureEvent`
objects for use by the graph interaction layer (node selection, drag, expand,
swipe-to-dismiss, etc.).

State machine per hand (independent for left and right):

    IDLE  ──(pinch_dist < close_thresh)──▶  PINCH_ENTER
                                                │
                              held ≥ min_hold_frames
                                                │
                                                ▼
                                           PINCHING  ──(dist > open_thresh)──▶  PINCH_EXIT ──▶  IDLE
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class GestureState(Enum):
    """Per-hand pinch FSM states."""
    IDLE = auto()
    PINCH_ENTER = auto()
    PINCHING = auto()
    PINCH_EXIT = auto()


@dataclass
class GestureEvent:
    """A single discrete gesture event emitted by the engine.

    Attributes:
        name: Event name — one of:
            ``"pinch_start"``, ``"pinching"``, ``"pinch_end"``,
            ``"swipe_left"``, ``"swipe_right"``, ``"swipe_up"``, ``"swipe_down"``,
            ``"open_palm"``.
        hand_index: 0 = first detected hand, 1 = second.
        position: Normalised ``(x, y)`` anchor (pinch midpoint or palm centre).
        magnitude: Scalar — pinch distance or swipe distance in normalised coords.
        timestamp: Unix time of event creation.
    """
    name: str
    hand_index: int
    position: Tuple[float, float] = (0.0, 0.0)
    magnitude: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (constant across all hand models)
# ---------------------------------------------------------------------------
_WRIST      = 0
_THUMB_TIP  = 4
_INDEX_TIP  = 8
_MIDDLE_TIP = 12
_RING_TIP   = 16
_PINKY_TIP  = 20
_INDEX_MCP  = 5
_MIDDLE_MCP = 9
_RING_MCP   = 13
_PINKY_MCP  = 17


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist2d(
    landmarks: List[Tuple[float, float, float]],
    a: int,
    b: int,
) -> float:
    """2-D Euclidean distance between two landmarks (z ignored).

    Args:
        landmarks: 21-element list of ``(x, y, z)`` normalised tuples.
        a: First landmark index.
        b: Second landmark index.

    Returns:
        Distance in normalised image coordinates.
    """
    ax, ay, _ = landmarks[a]
    bx, by, _ = landmarks[b]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _midpoint(
    landmarks: List[Tuple[float, float, float]],
    a: int,
    b: int,
) -> Tuple[float, float]:
    """Midpoint between two landmarks.

    Args:
        landmarks: 21-element list.
        a: First index.
        b: Second index.

    Returns:
        ``(x, y)`` midpoint.
    """
    ax, ay, _ = landmarks[a]
    bx, by, _ = landmarks[b]
    return ((ax + bx) / 2.0, (ay + by) / 2.0)


def _is_open_palm(landmarks: List[Tuple[float, float, float]]) -> bool:
    """Return True if all four fingers appear extended (open palm).

    Uses the heuristic that finger tips should be above their MCP base
    knuckles in normalised y-space (y increases downward in MediaPipe).

    Args:
        landmarks: 21-element landmark list.

    Returns:
        True if the hand looks open/flat.
    """
    pairs = [
        (_INDEX_TIP,  _INDEX_MCP),
        (_MIDDLE_TIP, _MIDDLE_MCP),
        (_RING_TIP,   _RING_MCP),
        (_PINKY_TIP,  _PINKY_MCP),
    ]
    return all(landmarks[tip][1] < landmarks[mcp][1] for tip, mcp in pairs)


# ---------------------------------------------------------------------------
# Per-hand tracker
# ---------------------------------------------------------------------------

class HandGestureTracker:
    """Finite state machine tracking pinch and swipe gestures for one hand.

    Args:
        hand_index: Which hand this tracker monitors (0 or 1).
        close_threshold: Thumb-index distance to enter PINCH_ENTER.
        open_threshold: Distance to return to IDLE from PINCHING.
        min_hold_frames: Consecutive frames below threshold before PINCHING.
        swipe_threshold: Normalised travel distance to trigger a swipe event.
        on_event: Optional callback invoked on every emitted event.
    """

    def __init__(
        self,
        hand_index: int = 0,
        close_threshold: float = 0.045,
        open_threshold: float = 0.085,
        min_hold_frames: int = 3,
        swipe_threshold: float = 0.14,
        on_event: Optional[Callable[[GestureEvent], None]] = None,
    ) -> None:
        """Initialise the per-hand tracker."""
        self.hand_index = hand_index
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_hold_frames = min_hold_frames
        self.swipe_threshold = swipe_threshold
        self.on_event = on_event or (lambda _: None)

        self.state = GestureState.IDLE
        self._hold_count = 0
        self._pinch_origin: Optional[Tuple[float, float]] = None

    def update(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> List[GestureEvent]:
        """Advance the FSM with one frame of landmark data.

        Args:
            landmarks: 21-element list of ``(x, y, z)`` tuples from MediaPipe.

        Returns:
            List of :class:`GestureEvent` objects emitted this frame (may be empty).
        """
        events: List[GestureEvent] = []
        dist = _dist2d(landmarks, _THUMB_TIP, _INDEX_TIP)
        mid = _midpoint(landmarks, _THUMB_TIP, _INDEX_TIP)
        palm_open = _is_open_palm(landmarks)

        # ---- FSM transitions ----
        if self.state == GestureState.IDLE:
            if dist < self.close_threshold:
                self.state = GestureState.PINCH_ENTER
                self._hold_count = 1
                self._pinch_origin = mid

        elif self.state == GestureState.PINCH_ENTER:
            if dist < self.close_threshold:
                self._hold_count += 1
                if self._hold_count >= self.min_hold_frames:
                    self.state = GestureState.PINCHING
                    ev = GestureEvent("pinch_start", self.hand_index, mid, dist)
                    events.append(ev); self.on_event(ev)
            else:
                self.state = GestureState.IDLE
                self._hold_count = 0

        elif self.state == GestureState.PINCHING:
            ev = GestureEvent("pinching", self.hand_index, mid, dist)
            events.append(ev); self.on_event(ev)

            if dist > self.open_threshold:
                self.state = GestureState.PINCH_EXIT
                ev_end = GestureEvent("pinch_end", self.hand_index, mid, dist)
                events.append(ev_end); self.on_event(ev_end)
                self._emit_swipe(mid, events)

        elif self.state == GestureState.PINCH_EXIT:
            self.state = GestureState.IDLE
            self._pinch_origin = None

        # ---- Open palm (stateless) ----
        if palm_open and self.state == GestureState.IDLE:
            ev_palm = GestureEvent("open_palm", self.hand_index, mid, 0.0)
            events.append(ev_palm); self.on_event(ev_palm)

        return events

    def _emit_swipe(
        self,
        end_pos: Tuple[float, float],
        events: List[GestureEvent],
    ) -> None:
        """Classify and emit a swipe event if travel exceeds the threshold.

        Args:
            end_pos: Final pinch midpoint position.
            events: List to append the swipe event to.
        """
        if self._pinch_origin is None:
            return
        dx = end_pos[0] - self._pinch_origin[0]
        dy = end_pos[1] - self._pinch_origin[1]
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < self.swipe_threshold:
            return
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        ev = GestureEvent(f"swipe_{direction}", self.hand_index, end_pos, mag)
        events.append(ev); self.on_event(ev)


# ---------------------------------------------------------------------------
# Multi-hand engine
# ---------------------------------------------------------------------------

class GestureEngine:
    """Manages up to two :class:`HandGestureTracker` instances.

    Feed one frame of MediaPipe hand-landmark data and receive all emitted
    :class:`GestureEvent` objects for both hands.

    Example::

        engine = GestureEngine()
        for frame_hands in mediapipe_stream:
            for ev in engine.update(frame_hands):
                route_event(ev)
    """

    def __init__(
        self,
        close_threshold: float = 0.045,
        open_threshold: float = 0.085,
        min_hold_frames: int = 3,
        swipe_threshold: float = 0.14,
        on_event: Optional[Callable[[GestureEvent], None]] = None,
    ) -> None:
        """Initialise trackers for both hands.

        Args:
            close_threshold: Pinch-close distance.
            open_threshold: Pinch-open distance.
            min_hold_frames: Frames before promoting PINCH_ENTER → PINCHING.
            swipe_threshold: Normalised travel to trigger a swipe.
            on_event: Global event callback.
        """
        self._trackers = [
            HandGestureTracker(
                hand_index=i,
                close_threshold=close_threshold,
                open_threshold=open_threshold,
                min_hold_frames=min_hold_frames,
                swipe_threshold=swipe_threshold,
                on_event=on_event,
            )
            for i in range(2)
        ]

    def update(self, hands_data: List[Dict]) -> List[GestureEvent]:
        """Process one frame of hand data and return all gesture events.

        Args:
            hands_data: List of per-hand dicts (output of
                :func:`~touchdesigner.hand_tracking.process_frame`).
                Each dict must have a ``"landmarks"`` key with 21 tuples.

        Returns:
            Flat list of :class:`GestureEvent` instances from both hands.
        """
        events: List[GestureEvent] = []
        for idx, tracker in enumerate(self._trackers):
            if idx < len(hands_data):
                events.extend(tracker.update(hands_data[idx]["landmarks"]))
        return events

    @property
    def states(self) -> List[GestureState]:
        """Return the current FSM state for each hand tracker."""
        return [t.state for t in self._trackers]
