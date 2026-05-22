"""
hand_tracking.py — MediaPipe Tasks-API hand tracking for TouchDesigner.

Uses HandLandmarker in LIVE_STREAM mode (mediapipe >= 0.10.30).
The legacy mp.solutions.hands API is NOT used — it is broken on macOS
with mediapipe 0.10.30+.

The .task model bundle is auto-downloaded on first run to data/hand_landmarker.task.

TouchDesigner usage
-------------------
1. Create a Script CHOP.
2. In the ``start()`` callback:   startup()
3. In the ``cook()`` callback:    cook(scriptOp)
4. In the ``stop()`` callback:    shutdown()

Output channels (per hand h=0,1; per landmark i=0..20; axis x/y/z):
    hand{h}_lm{i}_x  — normalised 0–1 horizontal position
    hand{h}_lm{i}_y  — normalised 0–1 vertical position
    hand{h}_lm{i}_z  — relative depth (arbitrary scale)

Standalone usage
----------------
    python hand_tracking.py
Displays an annotated webcam preview. Press Q to quit.
"""

from __future__ import annotations

import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard dependencies
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError as exc:
    raise ImportError("pip install opencv-python") from exc

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe — optional; gracefully absent when running inside TouchDesigner
# whose embedded Python cannot load compiled C extensions from the venv.
# All public functions degrade silently: _latest_hands stays [] and cook()
# writes zero-value channels.  Set TD's Python 64-bit Module Path in
# Edit → Preferences → DATs to the venv site-packages to enable it.
# ---------------------------------------------------------------------------
_MP_AVAILABLE = False
mp = None                  # type: ignore[assignment]
HandLandmarker = None      # type: ignore[assignment]
HandLandmarkerOptions = None  # type: ignore[assignment]
RunningMode = None         # type: ignore[assignment]
BaseOptions = None         # type: ignore[assignment]

try:
    import mediapipe as _mp_mod
    from mediapipe.tasks.python.vision import (
        HandLandmarker as _HandLandmarker,
        HandLandmarkerOptions as _HandLandmarkerOptions,
        RunningMode as _RunningMode,
    )
    from mediapipe.tasks.python.core.base_options import BaseOptions as _BaseOptions
    mp = _mp_mod
    HandLandmarker = _HandLandmarker
    HandLandmarkerOptions = _HandLandmarkerOptions
    RunningMode = _RunningMode
    BaseOptions = _BaseOptions
    _MP_AVAILABLE = True
except ImportError:
    logger.warning(
        "hand_tracking: mediapipe not available in this Python environment. "
        "Hand tracking disabled. To enable, set TouchDesigner's Python 64-bit "
        "Module Path (Edit → Preferences → DATs) to the venv site-packages: "
        "/Users/akshaymohanrevankar/Desktop/Motion/topology_of_thought"
        "/.venv/lib/python3.11/site-packages"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
MAX_HANDS = 2
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Module-level singletons (persist across TD cook calls)
# ---------------------------------------------------------------------------
_cap: Optional[cv2.VideoCapture] = None
_detector: Optional[HandLandmarker] = None
_latest_hands: List[Dict[str, List[Tuple[float, float, float]]]] = []
_frame_ts: int = 0   # monotonically increasing ms timestamp for LIVE_STREAM


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def _ensure_model() -> str:
    """Download the hand_landmarker.task bundle if not already present.

    Returns:
        Absolute path string to the local .task file.

    Raises:
        RuntimeError: If the download fails.
    """
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        logger.info("Downloading hand_landmarker.task (~8 MB)…")
        print("hand_tracking: downloading hand_landmarker.task model (~8 MB)…")
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
            print(f"hand_tracking: model saved to {_MODEL_PATH}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download hand_landmarker.task: {exc}\n"
                f"Download manually from:\n  {_MODEL_URL}\n"
                f"and place at: {_MODEL_PATH}"
            ) from exc
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Camera initialisation
# ---------------------------------------------------------------------------

def _open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open the webcam at *index*.

    Args:
        index: OpenCV device index (0 = default camera).

    Returns:
        Open :class:`cv2.VideoCapture` instance.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {index}. "
            "Ensure a webcam is connected and camera permission is granted "
            "(System Settings → Privacy & Security → Camera)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# ---------------------------------------------------------------------------
# MediaPipe Tasks detector
# ---------------------------------------------------------------------------

def _result_callback(result, output_image, timestamp_ms: int) -> None:  # noqa: ANN001
    """Called by MediaPipe after each async inference.

    Stores results in the module-level ``_latest_hands`` list so the main
    thread can read them each frame without blocking.

    Args:
        result: HandLandmarkerResult containing detected hands.
        output_image: The processed image (unused).
        timestamp_ms: Timestamp in milliseconds (unused).
    """
    global _latest_hands
    hands: List[Dict] = []
    if result.hand_landmarks:
        for hand_lms in result.hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms]
            hands.append({"landmarks": landmarks})
    _latest_hands = hands


def _create_detector(max_hands: int = MAX_HANDS) -> HandLandmarker:
    """Create and return a HandLandmarker in LIVE_STREAM mode.

    Args:
        max_hands: Maximum simultaneous hands to detect.

    Returns:
        Configured :class:`HandLandmarker` instance.
    """
    model_path = _ensure_model()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=max_hands,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.5,
        result_callback=_result_callback,
    )
    return HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def startup(camera_index: int = 0) -> None:
    """Initialise the camera and MediaPipe HandLandmarker.

    Call once in TouchDesigner's ``start()`` callback or before the main loop.
    If mediapipe is unavailable the camera is still opened so the webcam feed
    works; hand detection channels will output zeros.

    Args:
        camera_index: Webcam device index.
    """
    global _cap, _detector
    _cap = _open_camera(camera_index)
    if _MP_AVAILABLE:
        _detector = _create_detector()
        logger.info("hand_tracking: ready (Tasks API, camera=%d).", camera_index)
    else:
        logger.warning(
            "hand_tracking: mediapipe unavailable — camera opened but "
            "hand detection disabled (camera=%d).", camera_index
        )


def shutdown() -> None:
    """Release the camera and detector. Call in TD's ``stop()`` callback."""
    global _cap, _detector
    if _cap is not None:
        _cap.release()
        _cap = None
    if _detector is not None:
        _detector.close()
        _detector = None
    logger.info("hand_tracking: resources released.")


def read_frame() -> Optional[np.ndarray]:
    """Grab one RGB frame from the webcam.

    Returns:
        H×W×3 uint8 RGB numpy array, or None on failure.
    """
    if _cap is None:
        return None
    ret, bgr = _cap.read()
    if not ret:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process_frame(rgb: np.ndarray) -> List[Dict[str, List[Tuple[float, float, float]]]]:
    """Submit *rgb* to the async HandLandmarker and return the latest results.

    Results lag by up to one frame because the Tasks LIVE_STREAM callback
    fires asynchronously.  This is imperceptible at 30 fps.
    Returns ``[]`` immediately when mediapipe is unavailable.

    Args:
        rgb: H×W×3 uint8 RGB image.

    Returns:
        List of per-hand dicts, each with ``"landmarks"`` → list of 21
        ``(x, y, z)`` normalised-coordinate tuples.
    """
    global _frame_ts
    if _detector is None or not _MP_AVAILABLE:
        return []

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _frame_ts += 33   # ~30 fps increment; must be monotonically increasing
    try:
        _detector.detect_async(mp_image, _frame_ts)
    except Exception as exc:
        logger.debug("detect_async error: %s", exc)

    return list(_latest_hands)


def cook(scriptOp) -> None:  # noqa: ANN001
    """TouchDesigner Script CHOP cook callback.

    Reads one webcam frame, runs inference, and writes CHOP channels.
    Channel names: ``hand{h}_lm{i}_x``, ``hand{h}_lm{i}_y``, ``hand{h}_lm{i}_z``
    for h ∈ {0,1} and i ∈ {0..20}.

    Args:
        scriptOp: TouchDesigner Script CHOP operator (injected by TD at runtime).
    """
    rgb = read_frame()
    hands = process_frame(rgb) if rgb is not None else []

    scriptOp.clear()
    axes = ("x", "y", "z")
    for h in range(MAX_HANDS):
        for i in range(NUM_LANDMARKS):
            for ax_idx, ax in enumerate(axes):
                ch = scriptOp.appendChan(f"hand{h}_lm{i}_{ax}")
                try:
                    ch[0] = hands[h]["landmarks"][i][ax_idx]
                except (IndexError, KeyError):
                    ch[0] = 0.0


# ---------------------------------------------------------------------------
# Standalone preview
# ---------------------------------------------------------------------------

def _draw_landmarks(bgr: np.ndarray, hands: List[Dict]) -> None:
    """Draw skeleton overlay on *bgr* for debugging.

    Args:
        bgr: BGR image to draw on (mutated in-place).
        hands: Output of :func:`process_frame`.
    """
    h_img, w_img = bgr.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for hand in hands:
        lms = hand["landmarks"]
        pts = [(int(lm[0] * w_img), int(lm[1] * h_img)) for lm in lms]
        for a, b in connections:
            cv2.line(bgr, pts[a], pts[b], (60, 200, 60), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(bgr, pt, 4, (100, 255, 100), -1, cv2.LINE_AA)


def _standalone_preview() -> None:
    """Open a live annotated webcam window. Press Q to quit."""
    startup()
    print("Hand tracking preview — press Q to quit.")
    try:
        while True:
            rgb = read_frame()
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            hands = process_frame(rgb)
            _draw_landmarks(bgr, hands)
            label = f"Hands detected: {len(hands)}"
            cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("Hand Tracking — Topology of Thought", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _standalone_preview()
