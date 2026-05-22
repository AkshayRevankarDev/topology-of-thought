"""
health_check.py — Verify all Topology of Thought dependencies are operational.

Checks:
  1. Ollama running + llama3.2 responding (3-word test prompt)
  2. Webcam detected via OpenCV
  3. MediaPipe imports without error
  4. PyMuPDF can parse a dummy PDF bytes object
  5. sentence-transformers loads all-MiniLM-L6-v2
  6. Whisper imports without error
  7. genanki imports without error

Prints a green checkmark or red X with an error message for each check.
Exits with code 0 only if all checks pass.
"""

import sys

# ANSI colour codes (fall back gracefully on terminals without colour support)
_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_NC = "\033[0m"

_PASS = f"{_GREEN}\u2713{_NC}"  # ✓
_FAIL = f"{_RED}\u2717{_NC}"   # ✗


def _print_result(label: str, ok: bool, message: str = "") -> None:
    """Print a single check result with a pass/fail indicator.

    Args:
        label: Human-readable name for this check.
        ok: True if the check passed.
        message: Optional detail shown on failure.
    """
    symbol = _PASS if ok else _FAIL
    status = "OK" if ok else f"FAILED — {message}"
    print(f"  {symbol}  {label}: {status}")


# ---------------------------------------------------------------------------
# Individual check functions — each returns (ok: bool, message: str)
# ---------------------------------------------------------------------------

def check_ollama() -> tuple[bool, str]:
    """Verify Ollama is running and llama3.2 can generate a response.

    Sends a minimal 3-word prompt and verifies a non-empty reply.

    Returns:
        ``(True, "")`` on success or ``(False, error_message)``.
    """
    try:
        import requests
    except ImportError:
        return False, "requests package not installed"

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Say hello world",
                "stream": False,
                "options": {"num_predict": 10},
            },
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama at localhost:11434 — is `ollama serve` running?"
    except requests.exceptions.Timeout:
        return False, "Ollama request timed out after 30 s"
    except Exception as exc:
        return False, str(exc)

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"

    reply = resp.json().get("response", "").strip()
    if not reply:
        return False, "llama3.2 returned an empty response"

    return True, ""


def check_webcam() -> tuple[bool, str]:
    """Detect a working webcam via OpenCV.

    On macOS, distinguishes between "no camera hardware" and "permission denied"
    by consulting ``system_profiler SPCameraDataType`` when OpenCV fails to open.

    Returns:
        ``(True, "")`` if a camera is found and opens successfully.
    """
    try:
        import cv2
    except ImportError:
        return False, "opencv-python not installed"

    # Try opening — retry once because macOS may show a permission dialog
    # asynchronously on the first attempt.
    import time
    opened = False
    try:
        for _ in range(2):
            cap = cv2.VideoCapture(0)
            opened = cap.isOpened()
            cap.release()
            if opened:
                break
            time.sleep(0.6)
    except Exception as exc:
        return False, f"OpenCV error: {exc}"

    if opened:
        return True, ""

    # Detect hardware via system_profiler — works without TCC permission.
    import platform
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType"],
                capture_output=True, text=True, timeout=8,
            ).stdout
            has_hw = "Model ID" in out or "Camera" in out and "Camera:" in out
        except Exception:
            has_hw = False

        if has_hw:
            return False, (
                "Camera detected by macOS but OpenCV access is BLOCKED. "
                "Grant camera permission to the parent app (Terminal / "
                "Claude.app / your IDE) in System Settings → Privacy & "
                "Security → Camera, then re-run."
            )

    return False, "cv2.VideoCapture(0).isOpened() returned False — no webcam found"


def check_mediapipe() -> tuple[bool, str]:
    """Verify MediaPipe can be imported and the Hand Landmarker task is accessible.

    Supports both the legacy solutions API (< 0.10.x) and the new Tasks API
    (0.10.30+).

    Returns:
        ``(True, "")`` on success.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return False, "mediapipe not installed"

    # Try new Tasks API first (mediapipe >= 0.10.30).
    try:
        from mediapipe.tasks.python.vision import HandLandmarker  # noqa: F401
        return True, ""
    except Exception:
        pass

    # Fall back to legacy solutions API.
    try:
        _ = mp.solutions.hands
        _ = mp.solutions.drawing_utils
        return True, ""
    except Exception as exc:
        return False, f"MediaPipe import error: {exc}"


def check_pymupdf() -> tuple[bool, str]:
    """Verify PyMuPDF can open a minimal in-memory PDF.

    Constructs the smallest valid PDF byte string (no external file needed).

    Returns:
        ``(True, "")`` on success.
    """
    try:
        import fitz
    except ImportError:
        return False, "pymupdf not installed (import name: fitz)"

    # Minimal 1-page blank PDF in bytes.
    _MINIMAL_PDF = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )

    try:
        doc = fitz.open(stream=_MINIMAL_PDF, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except Exception as exc:
        return False, f"PyMuPDF failed to open dummy PDF: {exc}"

    if page_count < 1:
        return False, "Dummy PDF opened but reported 0 pages"
    return True, ""


def check_sentence_transformers() -> tuple[bool, str]:
    """Load the all-MiniLM-L6-v2 model from sentence-transformers.

    Returns:
        ``(True, "")`` on success.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False, "sentence-transformers not installed"

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode(["test"], normalize_embeddings=True)
        if vec.shape[0] != 1:
            return False, "Unexpected embedding output shape"
    except Exception as exc:
        return False, f"SentenceTransformer load/encode error: {exc}"

    return True, ""


def check_whisper() -> tuple[bool, str]:
    """Verify the openai-whisper package can be imported.

    Does not load a model (avoids downloading on first run).

    Returns:
        ``(True, "")`` on success.
    """
    try:
        import whisper
        _ = whisper.available_models()
    except ImportError:
        return False, "openai-whisper not installed"
    except Exception as exc:
        return False, f"Whisper import error: {exc}"
    return True, ""


def check_genanki() -> tuple[bool, str]:
    """Verify genanki can be imported and a basic Deck can be instantiated.

    Returns:
        ``(True, "")`` on success.
    """
    try:
        import genanki
        _ = genanki.Deck(deck_id=123456789, name="health_check_deck")
    except ImportError:
        return False, "genanki not installed"
    except Exception as exc:
        return False, f"genanki error: {exc}"
    return True, ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("Ollama running + llama3.2 responding", check_ollama),
    ("Webcam detected (OpenCV)", check_webcam),
    ("MediaPipe imports", check_mediapipe),
    ("PyMuPDF dummy PDF parse", check_pymupdf),
    ("sentence-transformers (all-MiniLM-L6-v2)", check_sentence_transformers),
    ("Whisper imports", check_whisper),
    ("genanki imports", check_genanki),
]


def main() -> int:
    """Run all health checks and print a summary.

    Returns:
        Exit code: 0 if all checks pass, 1 otherwise.
    """
    print("\nTopology of Thought — Health Check\n" + "=" * 40)
    all_passed = True

    for label, fn in CHECKS:
        try:
            ok, message = fn()
        except Exception as exc:
            ok, message = False, f"Unexpected error: {exc}"

        _print_result(label, ok, message)
        if not ok:
            all_passed = False

    print("=" * 40)
    if all_passed:
        print(f"{_PASS}  All checks passed — system is ready.\n")
    else:
        print(f"{_FAIL}  One or more checks failed. See messages above.\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
