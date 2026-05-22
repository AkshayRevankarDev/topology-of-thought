"""
session_memory.py — Persistent graph storage using JSON snapshots.

Sessions are saved to ``data/sessions/<name>.json``.  Each save is atomic:
we write to a temporary file then rename it so a crash during save never
leaves a corrupt state file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from core.graph_state import GraphState

logger = logging.getLogger(__name__)

_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def _sessions_dir() -> Path:
    """Return the sessions directory, creating it if absent.

    Returns:
        Resolved path to the sessions directory.
    """
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


def session_path(name: str) -> Path:
    """Resolve the file path for a given session name.

    Args:
        name: Session identifier (alphanumeric + underscores/hyphens).

    Returns:
        Absolute Path to the ``.json`` file.
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return _sessions_dir() / f"{safe_name}.json"


def save_session(
    graph: GraphState,
    name: Optional[str] = None,
    indent: int = 2,
) -> Path:
    """Serialise *graph* to a JSON file in the sessions directory.

    The write is atomic: we write to a temp file in the same directory, then
    rename it to the target path so a crash during write leaves the old file
    intact.

    Args:
        graph: The GraphState to persist.
        name: Session name override.  If omitted, ``graph.session_name`` is
            used.  Falls back to a timestamp string.
        indent: JSON indentation level (use 0 for compact storage).

    Returns:
        Path of the written session file.

    Raises:
        OSError: If the file cannot be written.
    """
    effective_name = name or graph.session_name or f"session_{int(time.time())}"
    if name:
        graph.session_name = effective_name

    target = session_path(effective_name)
    data = graph.to_dict()

    try:
        dir_ = target.parent
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=dir_,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tf:
            json.dump(data, tf, indent=indent, ensure_ascii=False)
            tmp_path = tf.name
        os.replace(tmp_path, target)
    except Exception as exc:
        raise OSError(f"Failed to save session '{effective_name}': {exc}") from exc

    logger.info("Session '%s' saved to '%s'.", effective_name, target)
    return target


def load_session(name: str) -> GraphState:
    """Load a previously saved session from disk.

    Args:
        name: Session name (without the ``.json`` extension).

    Returns:
        Reconstructed :class:`~core.graph_state.GraphState`.

    Raises:
        FileNotFoundError: If the session file does not exist.
        ValueError: If the file contains invalid JSON or unexpected structure.
    """
    path = session_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Session '{name}' not found.  "
            f"Expected file: {path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt session file '{path}': {exc}") from exc
    except OSError as exc:
        raise OSError(f"Cannot read session file '{path}': {exc}") from exc

    graph = GraphState.from_dict(data)
    logger.info(
        "Session '%s' loaded: %d nodes, %d edges.",
        name,
        len(graph.nodes),
        len(graph.edges),
    )
    return graph


def list_sessions() -> List[str]:
    """Return a list of saved session names (sorted by modification time, newest first).

    Returns:
        List of session name strings (without the ``.json`` extension).
    """
    dir_ = _sessions_dir()
    files = sorted(dir_.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.stem for p in files]


def delete_session(name: str) -> None:
    """Delete a saved session file.

    Args:
        name: Session name to delete.

    Raises:
        FileNotFoundError: If the session does not exist.
    """
    path = session_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Session '{name}' not found at '{path}'.")
    path.unlink()
    logger.info("Session '%s' deleted.", name)


def autosave(
    graph: GraphState,
    interval_seconds: float = 60.0,
) -> "AutoSaver":
    """Create and start an :class:`AutoSaver` that periodically saves *graph*.

    Args:
        graph: The GraphState to auto-save.
        interval_seconds: Seconds between saves.

    Returns:
        A started :class:`AutoSaver` instance.  Call ``.stop()`` to cancel.
    """
    saver = AutoSaver(graph, interval_seconds=interval_seconds)
    saver.start()
    return saver


class AutoSaver:
    """Background thread that periodically saves a GraphState to disk.

    Args:
        graph: Target graph.
        interval_seconds: Save frequency.
    """

    def __init__(self, graph: GraphState, interval_seconds: float = 60.0) -> None:
        """Initialise the auto-saver."""
        import threading
        self.graph = graph
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="autosave"
        )

    def start(self) -> None:
        """Start the background save thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the background save thread and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        """Periodic save loop — runs in the daemon thread."""
        while not self._stop_event.wait(self.interval):
            try:
                save_session(self.graph)
            except Exception as exc:
                logger.warning("AutoSaver error: %s", exc)
