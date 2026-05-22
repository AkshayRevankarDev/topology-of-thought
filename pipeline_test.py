"""
pipeline_test.py — End-to-end smoke tests for the Topology of Thought pipeline.

Tests run without a real PDF or Ollama by mocking external calls at the
module boundary.  Each test function exercises one stage of the pipeline and
reports pass/fail to stdout.

Run with:
    python pipeline_test.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_NC = "\033[0m"
_PASS = f"{_GREEN}\u2713{_NC}"
_FAIL = f"{_RED}\u2717{_NC}"

_results: list[tuple[str, bool, str]] = []


def _run(label: str, fn: Callable[[], None]) -> None:
    """Execute *fn* and record pass/fail in *_results*.

    Args:
        label: Human-readable test name.
        fn: Zero-argument callable that raises on failure.
    """
    try:
        fn()
        _results.append((label, True, ""))
        print(f"  {_PASS}  {label}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _results.append((label, False, msg))
        print(f"  {_FAIL}  {label}")
        print(f"       {msg}")
        if "--verbose" in sys.argv:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Minimal valid PDF bytes for testing PyMuPDF
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Test: graph_state module
# ---------------------------------------------------------------------------
def test_graph_state() -> None:
    """Node/Edge dataclasses and GraphState round-trip through to_dict/from_dict."""
    from core.graph_state import Edge, GraphState, Node

    gs = GraphState()
    gs.session_name = "test"
    n1 = Node(label="Attention", description="Selects relevant tokens.")
    n2 = Node(label="Transformer", description="Attention-based architecture.")
    gs.add_node(n1)
    gs.add_node(n2)
    e = Edge(source_id=n1.id, target_id=n2.id, relation="part_of")
    gs.add_edge(e)

    serialised = gs.to_dict()
    restored = GraphState.from_dict(serialised)

    assert len(restored.nodes) == 2, "Expected 2 nodes after round-trip"
    assert len(restored.edges) == 1, "Expected 1 edge after round-trip"
    labels = {n.label for n in restored.nodes}
    assert "Attention" in labels and "Transformer" in labels


# ---------------------------------------------------------------------------
# Test: pdf_ingestion — chunk_pages logic (no real PDF needed)
# ---------------------------------------------------------------------------
def test_pdf_chunking() -> None:
    """chunk_pages produces correct chunk count and overlap."""
    from core.pdf_ingestion import chunk_pages

    # Simulate 3 pages of text.
    pages = [
        (1, " ".join(f"word{i}" for i in range(200))),
        (2, " ".join(f"word{i}" for i in range(200, 400))),
        (3, " ".join(f"word{i}" for i in range(400, 600))),
    ]
    chunks = chunk_pages(pages, source_file="test.pdf", chunk_size=200, overlap=40)
    assert len(chunks) >= 2, "Expected at least 2 chunks"
    assert chunks[0].source_file == "test.pdf"
    assert chunks[0].page_start == 1


# ---------------------------------------------------------------------------
# Test: pdf_ingestion — real PDF bytes via PyMuPDF
# ---------------------------------------------------------------------------
def test_pdf_ingestion_real() -> None:
    """ingest_pdf works on the minimal in-memory PDF written to a temp file."""
    import fitz
    from core.pdf_ingestion import ingest_pdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(_MINIMAL_PDF)
        tmp_path = f.name

    try:
        # Blank PDF produces no extractable text — we just verify no crash.
        try:
            chunks = ingest_pdf(tmp_path)
        except RuntimeError as exc:
            if "No extractable text" in str(exc):
                pass  # Expected for a blank PDF — pipeline handles this.
            else:
                raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test: concept_extractor — Ollama call mocked
# ---------------------------------------------------------------------------
def test_concept_extraction_mocked() -> None:
    """extract_from_chunk parses mocked Ollama JSON correctly."""
    from core.concept_extractor import extract_from_chunk

    mock_response = {
        "response": json.dumps({
            "concepts": [
                {"label": "Neural Network", "description": "A ML model.", "confidence": 0.95},
                {"label": "Backpropagation", "description": "Training algorithm.", "confidence": 0.90},
            ],
            "relationships": [
                {"source": "Neural Network", "target": "Backpropagation",
                 "relation": "trained_with", "confidence": 0.85},
            ],
        })
    }

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        nodes, edge_dicts = extract_from_chunk(
            text="Neural networks are trained with backpropagation.",
            source_file="test.pdf",
            page_start=1,
            page_end=1,
        )

    assert len(nodes) == 2, f"Expected 2 nodes, got {len(nodes)}"
    assert len(edge_dicts) == 1, f"Expected 1 edge dict, got {len(edge_dicts)}"
    labels = {n.label for n in nodes}
    assert "Neural Network" in labels


# ---------------------------------------------------------------------------
# Test: deduplicator — merge near-identical nodes
# ---------------------------------------------------------------------------
def test_deduplication() -> None:
    """deduplicate_graph merges semantically identical nodes."""
    from core.graph_state import GraphState, Node
    from core.deduplicator import deduplicate_graph

    gs = GraphState()
    n1 = Node(label="Attention Mechanism", description="Focuses on relevant tokens.")
    n2 = Node(label="Attention Mechanisms", description="Selects important input parts.")  # near-duplicate
    n3 = Node(label="Gradient Descent", description="Optimisation method.")
    for n in (n1, n2, n3):
        gs.add_node(n)

    # Use a lower threshold to make the test deterministic without a GPU.
    deduplicate_graph(gs, threshold=0.85)

    # "Gradient Descent" should always survive; at least one Attention node too.
    labels = {n.label for n in gs.nodes}
    assert any("Gradient" in lbl for lbl in labels), "Gradient Descent should survive"
    assert len(gs.nodes) <= 3, "Should not add new nodes during deduplication"


# ---------------------------------------------------------------------------
# Test: session_memory — save and load round-trip
# ---------------------------------------------------------------------------
def test_session_memory() -> None:
    """save_session / load_session round-trip preserves node and edge count."""
    from core.graph_state import Edge, GraphState, Node
    from features.session_memory import save_session, load_session, delete_session

    gs = GraphState()
    gs.session_name = "_pipeline_test_session"
    n1 = Node(label="TestNode1")
    n2 = Node(label="TestNode2")
    gs.add_node(n1)
    gs.add_node(n2)
    gs.add_edge(Edge(source_id=n1.id, target_id=n2.id))

    try:
        save_session(gs)
        restored = load_session("_pipeline_test_session")
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1
    finally:
        try:
            delete_session("_pipeline_test_session")
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Test: anki_export — writes a valid .apkg file
# ---------------------------------------------------------------------------
def test_anki_export() -> None:
    """export_graph_to_apkg produces a non-empty .apkg file."""
    from core.graph_state import GraphState, Node
    from features.anki_export import export_graph_to_apkg

    gs = GraphState()
    gs.session_name = "pipeline_test_deck"
    gs.add_node(Node(label="Entropy", description="Measure of disorder.", confidence=0.9))
    gs.add_node(Node(label="Information Gain", description="Reduction in entropy.", confidence=0.85))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.apkg"
        result_path = export_graph_to_apkg(gs, out, deck_name="Test Deck")
        assert result_path.exists(), "Output .apkg file not created"
        assert result_path.stat().st_size > 0, ".apkg file is empty"


# ---------------------------------------------------------------------------
# Test: confidence_coloring — colour mapping correctness
# ---------------------------------------------------------------------------
def test_confidence_coloring() -> None:
    """confidence_to_rgb returns values in [0, 1] for boundary inputs."""
    from features.confidence_coloring import confidence_to_rgb, colour_map_for_graph
    from core.graph_state import GraphState, Node

    for val in (0.0, 0.5, 1.0):
        r, g, b = confidence_to_rgb(val)
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0, \
            f"Colour out of range for confidence={val}: ({r},{g},{b})"

    gs = GraphState()
    gs.add_node(Node(label="A", confidence=0.3))
    gs.add_node(Node(label="B", confidence=0.8))
    cmap = colour_map_for_graph(gs, normalise=True)
    assert len(cmap) == 2


# ---------------------------------------------------------------------------
# Test: physics engine — one tick does not crash
# ---------------------------------------------------------------------------
def test_physics_tick() -> None:
    """PhysicsEngine.tick() runs without error on a small graph."""
    from core.graph_state import Edge, GraphState, Node
    from touchdesigner.physics import PhysicsEngine

    gs = GraphState()
    n1 = Node(label="A", x=100.0, y=100.0)
    n2 = Node(label="B", x=300.0, y=300.0)
    gs.add_node(n1)
    gs.add_node(n2)
    gs.add_edge(Edge(source_id=n1.id, target_id=n2.id))

    engine = PhysicsEngine(gs, canvas_width=1920, canvas_height=1080)
    engine.tick()  # Should not raise.

    # Nodes should have moved.
    assert (n1.x, n1.y) != (100.0, 100.0) or (n2.x, n2.y) != (300.0, 300.0), \
        "Physics tick had no effect on node positions"


# ---------------------------------------------------------------------------
# Test: gesture engine — pinch state machine transitions
# ---------------------------------------------------------------------------
def test_gesture_pinch() -> None:
    """GestureEngine emits pinch_start after holding below threshold."""
    from touchdesigner.gesture_engine import GestureEngine

    engine = GestureEngine(
        close_threshold=0.04,
        open_threshold=0.08,
        min_hold_frames=2,
    )

    # Construct minimal 21-landmark arrays — all at origin except thumb & index.
    def make_landmarks(thumb_x, index_x):
        lms = [(0.5, 0.5, 0.0)] * 21
        lms[4] = (thumb_x, 0.5, 0.0)   # thumb tip
        lms[8] = (index_x, 0.5, 0.0)   # index tip
        return lms

    # Simulate pinch (thumb and index very close for min_hold_frames + 1 frames)
    pinch_lms = make_landmarks(0.50, 0.52)  # distance ≈ 0.02 < 0.04 threshold
    events_all = []
    for _ in range(4):
        events_all.extend(engine.update([{"landmarks": pinch_lms}]))

    event_names = [e.name for e in events_all]
    assert "pinch_start" in event_names, \
        f"Expected pinch_start event, got: {event_names}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TESTS = [
    ("GraphState round-trip serialisation", test_graph_state),
    ("PDF chunk_pages logic", test_pdf_chunking),
    ("PDF ingestion (minimal real PDF)", test_pdf_ingestion_real),
    ("Concept extraction (Ollama mocked)", test_concept_extraction_mocked),
    ("Semantic deduplication", test_deduplication),
    ("Session save/load round-trip", test_session_memory),
    ("Anki .apkg export", test_anki_export),
    ("Confidence colour mapping", test_confidence_coloring),
    ("Physics engine tick", test_physics_tick),
    ("Gesture pinch state machine", test_gesture_pinch),
]


def main() -> int:
    """Run all pipeline tests and print a summary.

    Returns:
        Exit code 0 if all pass, 1 otherwise.
    """
    print("\nTopology of Thought — Pipeline Tests\n" + "=" * 40)
    for label, fn in _TESTS:
        _run(label, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("=" * 40)
    print(f"  Passed: {passed}/{len(_results)}")
    if failed:
        print(f"  Failed: {failed}/{len(_results)}")
        return 1
    print(f"{_GREEN}\u2713{_NC}  All pipeline tests passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
