"""
anki_export.py — Export knowledge-graph nodes as Anki flashcards.

Each graph node becomes a Basic card:
  - Front: concept label
  - Back: description + source paper + page references

The output is an .apkg file importable directly into Anki.
Requires the ``genanki`` library (no Anki desktop installation needed at
export time).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional

from core.graph_state import GraphState, Node

logger = logging.getLogger(__name__)

try:
    import genanki
except ImportError as exc:
    raise ImportError(
        "genanki is required.  Install with: pip install genanki"
    ) from exc


# ---------------------------------------------------------------------------
# Deck and model definitions
# ---------------------------------------------------------------------------

def _stable_id(seed: str) -> int:
    """Generate a stable integer ID from a string seed.

    Anki uses 32-bit integer IDs.  We derive them deterministically so that
    re-exporting the same content produces the same deck/model IDs and Anki
    does not create duplicate decks.

    Args:
        seed: Arbitrary string to hash.

    Returns:
        Positive integer in the range Anki expects.
    """
    digest = hashlib.md5(seed.encode()).hexdigest()
    return int(digest[:8], 16)


_CARD_MODEL = genanki.Model(
    model_id=_stable_id("topology_of_thought_basic_v1"),
    name="Topology of Thought — Basic",
    fields=[
        {"name": "Concept"},
        {"name": "Explanation"},
        {"name": "Source"},
        {"name": "Pages"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "<h2 style='font-family:sans-serif;'>{{Concept}}</h2>",
            "afmt": (
                "{{FrontSide}}"
                "<hr id='answer'>"
                "<p style='font-family:sans-serif;'>{{Explanation}}</p>"
                "<p style='font-size:0.8em;color:#888;'>Source: {{Source}}"
                " &nbsp;|&nbsp; Pages: {{Pages}}</p>"
            ),
        }
    ],
    css=(
        ".card { font-family: sans-serif; font-size: 18px; "
        "text-align: center; background: #1a1a2e; color: #e0e0e0; }"
        " h2 { color: #64a0f0; }"
    ),
)


def node_to_note(node: Node) -> genanki.Note:
    """Convert a single Node to a genanki Note.

    Args:
        node: The concept node to convert.

    Returns:
        A :class:`genanki.Note` ready to be added to a deck.
    """
    page_str = (
        ", ".join(str(p) for p in sorted(node.page_refs))
        if node.page_refs
        else "N/A"
    )
    return genanki.Note(
        model=_CARD_MODEL,
        fields=[
            node.label,
            node.description or "(no description)",
            node.source_paper or "unknown",
            page_str,
        ],
        guid=genanki.guid_for(node.id),
    )


def export_graph_to_apkg(
    graph: GraphState,
    output_path: str | Path,
    deck_name: Optional[str] = None,
    min_confidence: float = 0.0,
) -> Path:
    """Export all (or filtered) nodes from *graph* to an Anki .apkg file.

    Args:
        graph: The knowledge graph to export.
        output_path: Filesystem path for the .apkg output file.
        deck_name: Human-readable deck name shown in Anki.  Defaults to the
            graph's session name.
        min_confidence: Only nodes with ``confidence >= min_confidence`` are
            exported.  Pass 0.0 to export everything.

    Returns:
        The resolved path of the written .apkg file.

    Raises:
        ValueError: If no nodes pass the confidence filter.
        OSError: If the output directory cannot be created.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not output_path.suffix:
        output_path = output_path.with_suffix(".apkg")

    name = deck_name or graph.session_name or "Topology of Thought"
    deck = genanki.Deck(deck_id=_stable_id(name), name=name)

    eligible_nodes = [
        n for n in graph.nodes if n.confidence >= min_confidence
    ]
    if not eligible_nodes:
        raise ValueError(
            f"No nodes with confidence >= {min_confidence} to export. "
            f"Graph has {len(graph.nodes)} total nodes."
        )

    for node in eligible_nodes:
        deck.add_note(node_to_note(node))

    package = genanki.Package(deck)
    try:
        package.write_to_file(str(output_path))
    except Exception as exc:
        raise OSError(f"Failed to write Anki package to '{output_path}': {exc}") from exc

    logger.info(
        "Exported %d cards to '%s'.",
        len(eligible_nodes),
        output_path,
    )
    return output_path


def export_selected_nodes(
    graph: GraphState,
    output_path: str | Path,
    deck_name: Optional[str] = None,
) -> Path:
    """Export only the currently *selected* nodes in the graph.

    Useful for gesture-driven "export what I'm looking at" workflows.

    Args:
        graph: GraphState containing selection state.
        output_path: Output .apkg path.
        deck_name: Optional deck name override.

    Returns:
        Path of the written file.

    Raises:
        ValueError: If no nodes are selected.
    """
    selected = [n for n in graph.nodes if n.selected]
    if not selected:
        raise ValueError("No nodes are currently selected for export.")

    # Build a temporary single-deck graph with only selected nodes.
    from core.graph_state import GraphState as GS
    temp_graph = GS()
    temp_graph.session_name = deck_name or (graph.session_name + " — Selection")
    for node in selected:
        temp_graph.add_node(node)

    return export_graph_to_apkg(temp_graph, output_path, deck_name=temp_graph.session_name)
