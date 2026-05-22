"""
concept_extractor.py — LLM-driven concept and relationship extraction via Ollama.

Sends text chunks to a locally-running Ollama instance (default: llama3.2 on
port 11434) and parses structured concept/edge data from the JSON response.
All network calls are local; no external APIs are used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.graph_state import Edge, GraphState, Node

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a knowledge-graph extraction assistant. Given an academic text excerpt,
extract the key concepts and the relationships between them.

Return ONLY valid JSON with this exact structure (no markdown fences, no extra text):
{{
  "concepts": [
    {{
      "label": "<short concept name, 1-4 words>",
      "description": "<one sentence explanation>",
      "confidence": <float 0.0-1.0>
    }}
  ],
  "relationships": [
    {{
      "source": "<label of source concept>",
      "target": "<label of target concept>",
      "relation": "<verb phrase, e.g. causes, extends, contradicts>",
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

Text excerpt (source: {source_file}, pages {page_start}-{page_end}):
\"\"\"
{text}
\"\"\"
"""


def _build_prompt(
    text: str,
    source_file: str = "",
    page_start: int = 0,
    page_end: int = 0,
) -> str:
    """Build the extraction prompt for a single text chunk.

    Args:
        text: The chunk text to analyse.
        source_file: Basename of the PDF for attribution.
        page_start: First page number covered by this chunk.
        page_end: Last page number covered by this chunk.

    Returns:
        Fully-formatted prompt string.
    """
    return _EXTRACTION_PROMPT.format(
        text=text[:3000],  # guard against accidentally huge chunks
        source_file=source_file or "unknown",
        page_start=page_start,
        page_end=page_end,
    )


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> str:
    """Send a generation request to Ollama and return the raw response text.

    Args:
        prompt: The full prompt string.
        model: Ollama model tag (e.g. ``"llama3.2"``).
        base_url: Base URL of the running Ollama server.
        timeout: HTTP request timeout in seconds.

    Returns:
        Raw text content of the model's reply.

    Raises:
        ConnectionError: If the Ollama server is unreachable.
        RuntimeError: If the server returns a non-200 status.
    """
    url = f"{base_url}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Ensure Ollama is running: `ollama serve`"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
        )

    data = response.json()
    return data.get("response", "")


def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object found in *raw*.

    The model occasionally wraps output in markdown fences; this strips them
    before attempting to parse.

    Args:
        raw: Raw model output that should contain a JSON object.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON object could be found.
    """
    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the outermost { ... } block.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw[:200]}")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error in model output: {exc}") from exc


def parse_extraction(raw_json: Dict[str, Any]) -> Tuple[List[Node], List[dict]]:
    """Convert the parsed model JSON into Node objects and raw edge dicts.

    Edges are returned as dicts keyed by source/target *labels* because the
    caller must resolve labels to Node ids after deduplication.

    Args:
        raw_json: Dictionary with ``"concepts"`` and ``"relationships"`` keys.

    Returns:
        Tuple of ``(nodes, edge_dicts)`` where *edge_dicts* have keys
        ``source``, ``target``, ``relation``, ``confidence``.
    """
    nodes: List[Node] = []
    for concept in raw_json.get("concepts", []):
        label = str(concept.get("label", "")).strip()
        if not label:
            continue
        nodes.append(
            Node(
                label=label,
                description=str(concept.get("description", "")),
                confidence=float(concept.get("confidence", 1.0)),
            )
        )

    edge_dicts: List[dict] = []
    for rel in raw_json.get("relationships", []):
        src = str(rel.get("source", "")).strip()
        tgt = str(rel.get("target", "")).strip()
        if src and tgt:
            edge_dicts.append(
                {
                    "source": src,
                    "target": tgt,
                    "relation": str(rel.get("relation", "related_to")),
                    "confidence": float(rel.get("confidence", 1.0)),
                }
            )

    return nodes, edge_dicts


def extract_from_chunk(
    text: str,
    source_file: str = "",
    page_start: int = 0,
    page_end: int = 0,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> Tuple[List[Node], List[dict]]:
    """Run the full extraction pipeline for a single text chunk.

    Args:
        text: Chunk text.
        source_file: PDF filename for attribution.
        page_start: First page in this chunk.
        page_end: Last page in this chunk.
        model: Ollama model tag.
        base_url: Ollama server URL.

    Returns:
        Tuple of ``(nodes, edge_dicts)``.  On LLM or parse failure an empty
        tuple is returned and the error is logged.
    """
    prompt = _build_prompt(text, source_file, page_start, page_end)
    try:
        raw = call_ollama(prompt, model=model, base_url=base_url)
        parsed = _extract_json(raw)
        nodes, edges = parse_extraction(parsed)
        # Tag nodes with provenance.
        for node in nodes:
            node.source_paper = source_file
            node.page_refs = list(range(page_start, page_end + 1))
        return nodes, edges
    except (ConnectionError, RuntimeError, ValueError) as exc:
        logger.warning("Extraction failed for chunk from '%s': %s", source_file, exc)
        return [], []


def build_graph_from_chunks(
    chunks,  # List[core.pdf_ingestion.Chunk]
    graph: Optional[GraphState] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> GraphState:
    """Extract concepts from every chunk and accumulate them into a GraphState.

    Labels from relationships are matched against extracted node labels
    (case-insensitive) to resolve edge endpoints.  Unresolvable endpoints are
    skipped with a warning.

    Args:
        chunks: Iterable of :class:`~core.pdf_ingestion.Chunk` objects.
        graph: Existing GraphState to append to, or None to create a new one.
        model: Ollama model tag.
        base_url: Ollama server URL.

    Returns:
        Populated :class:`~core.graph_state.GraphState`.
    """
    if graph is None:
        graph = GraphState()

    for chunk in chunks:
        nodes, edge_dicts = extract_from_chunk(
            text=chunk.text,
            source_file=chunk.source_file,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            model=model,
            base_url=base_url,
        )

        # Build a label → node_id map for this chunk's nodes so we can wire
        # edges before adding to the graph.
        label_to_id: Dict[str, str] = {}
        for node in nodes:
            graph.add_node(node)
            label_to_id[node.label.lower()] = node.id

        for ed in edge_dicts:
            src_id = label_to_id.get(ed["source"].lower())
            tgt_id = label_to_id.get(ed["target"].lower())
            if src_id is None or tgt_id is None:
                logger.debug(
                    "Skipping edge '%s' → '%s': label not resolved.",
                    ed["source"],
                    ed["target"],
                )
                continue
            try:
                graph.add_edge(
                    Edge(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation=ed["relation"],
                        confidence=ed["confidence"],
                        source_paper=chunk.source_file,
                    )
                )
            except ValueError as exc:
                logger.debug("Edge skipped: %s", exc)

    return graph
