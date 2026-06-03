"""
rag_chat.py — Graph-augmented RAG conversational layer.

Pipeline per query:
    1. Embed the query, retrieve top-k chunks from the vector index.
    2. Score each GraphState node against the query (cosine on the same
       embedding model).  Take the top-m nodes.
    3. Pull the immediate neighbourhood of those top nodes from the graph —
       this is the "graph augmentation" step that distinguishes this from a
       vanilla RAG.
    4. Build a prompt that lists chunks as numbered evidence and the graph
       neighbourhood as a structured context block.
    5. Call Ollama (reusing concept_extractor.call_ollama) and return the
       answer plus resolved citations.

Citations resolve a `[n]` marker in the answer back to:
    {chunk_id, source_id, source_paper, page_refs, score, node_ids}

Both the chat layer and the TD visualizer can light up the same node_ids,
giving the Tony-Stark feeling: ask a question, the relevant constellation
glows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.graph_state import GraphState
from core.retrieval.indexer import Indexer
from core.retrieval.vector_index import Retrieved

logger = logging.getLogger(__name__)

DEFAULT_K_CHUNKS = 6
DEFAULT_K_NODES  = 8


@dataclass
class Citation:
    """One numbered citation surfaced in an answer."""
    marker: int                       # the [n] number the LLM used
    chunk_id: str
    source_id: str
    source_paper: str = ''
    page_refs: List[int] = field(default_factory=list)
    score: float = 0.0
    node_ids: List[str] = field(default_factory=list)
    text_snippet: str = ''


@dataclass
class Answer:
    """Bundle returned by RagChat.ask()."""
    question: str
    answer: str
    citations: List[Citation]
    node_ids_lit: List[str]   # graph nodes the UI should highlight


class RagChat:
    """Stateless graph-augmented RAG over a GraphState + vector index.

    Args:
        indexer: A configured :class:`core.retrieval.Indexer`.
        graph:   The live :class:`GraphState` (we read embeddings + metadata).
        model:   Ollama model tag (default mirrors concept_extractor).
    """

    def __init__(
        self,
        indexer: Indexer,
        graph: GraphState,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.indexer = indexer
        self.graph = graph
        # Late import to keep concept_extractor's heavy deps out of import-time.
        from core import concept_extractor as ce
        self._call_ollama = ce.call_ollama
        self.model = model or ce.DEFAULT_MODEL
        self.base_url = base_url or ce.OLLAMA_BASE_URL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        k_chunks: int = DEFAULT_K_CHUNKS,
        k_nodes: int = DEFAULT_K_NODES,
    ) -> Answer:
        """Run the full RAG pipeline and return a structured Answer."""
        hits = self.indexer.search(question, k=k_chunks)
        nodes_relevant = self._rank_nodes(question, k=k_nodes)
        neighbourhood  = self._expand_neighbourhood(nodes_relevant)

        prompt = self._build_prompt(question, hits, neighbourhood)
        raw = self._call_ollama(prompt, model=self.model, base_url=self.base_url)
        answer_text, used_markers = self._parse_markers(raw)
        citations = self._resolve_citations(hits, used_markers, nodes_relevant)

        node_ids_lit = list({nid for n in nodes_relevant for nid in [n[0]]})
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            node_ids_lit=node_ids_lit,
        )

    # ------------------------------------------------------------------
    # Graph augmentation
    # ------------------------------------------------------------------
    def _rank_nodes(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Cosine-rank graph nodes by their stored label embedding.

        Returns a list of (node_id, score) sorted desc.  Nodes without an
        embedding are skipped silently.
        """
        nodes_with_emb = [
            (n.id, np.asarray(n.embedding, dtype=np.float32))
            for n in self.graph.nodes
            if n.embedding is not None
        ]
        if not nodes_with_emb:
            return []

        # Embed the query in the same space.
        try:
            from core import deduplicator
            embedder = deduplicator._get_embedder()  # noqa: SLF001
            q_vec = np.asarray(
                embedder.encode([query], show_progress_bar=False)[0],
                dtype=np.float32,
            )
        except Exception as exc:
            logger.warning('Could not embed query for node ranking: %s', exc)
            return []

        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scored: List[Tuple[str, float]] = []
        for nid, emb in nodes_with_emb:
            n = emb / (np.linalg.norm(emb) + 1e-9)
            scored.append((nid, float(np.dot(n, q_norm))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _expand_neighbourhood(
        self,
        ranked_nodes: List[Tuple[str, float]],
    ) -> List[Dict[str, Any]]:
        """For each top node, gather 1-hop neighbours from the graph."""
        out: List[Dict[str, Any]] = []
        for nid, score in ranked_nodes:
            node = self.graph.get_node(nid)
            if node is None:
                continue
            neighbours = []
            for nbr_id, edge in self.graph.neighbors(nid):
                nbr = self.graph.get_node(nbr_id)
                if nbr is None:
                    continue
                neighbours.append({
                    'label': nbr.label,
                    'relation': edge.relation,
                })
            out.append({
                'label': node.label,
                'description': node.description,
                'score': score,
                'neighbours': neighbours,
            })
        return out

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        question: str,
        hits: List[Retrieved],
        neighbourhood: List[Dict[str, Any]],
    ) -> str:
        evidence_lines: List[str] = []
        for i, hit in enumerate(hits, start=1):
            paper = hit.source_meta.get('paper') or hit.source_id
            page  = hit.source_meta.get('page', '')
            tag   = f'{paper} p.{page}' if page != '' else paper
            evidence_lines.append(f'[{i}] ({tag}) {hit.text.strip()}')
        evidence = '\n\n'.join(evidence_lines) if evidence_lines else '(no retrieved evidence)'

        graph_lines: List[str] = []
        for item in neighbourhood:
            neighbours = ', '.join(
                f'{nb["label"]} ({nb["relation"]})' for nb in item['neighbours']
            ) or '—'
            graph_lines.append(
                f'• {item["label"]}: {item["description"][:160]}\n'
                f'    related → {neighbours}'
            )
        graph_block = '\n'.join(graph_lines) if graph_lines else '(no related concepts)'

        return (
            'You are a careful research assistant. Answer the user question using\n'
            'ONLY the evidence and concept map provided below. Cite every claim\n'
            'with bracketed numbers like [1], [2] that correspond to the evidence\n'
            'items. If the evidence does not answer the question, say so plainly.\n\n'
            f'### Question\n{question}\n\n'
            f'### Evidence\n{evidence}\n\n'
            f'### Related concepts (from the knowledge graph)\n{graph_block}\n\n'
            '### Answer (with [n] citations)\n'
        )

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    _MARKER_RE = re.compile(r'\[(\d+)\]')

    def _parse_markers(self, raw: str) -> Tuple[str, List[int]]:
        text = raw.strip()
        markers = sorted({int(m.group(1)) for m in self._MARKER_RE.finditer(text)})
        return text, markers

    def _resolve_citations(
        self,
        hits: List[Retrieved],
        used_markers: List[int],
        ranked_nodes: List[Tuple[str, float]],
    ) -> List[Citation]:
        node_ids_pool = [nid for nid, _ in ranked_nodes]
        citations: List[Citation] = []
        for m in used_markers:
            idx = m - 1
            if idx < 0 or idx >= len(hits):
                continue
            hit = hits[idx]
            citations.append(Citation(
                marker=m,
                chunk_id=hit.chunk_id,
                source_id=hit.source_id,
                source_paper=str(hit.source_meta.get('paper', '')),
                page_refs=[int(hit.source_meta['page'])] if 'page' in hit.source_meta else [],
                score=hit.score,
                node_ids=list(node_ids_pool),
                text_snippet=hit.text[:240],
            ))
        return citations
