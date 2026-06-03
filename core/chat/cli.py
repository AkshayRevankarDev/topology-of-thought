"""
cli.py — Interactive REPL for the RAG chat layer.

Usage:
    python -m core.chat.cli                       # default session
    python -m core.chat.cli --session attention   # use a named session

Loads:
    * the latest GraphState JSON from data/sessions/
    * the Chroma collection of the same session name

Commands:
    /reindex <pdf_path...>   chunk + embed PDFs into the current collection
    /count                   show how many chunks are indexed
    /quit                    exit
    <anything else>          treated as a question
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from core.graph_state import GraphState
from core.retrieval.indexer import Indexer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR  = PROJECT_ROOT / 'data' / 'sessions'


def _load_graph(session: str) -> GraphState:
    path = SESSION_DIR / f'{session}.json'
    if not path.exists():
        # Fall back to the most recently modified session.
        candidates = sorted(SESSION_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f'No GraphState found under {SESSION_DIR}')
        path = candidates[-1]
        print(f'[chat] session "{session}" not found; using {path.name}')
    with open(path, 'r', encoding='utf-8') as f:
        return GraphState.from_dict(json.load(f))


def _reindex(indexer: Indexer, pdf_paths: List[str]) -> None:
    from core.ingestion.pdf_ingestion import iter_pdf_documents  # see ingestion task
    docs = list(iter_pdf_documents(pdf_paths))
    n = indexer.add_documents(docs)
    print(f'[chat] indexed {n} chunks from {len(pdf_paths)} document(s)')


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Topology of Thought — RAG chat')
    parser.add_argument('--session', default='default',
                        help='Session name (GraphState filename + Chroma collection).')
    args = parser.parse_args(argv)

    graph = _load_graph(args.session)
    indexer = Indexer.default(session=args.session)

    # Import here so Ollama only loads if the user actually asks something.
    from core.chat import RagChat
    chat = RagChat(indexer=indexer, graph=graph)

    print(f'[chat] graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges')
    print(f'[chat] vector index: {indexer.count()} chunks')
    print('[chat] type your question, or /quit to exit.')

    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == '/quit':
            return 0
        if line == '/count':
            print(indexer.count(), 'chunks')
            continue
        if line.startswith('/reindex'):
            paths = line.split()[1:]
            if not paths:
                print('usage: /reindex <pdf> [<pdf>...]')
                continue
            _reindex(indexer, paths)
            continue

        ans = chat.ask(line)
        print()
        print(ans.answer)
        print()
        if ans.citations:
            print('Sources:')
            for c in ans.citations:
                where = c.source_paper or c.source_id
                pages = f' p.{",".join(str(p) for p in c.page_refs)}' if c.page_refs else ''
                print(f'  [{c.marker}] {where}{pages}  (score {c.score:.2f})')
        if ans.node_ids_lit:
            print(f'Graph highlights: {len(ans.node_ids_lit)} nodes lit')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
