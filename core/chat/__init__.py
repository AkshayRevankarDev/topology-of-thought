"""
core.chat — conversational layer over the indexed corpus + knowledge graph.

The architecture is intentionally small:

    user query
       │
       ▼
   retrieval (top-k chunks)  ──► graph augmentation (related Nodes)
       │                                    │
       └──────────► prompt builder ◄────────┘
                          │
                          ▼
                  LLM (Ollama)
                          │
                          ▼
               answer with [n]-style citations resolved back to
               (source_paper, page_refs) on the way out.
"""

from core.chat.rag_chat import RagChat, Answer, Citation

__all__ = ['RagChat', 'Answer', 'Citation']
