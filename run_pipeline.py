from core.pdf_ingestion import ingest_pdf
from core.concept_extractor import build_graph_from_chunks
from core.deduplicator import deduplicate_graph
from features.session_memory import save_session

print("Ingesting PDF...")
chunks = ingest_pdf("data/pdfs/NIPS-2017-attention-is-all-you-need-Paper (1).pdf")
print(f"  {len(chunks)} chunks created")

print("Extracting concepts (this takes a few minutes)...")
graph = build_graph_from_chunks(chunks)
print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges before dedup")

deduplicate_graph(graph)
graph.session_name = "attention_is_all_you_need"
print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges after dedup")

save_session(graph)
print("Saved to data/sessions/attention_is_all_you_need.json")
