"""
query_mode.py — Voice Q&A over knowledge-graph nodes.

Pipeline:
  1. Record audio from the microphone via PyAudio / SpeechRecognition.
  2. Transcribe with OpenAI Whisper (local, no API key).
  3. Find the most relevant graph nodes using embedding similarity.
  4. Ask Ollama to synthesise an answer grounded in those nodes.
  5. Return the answer text (caller decides how to display it).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import List, Tuple

import numpy as np

from core.graph_state import GraphState, Node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — heavy models are loaded on first use.
# ---------------------------------------------------------------------------
# NOTE: threading and queue are used by callers who run voice_query in a
# background thread.  They are imported here as a convenience re-export so
# callers don't need a separate import.
import queue  # noqa: F401  (re-exported for caller convenience)
import threading  # noqa: F401  (re-exported for caller convenience)

_whisper_model = None
_embedder = None


def _load_whisper(model_size: str = "base") -> object:
    """Load the Whisper ASR model (cached after first call).

    Args:
        model_size: Whisper model variant (``"tiny"``, ``"base"``, ``"small"``…).

    Returns:
        A loaded whisper model instance.

    Raises:
        ImportError: If openai-whisper is not installed.
        RuntimeError: If the model cannot be loaded.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required.  Install with: pip install openai-whisper"
        ) from exc
    try:
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper model '%s' loaded.", model_size)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Whisper model '{model_size}': {exc}") from exc
    return _whisper_model


def _load_embedder():
    """Load the sentence-transformer model (cached after first call)."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedder: {exc}") from exc
    return _embedder


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

def record_audio_blocking(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Record mono audio from the default microphone (blocking).

    Args:
        duration_seconds: Recording duration.
        sample_rate: Audio sample rate in Hz (Whisper expects 16 kHz).

    Returns:
        Float32 numpy array of shape ``(n_samples,)`` normalised to [-1, 1].

    Raises:
        RuntimeError: If PyAudio or a microphone is unavailable.
    """
    try:
        import pyaudio
    except ImportError as exc:
        raise ImportError(
            "pyaudio is required.  Install with: pip install pyaudio"
        ) from exc

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024,
        )
        frames = []
        n_chunks = int(sample_rate / 1024 * duration_seconds)
        for _ in range(n_chunks):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
    except Exception as exc:
        raise RuntimeError(f"Audio recording failed: {exc}") from exc
    finally:
        pa.terminate()

    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def transcribe_audio(
    audio: np.ndarray,
    model_size: str = "base",
    language: str = "en",
) -> str:
    """Transcribe audio with Whisper.

    Args:
        audio: Float32 numpy array at 16 kHz.
        model_size: Whisper model variant.
        language: ISO 639-1 language code.

    Returns:
        Transcribed text string.
    """
    model = _load_whisper(model_size)
    result = model.transcribe(audio, language=language, fp16=False)
    text = result.get("text", "").strip()
    logger.debug("Whisper transcript: %r", text)
    return text


# ---------------------------------------------------------------------------
# Semantic node retrieval
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> np.ndarray:
    """Embed a query string to a unit-norm vector.

    Args:
        query: User's question text.

    Returns:
        Float32 1-D numpy array.
    """
    model = _load_embedder()
    vec = model.encode([query], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def retrieve_relevant_nodes(
    query: str,
    graph: GraphState,
    top_k: int = 5,
) -> List[Tuple[Node, float]]:
    """Return the *top_k* graph nodes most semantically similar to *query*.

    Nodes that lack embeddings are assigned one on-the-fly.

    Args:
        query: The user's question.
        graph: The knowledge graph to search.
        top_k: Number of nodes to return.

    Returns:
        List of ``(node, similarity_score)`` tuples sorted descending.
    """
    nodes = graph.nodes
    if not nodes:
        return []

    embedder = _load_embedder()
    # Ensure all nodes have embeddings.
    missing = [n for n in nodes if n.embedding is None]
    if missing:
        vecs = embedder.encode([n.label for n in missing], normalize_embeddings=True)
        for node, vec in zip(missing, vecs):
            node.embedding = vec.tolist()

    query_vec = _embed_query(query)
    scores = []
    for node in nodes:
        if node.embedding is None:
            continue
        node_vec = np.array(node.embedding, dtype=np.float32)
        sim = float(np.dot(query_vec, node_vec))
        scores.append((node, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# Answer synthesis via Ollama
# ---------------------------------------------------------------------------

def synthesise_answer(
    query: str,
    context_nodes: List[Node],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> str:
    """Ask Ollama to answer *query* using *context_nodes* as grounding.

    Args:
        query: The user's question.
        context_nodes: Relevant nodes retrieved from the graph.
        model: Ollama model tag.
        base_url: Ollama server URL.

    Returns:
        Answer string from the LLM.

    Raises:
        ConnectionError: If Ollama is unreachable.
    """
    import requests

    context_text = "\n".join(
        f"- {n.label}: {n.description}" for n in context_nodes
    )
    prompt = (
        "You are a knowledgeable assistant.  Answer the question below using "
        "only the provided concept summaries.  Be concise (2-4 sentences).\n\n"
        f"Concepts:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 256},
    }
    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Cannot reach Ollama at {base_url}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:200]}")

    return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def voice_query(
    graph: GraphState,
    duration_seconds: float = 5.0,
    top_k: int = 5,
    whisper_model: str = "base",
    ollama_model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> Tuple[str, str, List[Node]]:
    """Record audio → transcribe → retrieve nodes → synthesise answer.

    Args:
        graph: The knowledge graph to query.
        duration_seconds: Microphone recording duration.
        top_k: Number of context nodes to retrieve.
        whisper_model: Whisper model size.
        ollama_model: Ollama model tag.
        ollama_url: Ollama base URL.

    Returns:
        Tuple of ``(transcript, answer, context_nodes)``.
    """
    audio = record_audio_blocking(duration_seconds=duration_seconds)
    transcript = transcribe_audio(audio, model_size=whisper_model)
    if not transcript:
        return ("", "No speech detected.", [])

    context_nodes_with_scores = retrieve_relevant_nodes(transcript, graph, top_k=top_k)
    context_nodes = [n for n, _ in context_nodes_with_scores]

    answer = synthesise_answer(
        query=transcript,
        context_nodes=context_nodes,
        model=ollama_model,
        base_url=ollama_url,
    )
    return (transcript, answer, context_nodes)
