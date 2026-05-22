"""
pdf_ingestion.py — PDF text extraction and sentence-aware chunking.

Uses PyMuPDF (fitz) for zero-dependency local PDF parsing.  Pages are
extracted as plain text, then split into overlapping chunks that fit
comfortably inside an LLM context window.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, List, NamedTuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError(
        "PyMuPDF is required for PDF ingestion.  "
        "Install it with: pip install pymupdf"
    ) from exc


class Chunk(NamedTuple):
    """A single text chunk produced by the ingestion pipeline.

    Attributes:
        text: The raw chunk text.
        page_start: First page number (1-indexed) covered by this chunk.
        page_end: Last page number (1-indexed) covered by this chunk.
        source_file: Basename of the originating PDF.
        chunk_index: Sequential index of this chunk within the document.
    """

    text: str
    page_start: int
    page_end: int
    source_file: str
    chunk_index: int


def extract_text_by_page(pdf_path: str | Path) -> List[tuple[int, str]]:
    """Extract plain text from every page of a PDF.

    Args:
        pdf_path: Filesystem path to the PDF file.

    Returns:
        A list of ``(page_number, text)`` tuples where *page_number* is
        1-indexed.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist.
        RuntimeError: If PyMuPDF is unable to open or parse the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"PyMuPDF failed to open '{pdf_path}': {exc}") from exc

    pages: List[tuple[int, str]] = []
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            # Collapse excessive whitespace while preserving paragraph breaks.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                pages.append((page_num + 1, text))
    finally:
        doc.close()

    return pages


def _sentence_split(text: str) -> List[str]:
    """Split *text* into individual sentences using a simple regex heuristic.

    Args:
        text: Raw paragraph or page text.

    Returns:
        List of sentence strings with leading/trailing whitespace stripped.
    """
    # Split on '.', '!', '?' followed by whitespace and an uppercase letter or
    # end-of-string — good enough for academic English.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"]|$)", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(
    pages: List[tuple[int, str]],
    source_file: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Chunk]:
    """Merge page text into overlapping word-count chunks.

    The overlap ensures that concepts spanning chunk boundaries are not split
    apart for the LLM extraction step.

    Args:
        pages: Output of :func:`extract_text_by_page`.
        source_file: Basename used to tag each Chunk.
        chunk_size: Target chunk size in *words* (not characters).
        overlap: Number of words carried over from the previous chunk.

    Returns:
        Ordered list of :class:`Chunk` objects ready for LLM processing.

    Raises:
        ValueError: If *chunk_size* <= *overlap*.
    """
    if chunk_size <= overlap:
        raise ValueError(
            f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})."
        )

    # Flatten pages into a list of (word, page_number) pairs.
    word_page: List[tuple[str, int]] = []
    for page_num, text in pages:
        for word in text.split():
            word_page.append((word, page_num))

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    idx = 0
    chunk_index = 0

    while idx < len(word_page):
        window = word_page[idx: idx + chunk_size]
        words = [w for w, _ in window]
        pages_in_window = [p for _, p in window]
        text_block = " ".join(words)
        chunks.append(
            Chunk(
                text=text_block,
                page_start=pages_in_window[0],
                page_end=pages_in_window[-1],
                source_file=source_file,
                chunk_index=chunk_index,
            )
        )
        idx += step
        chunk_index += 1

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[Chunk]:
    """Full pipeline: open PDF → extract text → return chunks.

    This is the primary entry-point for downstream consumers.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Target chunk size in words.
        overlap: Word overlap between consecutive chunks.

    Returns:
        List of :class:`Chunk` objects.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If extraction fails.
    """
    pdf_path = Path(pdf_path)
    pages = extract_text_by_page(pdf_path)
    if not pages:
        raise RuntimeError(f"No extractable text found in '{pdf_path.name}'.")
    return chunk_pages(pages, source_file=pdf_path.name, chunk_size=chunk_size, overlap=overlap)


def iter_chunks(pdf_path: str | Path, **kwargs) -> Generator[Chunk, None, None]:
    """Lazy generator variant of :func:`ingest_pdf`.

    Useful when processing very large PDFs where holding all chunks in memory
    at once is undesirable.

    Args:
        pdf_path: Path to the PDF.
        **kwargs: Forwarded to :func:`ingest_pdf`.

    Yields:
        :class:`Chunk` instances in document order.
    """
    for chunk in ingest_pdf(pdf_path, **kwargs):
        yield chunk
