"""Tests for the PDF ingestion pipeline."""
from app.ingestion import chunk_text


def test_chunk_text_basic():
    """Text longer than chunk_size gets split."""
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_short_input():
    """Text shorter than chunk_size returns one chunk."""
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_input():
    """Empty text returns no chunks."""
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert len(chunks) == 0


def test_chunk_text_no_empty_chunks():
    """No chunk is empty or whitespace only."""
    text = " ".join(["word"] * 1000)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_text_preserves_content():
    """All words from input appear somewhere in chunks."""
    text = "alpha beta gamma delta epsilon"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    combined = " ".join(chunks)
    for word in text.split():
        assert word in combined


def test_chunk_text_overlap():
    """Consecutive chunks share overlapping words."""
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    if len(chunks) > 1:
        chunk1_last = chunks[0].split()[-50:]
        chunk2_first = chunks[1].split()[:50]
        assert chunk1_last == chunk2_first