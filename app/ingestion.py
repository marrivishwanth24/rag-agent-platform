"""
ingestion.py — Document ingestion with semantic embeddings via Voyage AI
"""

import os
import uuid
import io

import asyncpg
from pypdf import PdfReader


async def get_embedding(text: str) -> list[float]:
    """
    Semantic embedding using Voyage AI voyage-3.5.
    1024 dimensions — input_type="document" optimises for storage-side retrieval.
    """
    import voyageai
    client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))
    result = await client.embed(
        [text],
        model="voyage-3.5",
        input_type="document",
    )
    return result.embeddings[0]


# ── Text extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(contents: bytes) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(io.BytesIO(contents))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    - chunk_size: words per chunk
    - overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ── Main ingestion function ────────────────────────────────────────────────────
async def ingest_document(
    contents: bytes,
    filename: str,
    user_id: str,
    pool: asyncpg.Pool,
) -> str:
    """
    Full ingestion pipeline:
    1. Extract text from PDF
    2. Chunk text with overlap
    3. Generate semantic embeddings via Voyage AI
    4. Store chunks + embeddings in pgvector

    Returns: document_id (UUID)
    """
    document_id = str(uuid.uuid4())

    pages = extract_text_from_pdf(contents)
    print(f"Extracted {len(pages)} pages from {filename}")

    if not pages:
        raise ValueError("Could not extract text from PDF")

    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({"chunk_text": chunk, "page_num": page["page_num"]})

    print(f"Created {len(all_chunks)} chunks")

    async with pool.acquire() as conn:
        for i, chunk_data in enumerate(all_chunks):
            embedding = await get_embedding(chunk_data["chunk_text"])
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            await conn.execute(
                """
                INSERT INTO document_chunks
                    (id, document_id, user_id, filename, page_num, chunk_text, embedding)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7::vector)
                """,
                str(uuid.uuid4()),
                document_id,
                user_id,
                filename,
                chunk_data["page_num"],
                chunk_data["chunk_text"],
                embedding_str,
            )

            if (i + 1) % 10 == 0:
                print(f"Embedded {i + 1}/{len(all_chunks)} chunks...")

    print(f"Ingestion complete — {len(all_chunks)} chunks stored")
    return document_id
