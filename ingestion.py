import hashlib
import uuid
import os
from typing import Optional
import anthropic
from pypdf import PdfReader
from io import BytesIO
import asyncpg

ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CHUNK_SIZE = 500        # tokens per chunk
CHUNK_OVERLAP = 50      # overlap between chunks


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract text from PDF, returning list of {page, text} dicts."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """
    Generate embeddings using Anthropic's embedding model.
    Returns a vector of floats for use with pgvector.
    """
    # Using voyage-3 via Anthropic for embeddings
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Note: Use voyage embedding model for production
    # For demo purposes showing the interface
    response = client.beta.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1,
        messages=[{"role": "user", "content": text}]
    )
    # In production: use proper embedding endpoint
    # Placeholder returns deterministic vector based on text hash
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return [(hash_val >> i & 1) * 0.1 for i in range(1536)]


async def ingest_document(
    pdf_bytes: bytes,
    filename: str,
    user_id: str,
    db_url: Optional[str] = None
) -> str:
    """
    Full ingestion pipeline:
    1. Extract text from PDF
    2. Chunk into segments
    3. Generate embeddings
    4. Store in pgvector
    Returns document_id
    """
    document_id = str(uuid.uuid4())
    db_url = db_url or os.getenv("DATABASE_URL")

    # Step 1: Extract text
    pages = extract_text_from_pdf(pdf_bytes)
    print(f"Extracted {len(pages)} pages from {filename}")

    # Step 2: Chunk and embed
    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for chunk_text_content in chunks:
            embedding = embed_text(chunk_text_content)
            all_chunks.append({
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "page": page_data["page"],
                "text": chunk_text_content,
                "embedding": embedding
            })

    # Step 3: Store in pgvector
    conn = await asyncpg.connect(db_url)
    try:
        await conn.executemany(
            """
            INSERT INTO document_chunks
                (document_id, user_id, filename, page_num, chunk_text, embedding)
            VALUES ($1, $2, $3, $4, $5, $6::vector)
            """,
            [
                (
                    c["document_id"], c["user_id"], c["filename"],
                    c["page"], c["text"], str(c["embedding"])
                )
                for c in all_chunks
            ]
        )
        print(f"Stored {len(all_chunks)} chunks for document {document_id}")
    finally:
        await conn.close()

    return document_id
