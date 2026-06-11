"""
retrieval.py — Semantic retrieval using vector similarity
"""

import os
import asyncpg


async def get_embedding(text: str) -> list[float]:
    """
    Same model as ingestion — MUST match exactly.
    Uses input_type='query' so Voyage AI optimises for the query side.
    """
    import voyageai
    client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))
    result = await client.embed(
        [text],
        model="voyage-3.5",
        input_type="query",
    )
    return result.embeddings[0]


async def retrieve_context(
    query: str,
    document_ids: list[str],
    user_id: str,
    pool: asyncpg.Pool,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantic retrieval pipeline:
    1. Embed the query with the same model used during ingestion
    2. Run cosine similarity search in pgvector
    3. Return top_k most semantically similar chunks
    """
    query_embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    async with pool.acquire() as conn:
        if document_ids:
            rows = await conn.fetch(
                """
                SELECT chunk_text, page_num, filename,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM document_chunks
                WHERE user_id = $2::uuid
                  AND document_id = ANY($3::text[])
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding_str,
                user_id,
                document_ids,
                top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT chunk_text, page_num, filename,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM document_chunks
                WHERE user_id = $2::uuid
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                user_id,
                top_k,
            )

    return [
        {
            "chunk_text": row["chunk_text"],
            "page_num": row["page_num"],
            "filename": row["filename"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


async def list_documents(user_id: str, pool: asyncpg.Pool) -> list[dict]:
    """Return distinct documents uploaded by a user."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, filename, MAX(page_num) AS page_count
            FROM document_chunks
            WHERE user_id = $1::uuid
            GROUP BY document_id, filename
            ORDER BY filename
            """,
            user_id,
        )
    return [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "page_count": row["page_count"],
        }
        for row in rows
    ]
