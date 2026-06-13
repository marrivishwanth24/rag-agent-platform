"""
retrieval.py — Semantic retrieval using vector similarity + reranking
"""

import os
import uuid
import asyncpg
import voyageai


async def get_embedding(text: str) -> list[float]:
    client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))
    result = await client.embed([text], model="voyage-3.5", input_type="query")
    return result.embeddings[0]


async def vector_search(
    query: str,
    document_ids: list[str],
    user_id: str,
    pool: asyncpg.Pool,
    limit: int,
) -> list[dict]:
    """Initial ANN retrieval from pgvector."""
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    user_uuid = uuid.UUID(user_id)

    async with pool.acquire() as conn:
        if document_ids:
            rows = await conn.fetch(
                """
                SELECT chunk_text, page_num, filename,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM document_chunks
                WHERE user_id = $2
                  AND document_id::text = ANY($3)
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                embedding_str, user_uuid, document_ids, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT chunk_text, page_num, filename,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM document_chunks
                WHERE user_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str, user_uuid, limit,
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


async def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates with Voyage AI cross-encoder, normalize scores to [0,1]."""
    client = voyageai.AsyncClient(api_key=os.getenv("VOYAGE_API_KEY"))
    result = await client.rerank(
        query=query,
        documents=[c["chunk_text"] for c in chunks],
        model="rerank-2",
        top_k=top_k,
    )

    scores = [r.relevance_score for r in result.results]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi != lo else 1.0

    reranked = []
    for r in result.results:
        chunk = dict(chunks[r.index])
        chunk["similarity"] = (r.relevance_score - lo) / span
        reranked.append(chunk)
    return reranked


async def retrieve_context(
    query: str,
    document_ids: list[str],
    user_id: str,
    pool: asyncpg.Pool,
    top_k: int = 5,
) -> list[dict]:
    """
    Two-stage retrieval:
    1. Fetch top_k * 3 candidates from pgvector (fast ANN)
    2. Rerank with Voyage AI cross-encoder (accurate), return top_k
    """
    candidates = await vector_search(query, document_ids, user_id, pool, limit=top_k * 3)
    if not candidates:
        return []
    return await rerank(query, candidates, top_k=top_k)


async def list_documents(user_id: str, pool: asyncpg.Pool) -> list[dict]:
    """Return distinct documents uploaded by a user."""
    user_uuid = uuid.UUID(user_id)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, filename, MAX(page_num) AS page_count
            FROM document_chunks
            WHERE user_id = $1
            GROUP BY document_id, filename
            ORDER BY filename
            """,
            user_uuid,
        )
    return [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "page_count": row["page_count"],
        }
        for row in rows
    ]
