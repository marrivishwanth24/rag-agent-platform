import os
import asyncpg
from app.ingestion import embed_text

TOP_K = 5          # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7


async def retrieve_context(
    query: str,
    document_ids: list[str] = None,
    user_id: str = None,
    top_k: int = TOP_K,
    db_url: str = None
) -> list[dict]:
    """
    Semantic retrieval using pgvector cosine similarity.
    Returns top-k most relevant chunks for the given query.

    Args:
        query: The user's question
        document_ids: Optional filter to search only specific documents
        user_id: Filter chunks by user ownership
        top_k: Number of chunks to return
    """
    db_url = db_url or os.getenv("DATABASE_URL")

    # Embed the query
    query_embedding = embed_text(query)
    embedding_str = str(query_embedding)

    conn = await asyncpg.connect(db_url)
    try:
        # Build query with optional filters
        filters = []
        params = [embedding_str, top_k]
        param_idx = 3

        if user_id:
            filters.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if document_ids:
            placeholders = ", ".join([f"${i}" for i in range(param_idx, param_idx + len(document_ids))])
            filters.append(f"document_id IN ({placeholders})")
            params.extend(document_ids)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        sql = f"""
            SELECT
                document_id,
                filename,
                page_num,
                chunk_text,
                1 - (embedding <=> $1::vector) AS similarity
            FROM document_chunks
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        rows = await conn.fetch(sql, *params)

        return [
            {
                "document_id": row["document_id"],
                "filename": row["filename"],
                "page": row["page_num"],
                "text": row["chunk_text"],
                "similarity": float(row["similarity"])
            }
            for row in rows
            if float(row["similarity"]) >= SIMILARITY_THRESHOLD
        ]

    finally:
        await conn.close()


async def get_document_stats(document_id: str, db_url: str = None) -> dict:
    """Get chunk count and coverage stats for a document."""
    db_url = db_url or os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) as chunk_count,
                MAX(page_num) as max_page,
                filename
            FROM document_chunks
            WHERE document_id = $1
            GROUP BY filename
            """,
            document_id
        )
        return dict(row) if row else {}
    finally:
        await conn.close()
