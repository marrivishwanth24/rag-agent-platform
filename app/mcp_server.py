"""
mcp_server.py — exposes this platform's document search as MCP tools.

Gives any MCP client (Claude Desktop, another agent, etc.) controlled,
read-only access to a user's uploaded documents, reusing the exact same
pgvector + Voyage AI rerank retrieval pipeline the web app uses — no
duplicated logic, just a second interface onto the same retrieval code
(app.retrieval).

Deliberately isolated from the main FastAPI app:
  - Separate entry point, separate process (stdio transport — the standard
    for most MCP clients, which spawn the server as a subprocess).
  - Separate requirements file (requirements-mcp.txt). The `mcp` SDK pulls
    in a Starlette version that conflicts with FastAPI 0.115.0's pin
    (discovered while building this — `pip install mcp` upgrades Starlette
    to 1.x, but FastAPI 0.115.0 requires <0.39.0). Installing both in the
    same environment breaks the web app, so the MCP server is a separate
    deployable unit rather than an in-process addition to main.py.

Session scoping: every tool takes a session_id argument that an MCP client
must be configured with out-of-band (e.g. in its server config) — the server
never infers or widens it, so a given MCP server instance can only ever see
one session's documents, the same isolation model the web app uses.

Run standalone:
    pip install -r requirements-mcp.txt
    DATABASE_URL=postgresql://... python -m app.mcp_server
"""

import os

import asyncpg
from mcp.server.mcpserver import MCPServer

from app.retrieval import list_documents as _list_documents
from app.retrieval import retrieve_context

mcp = MCPServer("rag-agent-platform")

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=5)
    return _pool


@mcp.tool()
async def search_documents(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search a user's uploaded documents and return the most relevant passages.

    Runs the same two-stage retrieval the web app uses: pgvector ANN search
    followed by a Voyage AI cross-encoder rerank. Each result includes the
    source filename, page number, the passage text, and a similarity score.

    session_id scopes the search to one user's private document space —
    results can never include another session's documents.
    """
    pool = await _get_pool()
    return await retrieve_context(query, document_ids=[], user_id=session_id, pool=pool, top_k=top_k)


@mcp.tool()
async def list_documents_for_session(session_id: str) -> list[dict]:
    """List the documents uploaded under a given session_id (filename + page count)."""
    pool = await _get_pool()
    return await _list_documents(session_id, pool)


if __name__ == "__main__":
    mcp.run(transport="stdio")
