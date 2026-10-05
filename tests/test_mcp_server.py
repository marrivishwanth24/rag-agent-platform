"""Tests for the MCP server's tool functions.

Mocks the DB pool / retrieval layer — no real Postgres connection or MCP
client needed. Verifies the MCP tools correctly delegate to app.retrieval
(the exact same pipeline the web app uses) and respect session scoping.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.mcp_server import list_documents_for_session, search_documents

SAMPLE_CHUNKS = [
    {"chunk_text": "revenue grew 20%", "page_num": 3, "filename": "q3.pdf", "similarity": 0.91},
]
SAMPLE_DOCS = [{"document_id": "d1", "filename": "q3.pdf", "page_count": 12}]


@pytest.mark.asyncio
async def test_search_documents_delegates_to_retrieve_context_with_session_id():
    with patch("app.mcp_server._get_pool", new=AsyncMock(return_value="fake-pool")), \
         patch("app.mcp_server.retrieve_context", new=AsyncMock(return_value=SAMPLE_CHUNKS)) as mock_retrieve:
        result = await search_documents(session_id="user-abc", query="revenue growth", top_k=3)

    assert result == SAMPLE_CHUNKS
    mock_retrieve.assert_called_once_with(
        "revenue growth", document_ids=[], user_id="user-abc", pool="fake-pool", top_k=3
    )


@pytest.mark.asyncio
async def test_search_documents_default_top_k():
    with patch("app.mcp_server._get_pool", new=AsyncMock(return_value="fake-pool")), \
         patch("app.mcp_server.retrieve_context", new=AsyncMock(return_value=[])) as mock_retrieve:
        await search_documents(session_id="user-abc", query="anything")

    assert mock_retrieve.call_args.kwargs["top_k"] == 5


@pytest.mark.asyncio
async def test_list_documents_for_session_delegates_with_session_id():
    with patch("app.mcp_server._get_pool", new=AsyncMock(return_value="fake-pool")), \
         patch("app.mcp_server._list_documents", new=AsyncMock(return_value=SAMPLE_DOCS)) as mock_list:
        result = await list_documents_for_session(session_id="user-abc")

    assert result == SAMPLE_DOCS
    mock_list.assert_called_once_with("user-abc", "fake-pool")


@pytest.mark.asyncio
async def test_two_sessions_cannot_see_each_others_results():
    """The tool's only scoping input is session_id — different IDs must produce
    independently-scoped calls, proving one session can't read another's data."""
    with patch("app.mcp_server._get_pool", new=AsyncMock(return_value="fake-pool")), \
         patch("app.mcp_server.retrieve_context", new=AsyncMock(return_value=[])) as mock_retrieve:
        await search_documents(session_id="user-a", query="q")
        await search_documents(session_id="user-b", query="q")

    called_user_ids = [c.kwargs["user_id"] for c in mock_retrieve.call_args_list]
    assert called_user_ids == ["user-a", "user-b"]
