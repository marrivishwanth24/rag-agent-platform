"""Tests for retrieval and list_documents — mocks DB and embedding calls."""
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import retrieve_context, list_documents


def make_mock_pool(rows: list) -> MagicMock:
    """Build an asyncpg Pool mock whose acquire() yields a connection returning rows."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = rows

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = cm
    return mock_pool


SAMPLE_ROWS = [
    {"chunk_text": "Paris is the capital of France.", "page_num": 1, "filename": "geo.pdf", "similarity": 0.91},
    {"chunk_text": "France is in Western Europe.", "page_num": 2, "filename": "geo.pdf", "similarity": 0.75},
]

MOCK_EMBEDDING = [0.1] * 1024


# ── retrieve_context ───────────────────────────────────────────────────────────

async def test_retrieve_context_returns_correct_count():
    pool = make_mock_pool(SAMPLE_ROWS)
    with patch("app.retrieval.get_embedding", new=AsyncMock(return_value=MOCK_EMBEDDING)):
        results = await retrieve_context("capital of France", [], "user-1", pool)
    assert len(results) == 2


async def test_retrieve_context_result_keys():
    pool = make_mock_pool(SAMPLE_ROWS[:1])
    with patch("app.retrieval.get_embedding", new=AsyncMock(return_value=MOCK_EMBEDDING)):
        results = await retrieve_context("capital", [], "user-1", pool)
    assert set(results[0].keys()) == {"chunk_text", "page_num", "filename", "similarity"}


async def test_retrieve_context_similarity_is_float():
    pool = make_mock_pool(SAMPLE_ROWS[:1])
    with patch("app.retrieval.get_embedding", new=AsyncMock(return_value=MOCK_EMBEDDING)):
        results = await retrieve_context("capital", [], "user-1", pool)
    assert isinstance(results[0]["similarity"], float)


async def test_retrieve_context_empty_db_returns_empty_list():
    pool = make_mock_pool([])
    with patch("app.retrieval.get_embedding", new=AsyncMock(return_value=MOCK_EMBEDDING)):
        results = await retrieve_context("anything", [], "user-1", pool)
    assert results == []


async def test_retrieve_context_calls_get_embedding_once():
    pool = make_mock_pool(SAMPLE_ROWS)
    mock_embed = AsyncMock(return_value=MOCK_EMBEDDING)
    with patch("app.retrieval.get_embedding", new=mock_embed):
        await retrieve_context("test query", [], "user-1", pool)
    mock_embed.assert_called_once()


# ── list_documents ─────────────────────────────────────────────────────────────

async def test_list_documents_returns_correct_count():
    rows = [
        {"document_id": "doc-1", "filename": "a.pdf", "page_count": 3},
        {"document_id": "doc-2", "filename": "b.pdf", "page_count": 7},
    ]
    pool = make_mock_pool(rows)
    results = await list_documents("user-1", pool)
    assert len(results) == 2


async def test_list_documents_result_keys():
    rows = [{"document_id": "doc-1", "filename": "a.pdf", "page_count": 3}]
    pool = make_mock_pool(rows)
    results = await list_documents("user-1", pool)
    assert set(results[0].keys()) == {"document_id", "filename", "page_count"}


async def test_list_documents_empty_returns_empty_list():
    pool = make_mock_pool([])
    results = await list_documents("user-1", pool)
    assert results == []
