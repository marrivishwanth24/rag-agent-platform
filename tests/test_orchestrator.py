"""Tests for the LangChain-orchestrated RetrievalAgent / SynthesisAgent / run_pipeline.

Mocks at the module boundary (app.orchestrator._haiku / _sonnet / vector_search /
rerank) so these run without real Anthropic/Voyage calls or a database, and verify
the LCEL chain's wiring (dedupe, fallback, SSE format) matches the pre-LangChain
behavior exactly.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.orchestrator import RetrievalAgent, SynthesisAgent, run_pipeline

CHUNK_A = {"chunk_text": "alpha", "page_num": 1, "filename": "doc.pdf", "similarity": 0.9}
CHUNK_B = {"chunk_text": "beta", "page_num": 2, "filename": "doc.pdf", "similarity": 0.8}


def _msg(content):
    return SimpleNamespace(content=content)


async def _fake_stream(texts):
    for t in texts:
        yield _msg(t)


@pytest.mark.asyncio
async def test_retrieval_agent_dedupes_and_reranks():
    """Two query variants return overlapping chunks; verify dedupe + rerank wiring."""
    with patch("app.orchestrator._haiku") as mock_haiku, \
         patch("app.orchestrator.vector_search", new=AsyncMock(
             side_effect=[[CHUNK_A], [CHUNK_A, CHUNK_B]]
         )) as mock_search, \
         patch("app.orchestrator.rerank", new=AsyncMock(
             return_value=[CHUNK_B, CHUNK_A]
         )) as mock_rerank:
        mock_haiku.ainvoke = AsyncMock(return_value=_msg(json.dumps(["alt query"])))

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert chunks == [CHUNK_B, CHUNK_A]
    assert mock_search.call_count == 2  # original question + 1 expanded query
    mock_rerank.assert_called_once()
    deduped_pool = mock_rerank.call_args.args[1]
    assert {c["chunk_text"] for c in deduped_pool} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_retrieval_agent_falls_back_on_bad_expansion_json():
    """If Haiku doesn't return valid JSON, fall back to the original question only."""
    with patch("app.orchestrator._haiku") as mock_haiku, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])) as mock_search, \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])):
        mock_haiku.ainvoke = AsyncMock(return_value=_msg("not json"))

        await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert mock_search.call_count == 1  # only the original question, no alternatives


@pytest.mark.asyncio
async def test_retrieval_agent_empty_candidates_skips_rerank():
    with patch("app.orchestrator._haiku") as mock_haiku, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.rerank", new=AsyncMock()) as mock_rerank:
        mock_haiku.ainvoke = AsyncMock(return_value=_msg(json.dumps([])))

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert chunks == []
    mock_rerank.assert_not_called()


@pytest.mark.asyncio
async def test_synthesis_agent_streams_tokens_then_citations_then_done():
    with patch("app.orchestrator._sonnet") as mock_sonnet:
        mock_sonnet.astream = lambda messages: _fake_stream(["Hello", " world"])

        events = [e async for e in SynthesisAgent().stream("q?", [CHUNK_A, CHUNK_B])]

    assert events[0] == "data: Hello\n\n"
    assert events[1] == "data:  world\n\n"
    assert events[2].startswith("data: [CITATIONS]")
    citations = json.loads(events[2].removeprefix("data: [CITATIONS]").strip())
    assert {c["filename"] for c in citations} == {"doc.pdf"}
    assert events[3] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_run_pipeline_wires_retrieval_into_synthesis():
    with patch("app.orchestrator._haiku") as mock_haiku, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])), \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])), \
         patch("app.orchestrator._sonnet") as mock_sonnet:
        mock_haiku.ainvoke = AsyncMock(return_value=_msg(json.dumps([])))
        mock_sonnet.astream = lambda messages: _fake_stream(["answer"])

        events = [e async for e in run_pipeline("q?", [], "user-1", pool=object())]

    assert events[0] == "data: answer\n\n"
    assert events[-1] == "data: [DONE]\n\n"
