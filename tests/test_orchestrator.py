"""Tests for the LangGraph-orchestrated RetrievalAgent / SynthesisAgent / run_pipeline.

Mocks at the module boundary (app.orchestrator._expander / _grader / _sonnet /
vector_search / rerank) so these run without real Anthropic/Voyage calls or a
database, and verify: dedupe + rerank wiring, the self-correction retry/escalate
branch, the JSON-expansion fallback, and the exact SSE output format.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.orchestrator import RetrievalAgent, SynthesisAgent, run_pipeline

CHUNK_A = {"chunk_text": "alpha", "page_num": 1, "filename": "doc.pdf", "similarity": 0.9}
CHUNK_B = {"chunk_text": "beta", "page_num": 2, "filename": "doc.pdf", "similarity": 0.8}


def _expansion(queries):
    return SimpleNamespace(queries=queries)


def _grade(sufficient, reason="because"):
    return SimpleNamespace(sufficient=sufficient, reason=reason)


async def _fake_stream(texts):
    for t in texts:
        yield SimpleNamespace(content=t)


@pytest.mark.asyncio
async def test_retrieval_agent_dedupes_and_reranks():
    """Two query variants return overlapping chunks; sufficient on first grade (no retry)."""
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(
             side_effect=[[CHUNK_A], [CHUNK_A, CHUNK_B]]
         )) as mock_search, \
         patch("app.orchestrator.rerank", new=AsyncMock(
             return_value=[CHUNK_B, CHUNK_A]
         )) as mock_rerank:
        mock_expander.ainvoke = AsyncMock(return_value=_expansion(["alt query"]))
        mock_grader.ainvoke = AsyncMock(return_value=_grade(True))

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert chunks == [CHUNK_B, CHUNK_A]
    assert mock_search.call_count == 2  # original question + 1 expanded query
    mock_rerank.assert_called_once()
    deduped_pool = mock_rerank.call_args.args[1]
    assert {c["chunk_text"] for c in deduped_pool} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_retrieval_agent_falls_back_on_expansion_error():
    """If structured-output expansion raises, fall back to the original question only."""
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])) as mock_search, \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])):
        mock_expander.ainvoke = AsyncMock(side_effect=RuntimeError("bad response"))
        mock_grader.ainvoke = AsyncMock(return_value=_grade(True))

        await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert mock_search.call_count == 1  # only the original question, no alternatives


@pytest.mark.asyncio
async def test_retrieval_agent_empty_candidates_skips_rerank_but_still_grades():
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[])), \
         patch("app.orchestrator.rerank", new=AsyncMock()) as mock_rerank:
        mock_expander.ainvoke = AsyncMock(return_value=_expansion([]))
        mock_grader.ainvoke = AsyncMock(return_value=_grade(True))

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    assert chunks == []
    mock_rerank.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_agent_retries_with_widened_search_when_insufficient():
    """Core self-correction test: first pass graded insufficient -> widened retry -> sufficient."""
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])) as mock_search, \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])) as mock_rerank:
        mock_expander.ainvoke = AsyncMock(return_value=_expansion([]))
        # First grade: insufficient -> triggers retry. Second grade: sufficient -> stop.
        mock_grader.ainvoke = AsyncMock(side_effect=[_grade(False), _grade(True)])

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object(), top_k=5)

    assert chunks == [CHUNK_A]
    assert mock_grader.ainvoke.call_count == 2          # graded twice: insufficient, then sufficient
    assert mock_search.call_count == 2                  # searched again on retry (1 query x 2 rounds)
    assert mock_rerank.call_count == 2                  # reranked again after the widened search
    # limit passed to vector_search widens on the second (retry) call
    first_call_limit = mock_search.call_args_list[0].kwargs["limit"]
    second_call_limit = mock_search.call_args_list[1].kwargs["limit"]
    assert second_call_limit > first_call_limit


@pytest.mark.asyncio
async def test_retrieval_agent_stops_retrying_after_max_attempts():
    """If grading always says insufficient, the graph still terminates (bounded retries)."""
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])), \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])):
        mock_expander.ainvoke = AsyncMock(return_value=_expansion([]))
        mock_grader.ainvoke = AsyncMock(return_value=_grade(False))  # always insufficient

        chunks = await RetrievalAgent().retrieve("question", [], "user-1", pool=object())

    # Must terminate (not hang/loop forever) and still return best-effort chunks.
    assert chunks == [CHUNK_A]
    assert mock_grader.ainvoke.call_count == 2  # MAX_RETRIEVAL_ATTEMPTS == 2


@pytest.mark.asyncio
async def test_synthesis_agent_streams_tokens_then_citations_then_done():
    with patch("app.orchestrator._sonnet") as mock_sonnet, \
         patch("app.orchestrator._maybe_call_tools", new=AsyncMock(return_value=None)):
        mock_sonnet.astream = lambda messages: _fake_stream(["Hello", " world"])

        events = [e async for e in SynthesisAgent().stream(
            "q?", [CHUNK_A, CHUNK_B], "user-1", pool=object()
        )]

    assert events[0] == "data: Hello\n\n"
    assert events[1] == "data:  world\n\n"
    assert events[2].startswith("data: [CITATIONS]")
    citations = json.loads(events[2].removeprefix("data: [CITATIONS]").strip())
    assert {c["filename"] for c in citations} == {"doc.pdf"}
    assert events[3] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_run_pipeline_wires_retrieval_into_synthesis():
    with patch("app.orchestrator._expander") as mock_expander, \
         patch("app.orchestrator._grader") as mock_grader, \
         patch("app.orchestrator.vector_search", new=AsyncMock(return_value=[CHUNK_A])), \
         patch("app.orchestrator.rerank", new=AsyncMock(return_value=[CHUNK_A])), \
         patch("app.orchestrator._sonnet") as mock_sonnet, \
         patch("app.orchestrator._maybe_call_tools", new=AsyncMock(return_value=None)):
        mock_expander.ainvoke = AsyncMock(return_value=_expansion([]))
        mock_grader.ainvoke = AsyncMock(return_value=_grade(True))
        mock_sonnet.astream = lambda messages: _fake_stream(["answer"])

        events = [e async for e in run_pipeline("q?", [], "user-1", pool=object())]

    assert events[0] == "data: answer\n\n"
    assert events[-1] == "data: [DONE]\n\n"


# ── Agentic tool calling ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_call_tools_returns_none_when_model_declines():
    """Common case: the model answers directly, no tool call requested."""
    from app.orchestrator import _maybe_call_tools

    with patch("app.orchestrator._tool_router") as mock_router, \
         patch("app.orchestrator.list_documents", new=AsyncMock()) as mock_list:
        mock_router.ainvoke = AsyncMock(return_value=SimpleNamespace(tool_calls=[]))

        result = await _maybe_call_tools("What does the contract say about liability?", "user-1", pool=object())

    assert result is None
    mock_list.assert_not_called()  # tool never executed if the model didn't request it


@pytest.mark.asyncio
async def test_maybe_call_tools_executes_tool_with_server_side_session_context():
    """When the model requests the tool, the server executes it with the real
    user_id/pool — not anything the model supplied (the schema takes no args)."""
    from app.orchestrator import _maybe_call_tools

    real_user_id, real_pool = "user-42", object()
    with patch("app.orchestrator._tool_router") as mock_router, \
         patch("app.orchestrator.list_documents", new=AsyncMock(
             return_value=[{"filename": "handbook.pdf"}, {"filename": "policy.pdf"}]
         )) as mock_list:
        mock_router.ainvoke = AsyncMock(
            return_value=SimpleNamespace(tool_calls=[{"name": "list_documents_tool", "args": {}}])
        )

        result = await _maybe_call_tools("What documents do I have?", real_user_id, real_pool)

    mock_list.assert_called_once_with(real_user_id, real_pool)
    assert "handbook.pdf" in result and "policy.pdf" in result


@pytest.mark.asyncio
async def test_synthesis_agent_folds_tool_result_into_context():
    with patch("app.orchestrator._sonnet") as mock_sonnet, \
         patch("app.orchestrator._maybe_call_tools", new=AsyncMock(
             return_value="[Tool: list_documents] Available documents: a.pdf, b.pdf"
         )):
        captured = {}

        def _capture_and_stream(messages):
            captured["messages"] = messages
            return _fake_stream(["ok"])

        mock_sonnet.astream = _capture_and_stream

        _ = [e async for e in SynthesisAgent().stream("what do I have?", [], "user-1", pool=object())]

    user_message_text = captured["messages"][1].content
    assert "Available documents: a.pdf, b.pdf" in user_message_text
