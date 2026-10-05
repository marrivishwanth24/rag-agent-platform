"""
Multi-agent orchestrator for the RAG pipeline — LangGraph + LangChain (LCEL).

RetrievalAgent is a LangGraph **state machine** with a self-correction branch:

    expand_queries -> parallel_search -> rerank -> grade_sufficiency
                                               ^                │
                                               │     insufficient & attempts < MAX
                                               └────────────────┘
                                                        │
                                          sufficient, or attempts >= MAX
                                                        │
                                                       END

If the reranked context looks insufficient to answer the question, the graph
widens the search (larger top_k) and retries — bounded by MAX_ATTEMPTS so a
stuck grading step can't loop forever or blow up latency/cost. This is a
small "self-correcting RAG" pattern: same escalate-and-retry shape as a
Next-Best-Action decision engine, scoped to retrieval. Each LLM step (query
expansion, sufficiency grading) uses LangChain's `with_structured_output`
instead of hand-parsed JSON, so a malformed response can't silently break
the pipeline.

The per-query-variant search fan-out is still a LangChain Runnable, run
concurrently via `.abatch()` — LangGraph owns the *decision* structure,
LCEL owns the *parallel execution* within a single node.

SynthesisAgent streams a grounded markdown answer via ChatAnthropic, then
emits [CITATIONS] and [DONE] SSE events.

run_pipeline() wires them together and is passed directly to FastAPI's
StreamingResponse. Public API (RetrievalAgent.retrieve, run_pipeline) is
unchanged — only the internals are LangGraph/LangChain-orchestrated.
"""

import json
from typing import AsyncGenerator, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.retrieval import vector_search, rerank

_haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=150)
_sonnet = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024, streaming=True)

MAX_RETRIEVAL_ATTEMPTS = 2  # initial pass + 1 widened retry — bounds latency/cost
MAX_SEARCH_TOP_K = 25       # cap on how far a retry can widen the candidate pool


def _as_text(content) -> str:
    """Normalize a LangChain message's .content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content) if content else ""


# ── Structured-output schemas for the two LLM decision steps ───────────────

class _QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="Exactly 2 alternative phrasings of the question, to improve retrieval recall"
    )


class _ContextGrade(BaseModel):
    sufficient: bool = Field(
        description="True if the context very likely contains enough information to answer the question"
    )
    reason: str = Field(description="One short sentence explaining the judgment")


_expander = _haiku.with_structured_output(_QueryExpansion)
_grader = _haiku.with_structured_output(_ContextGrade)


# ── Retrieval state machine (LangGraph) ─────────────────────────────────────

class RetrievalState(TypedDict, total=False):
    question: str
    document_ids: list[str]
    user_id: str
    pool: object
    top_k: int
    queries: list[str]
    candidates: list[dict]
    chunks: list[dict]
    attempts: int
    sufficient: bool


async def _search_one(args: tuple) -> list[dict]:
    query, document_ids, user_id, pool, limit = args
    return await vector_search(query, document_ids, user_id, pool, limit=limit)


_search_one_runnable = RunnableLambda(_search_one)


async def expand_queries_node(state: RetrievalState) -> dict:
    """Ask Claude Haiku for 2 alternative phrasings, via structured output."""
    question = state["question"]
    alternatives: list[str] = []
    try:
        result = await _expander.ainvoke([
            HumanMessage(content=(
                "Generate 2 alternative search queries for retrieving relevant "
                "document passages that answer this question.\n\n"
                f"Question: {question}"
            ))
        ])
        alternatives = [q for q in result.queries if isinstance(q, str)][:2]
    except Exception:
        pass  # fall back to just the original question
    return {"queries": [question] + alternatives, "attempts": 0}


async def parallel_search_node(state: RetrievalState) -> dict:
    """Fetch candidates for every query variant concurrently via Runnable.abatch.

    On a retry (attempts > 0), widen the candidate pool instead of just
    repeating the identical search — that's what makes the retry meaningful.
    """
    base_top_k = state.get("top_k", 5)
    attempts = state.get("attempts", 0)
    effective_top_k = min(base_top_k * 2 * (attempts + 1), MAX_SEARCH_TOP_K)

    batch_args = [
        (q, state["document_ids"], state["user_id"], state["pool"], effective_top_k)
        for q in state["queries"]
    ]
    candidate_lists = await _search_one_runnable.abatch(batch_args)

    seen: dict[str, dict] = {}
    for chunks in candidate_lists:
        for chunk in chunks:
            key = chunk["chunk_text"]
            if key not in seen or chunk["similarity"] > seen[key]["similarity"]:
                seen[key] = chunk

    return {"candidates": list(seen.values())}


async def rerank_node(state: RetrievalState) -> dict:
    """Single Voyage AI cross-encoder rerank pass over the deduplicated pool."""
    candidates = state["candidates"]
    if not candidates:
        return {"chunks": []}
    chunks = await rerank(state["question"], candidates, top_k=state.get("top_k", 5))
    return {"chunks": chunks}


async def grade_sufficiency_node(state: RetrievalState) -> dict:
    """Judge whether the reranked context likely answers the question."""
    attempts = state.get("attempts", 0) + 1
    chunks = state["chunks"]
    if not chunks:
        return {"sufficient": False, "attempts": attempts}

    preview = "\n\n".join(c["chunk_text"][:300] for c in chunks[:3])
    try:
        grade = await _grader.ainvoke([
            HumanMessage(content=(
                "Given this question and the retrieved context, judge whether the "
                "context likely contains enough information to answer the question.\n\n"
                f"Question: {state['question']}\n\nContext:\n{preview}"
            ))
        ])
        sufficient = bool(grade.sufficient)
    except Exception:
        sufficient = True  # fail-open: don't retry forever on a grading error

    return {"sufficient": sufficient, "attempts": attempts}


def _route_after_grade(state: RetrievalState) -> str:
    if state.get("sufficient") or state.get("attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "done"
    return "retry"


def _build_retrieval_graph():
    g = StateGraph(RetrievalState)
    g.add_node("expand_queries", expand_queries_node)
    g.add_node("parallel_search", parallel_search_node)
    g.add_node("rerank", rerank_node)
    g.add_node("grade_sufficiency", grade_sufficiency_node)

    g.set_entry_point("expand_queries")
    g.add_edge("expand_queries", "parallel_search")
    g.add_edge("parallel_search", "rerank")
    g.add_edge("rerank", "grade_sufficiency")
    g.add_conditional_edges(
        "grade_sufficiency",
        _route_after_grade,
        {"done": END, "retry": "parallel_search"},
    )
    return g.compile()


_retrieval_graph = _build_retrieval_graph()


class RetrievalAgent:
    """Query expansion + parallel retrieval + rerank + self-correction (LangGraph)."""

    async def retrieve(
        self,
        question: str,
        document_ids: list[str],
        user_id: str,
        pool,
        top_k: int = 5,
    ) -> list[dict]:
        result = await _retrieval_graph.ainvoke({
            "question": question,
            "document_ids": document_ids,
            "user_id": user_id,
            "pool": pool,
            "top_k": top_k,
        })
        return result["chunks"]


# ── Synthesis Agent ──────────────────────────────────────────────────────────

class SynthesisAgent:
    """Stream a grounded answer via ChatAnthropic and emit citation metadata."""

    _SYSTEM = (
        "You are a precise AI assistant that answers questions strictly based on "
        "provided document context.\n"
        "- Ground every claim in the supplied context\n"
        "- If context is insufficient, say so clearly rather than guessing\n"
        "- Use markdown: headings, bullet lists, bold for key terms\n"
    )

    async def stream(
        self,
        question: str,
        chunks: list[dict],
    ) -> AsyncGenerator[str, None]:
        formatted = "\n\n".join(
            f"[Source: {c['filename']}, Page {c.get('page_num', '?')}]\n{c['chunk_text']}"
            for c in chunks
        )
        user_msg = f"Context:\n{formatted}\n\n---\n\nQuestion: {question}"
        messages = [SystemMessage(content=self._SYSTEM), HumanMessage(content=user_msg)]

        async for chunk in _sonnet.astream(messages):
            text = _as_text(chunk.content)
            if text:
                yield f"data: {text}\n\n"

        # Deduplicate citations and emit as a single SSE event
        seen: dict[tuple, float] = {}
        for c in chunks:
            key = (c["filename"], c.get("page_num", 0))
            if key not in seen or c["similarity"] > seen[key]:
                seen[key] = c["similarity"]
        sources = [
            {"filename": fn, "page_num": pg, "similarity": round(sim, 2)}
            for (fn, pg), sim in sorted(seen.items(), key=lambda x: -x[1])
        ]
        yield f"data: [CITATIONS]{json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"


# ── Pipeline entry point ──────────────────────────────────────────────────────

async def run_pipeline(
    question: str,
    document_ids: list[str],
    user_id: str,
    pool,
) -> AsyncGenerator[str, None]:
    """
    Full two-agent pipeline:
      1. RetrievalAgent  — LangGraph state machine: expand -> search -> rerank
                            -> grade -> (retry widened search if insufficient)
      2. SynthesisAgent  — grounded streaming answer + citations via ChatAnthropic
    """
    chunks = await RetrievalAgent().retrieve(question, document_ids, user_id, pool)
    async for event in SynthesisAgent().stream(question, chunks):
        yield event
