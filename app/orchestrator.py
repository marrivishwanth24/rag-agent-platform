"""
Multi-agent orchestrator for the RAG pipeline — built with LangChain (LCEL).

Two specialized agents, composed as LangChain Runnables:

  RetrievalAgent  — an LCEL chain: Claude Haiku query expansion
                    -> parallel pgvector search (Runnable.abatch)
                    -> dedupe -> single Voyage AI rerank pass.

  SynthesisAgent  — streams a grounded markdown answer via ChatAnthropic
                    (Claude Sonnet), then emits [CITATIONS] and [DONE] SSE
                    events.

run_pipeline() wires them together and is passed directly to FastAPI's
StreamingResponse. Public API (RetrievalAgent.retrieve, run_pipeline) is
unchanged from the previous hand-rolled version — only the internals are
now LangChain-orchestrated.
"""

import json
from typing import AsyncGenerator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from app.retrieval import vector_search, rerank

_haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=150)
_sonnet = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024, streaming=True)


def _as_text(content) -> str:
    """Normalize a LangChain message's .content to plain text.

    ChatAnthropic returns a plain string for simple text responses, but the
    content field can be a list of content blocks in other configurations —
    handle both so this doesn't silently break on a library version bump.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content) if content else ""


# ── Retrieval Agent — an LCEL chain ─────────────────────────────────────────
#
# State is a dict threaded through each step (question/document_ids/user_id/
# pool/top_k in, enriched with queries -> candidates -> chunks as it flows
# through the chain). Each step is a plain async function wrapped in a
# RunnableLambda; `|` composes them into a single LangChain RunnableSequence.

async def _expand_queries(state: dict) -> dict:
    """Ask Claude Haiku for 2 alternative phrasings to improve recall."""
    question = state["question"]
    resp = await _haiku.ainvoke([
        HumanMessage(content=(
            "Generate 2 alternative search queries for retrieving relevant "
            "document passages that answer this question. "
            "Return ONLY a JSON array of 2 strings, nothing else.\n\n"
            f"Question: {question}"
        ))
    ])
    queries = [question]
    try:
        alternatives = json.loads(_as_text(resp.content))
        if isinstance(alternatives, list):
            queries += [q for q in alternatives if isinstance(q, str)][:2]
    except Exception:
        pass
    return {**state, "queries": queries}


async def _search_one(args: tuple) -> list[dict]:
    query, document_ids, user_id, pool, top_k = args
    return await vector_search(query, document_ids, user_id, pool, limit=top_k * 2)


_search_one_runnable = RunnableLambda(_search_one)


async def _parallel_search(state: dict) -> dict:
    """Fetch candidates for every query variant concurrently via Runnable.abatch."""
    top_k = state.get("top_k", 5)
    batch_args = [
        (q, state["document_ids"], state["user_id"], state["pool"], top_k)
        for q in state["queries"]
    ]
    candidate_lists = await _search_one_runnable.abatch(batch_args)

    # Deduplicate by chunk text, keep highest cosine similarity score
    seen: dict[str, dict] = {}
    for chunks in candidate_lists:
        for chunk in chunks:
            key = chunk["chunk_text"]
            if key not in seen or chunk["similarity"] > seen[key]["similarity"]:
                seen[key] = chunk

    return {**state, "candidates": list(seen.values())}


async def _rerank_step(state: dict) -> dict:
    """Single Voyage AI cross-encoder rerank pass over the deduplicated pool."""
    candidates = state["candidates"]
    if not candidates:
        return {**state, "chunks": []}
    chunks = await rerank(state["question"], candidates, top_k=state.get("top_k", 5))
    return {**state, "chunks": chunks}


_retrieval_chain = (
    RunnableLambda(_expand_queries)
    | RunnableLambda(_parallel_search)
    | RunnableLambda(_rerank_step)
)


class RetrievalAgent:
    """Query expansion + parallel retrieval + rerank, as a LangChain LCEL chain."""

    async def retrieve(
        self,
        question: str,
        document_ids: list[str],
        user_id: str,
        pool,
        top_k: int = 5,
    ) -> list[dict]:
        result = await _retrieval_chain.ainvoke({
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
    Full two-agent pipeline, orchestrated with LangChain:
      1. RetrievalAgent  — LCEL chain: query expansion -> parallel fetch -> rerank
      2. SynthesisAgent  — grounded streaming answer + citations via ChatAnthropic
    """
    chunks = await RetrievalAgent().retrieve(question, document_ids, user_id, pool)
    async for event in SynthesisAgent().stream(question, chunks):
        yield event
