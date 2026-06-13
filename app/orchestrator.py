"""
Multi-agent orchestrator for the RAG pipeline.

Two specialized agents:

  RetrievalAgent  — asks Claude to rephrase the user question into 2-3
                    alternative search queries (query expansion), runs them
                    against pgvector in parallel, deduplicates the results,
                    then reranks the combined pool once with Voyage AI.

  SynthesisAgent  — streams a grounded markdown answer using the retrieved
                    context, then emits [CITATIONS] and [DONE] SSE events.

run_pipeline() wires them together and is passed directly to FastAPI's
StreamingResponse.
"""

import asyncio
import json
import os
from typing import AsyncGenerator

import anthropic
from app.retrieval import vector_search, rerank

_anthropic = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Retrieval Agent ───────────────────────────────────────────────────────────

class RetrievalAgent:
    """Query expansion + parallel retrieval + single combined rerank pass."""

    async def _expand_queries(self, question: str) -> list[str]:
        """Generate 2 alternative phrasings to improve recall."""
        resp = await _anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Generate 2 alternative search queries for retrieving relevant "
                    "document passages that answer this question. "
                    "Return ONLY a JSON array of 2 strings, nothing else.\n\n"
                    f"Question: {question}"
                ),
            }],
        )
        try:
            alternatives = json.loads(resp.content[0].text)
            if isinstance(alternatives, list):
                return [question] + [q for q in alternatives if isinstance(q, str)][:2]
        except Exception:
            pass
        return [question]

    async def retrieve(
        self,
        question: str,
        document_ids: list[str],
        user_id: str,
        pool,
        top_k: int = 5,
    ) -> list[dict]:
        queries = await self._expand_queries(question)

        # Fetch candidates for every query variant in parallel
        candidate_lists = await asyncio.gather(*[
            vector_search(q, document_ids, user_id, pool, limit=top_k * 2)
            for q in queries
        ])

        # Deduplicate by chunk text, keep highest cosine similarity score
        seen: dict[str, dict] = {}
        for chunks in candidate_lists:
            for chunk in chunks:
                key = chunk["chunk_text"]
                if key not in seen or chunk["similarity"] > seen[key]["similarity"]:
                    seen[key] = chunk

        candidates = list(seen.values())
        if not candidates:
            return []

        # Single rerank pass over the combined, deduplicated pool
        return await rerank(question, candidates, top_k=top_k)


# ── Synthesis Agent ───────────────────────────────────────────────────────────

class SynthesisAgent:
    """Stream a grounded answer and emit citation metadata at the end."""

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

        async with _anthropic.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            async for text in stream.text_stream:
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
      1. RetrievalAgent  — query expansion → parallel fetch → rerank
      2. SynthesisAgent  — grounded streaming answer + citations
    """
    chunks = await RetrievalAgent().retrieve(question, document_ids, user_id, pool)
    async for event in SynthesisAgent().stream(question, chunks):
        yield event
