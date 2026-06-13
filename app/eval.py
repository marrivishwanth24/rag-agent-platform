"""
Eval harness for the RAG pipeline.

Three metrics, all scored 0.0 – 1.0:

  faithfulness      — does the answer stay within the retrieved context?
                      (LLM-as-judge via Claude Haiku)

  answer_relevance  — does the answer actually address the question?
                      (LLM-as-judge via Claude Haiku)

  context_quality   — average reranker similarity of retrieved chunks;
                      proxy for retrieval precision (no extra API call)

Scores are averaged into an `overall` field.
"""

import asyncio
import json
import os

import anthropic

_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_JUDGE_SYSTEM = (
    "You are an objective evaluator of AI-generated answers. "
    "Score strictly on the stated criteria. "
    "Return ONLY valid JSON — no explanation outside the JSON object."
)


async def _judge(prompt: str) -> float:
    resp = await _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(resp.content[0].text)
        return max(0.0, min(1.0, float(data["score"])))
    except Exception:
        return 0.0


async def score_faithfulness(answer: str, chunks: list[dict]) -> float:
    """Does every claim in the answer appear in the retrieved context?"""
    context = "\n\n".join(c["chunk_text"] for c in chunks)
    return await _judge(
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Rate how faithfully the answer sticks to the context (0.0 = completely "
        "hallucinated, 1.0 = every claim is grounded in the context). "
        'Return: {"score": <float 0-1>, "reason": "<one sentence>"}'
    )


async def score_answer_relevance(question: str, answer: str) -> float:
    """Does the answer actually address the question that was asked?"""
    return await _judge(
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Rate how well the answer addresses the question (0.0 = completely "
        "off-topic, 1.0 = fully and directly answers it). "
        'Return: {"score": <float 0-1>, "reason": "<one sentence>"}'
    )


def score_context_quality(chunks: list[dict]) -> float:
    """Average reranker similarity — how relevant were the retrieved chunks?"""
    if not chunks:
        return 0.0
    return round(sum(c["similarity"] for c in chunks) / len(chunks), 3)


async def evaluate(
    question: str,
    answer: str,
    chunks: list[dict],
) -> dict:
    """Run all three metrics (faithfulness + relevance in parallel) and return scorecard."""
    faithfulness, relevance = await asyncio.gather(
        score_faithfulness(answer, chunks),
        score_answer_relevance(question, answer),
    )
    context_quality = score_context_quality(chunks)
    overall = round((faithfulness + relevance + context_quality) / 3, 3)

    return {
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(relevance, 3),
        "context_quality": context_quality,
        "overall": overall,
    }
