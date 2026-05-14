import anthropic
import os
from typing import AsyncGenerator

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document context.

When answering:
- Base your response strictly on the provided context
- Cite the source document when referencing specific information
- If the context doesn't contain the answer, say so clearly
- Be concise and precise
"""


async def stream_agent_response(
    question: str,
    context_chunks: list[dict]
) -> AsyncGenerator[str, None]:
    """
    Stream a response from Claude given a question and retrieved context chunks.
    Uses the Claude API with streaming for real-time token delivery.
    """
    # Format context with source citations
    formatted_context = "\n\n".join([
        f"[Source: {chunk['filename']}, Page {chunk.get('page', 'N/A')}]\n{chunk['text']}"
        for chunk in context_chunks
    ])

    user_message = f"""Context from uploaded documents:

{formatted_context}

---

Question: {question}

Please answer based on the context above."""

    # Stream response from Claude
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {text}\n\n"

    yield "data: [DONE]\n\n"


async def run_agent_with_tools(
    question: str,
    context_chunks: list[dict],
    available_tools: list[dict] = None
) -> str:
    """
    Run a full agent loop with tool use for more complex queries.
    Supports multi-turn reasoning and tool invocation.
    """
    formatted_context = "\n\n".join([
        f"[{chunk['filename']}]: {chunk['text']}"
        for chunk in context_chunks
    ])

    messages = [
        {
            "role": "user",
            "content": f"Context:\n{formatted_context}\n\nQuestion: {question}"
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=available_tools or []
    )

    return response.content[0].text
