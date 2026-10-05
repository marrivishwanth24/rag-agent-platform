# RAG Agent Platform

An end-to-end **AI-powered document Q&A system** with LangChain-orchestrated multi-agent retrieval, semantic reranking, and an automated eval harness. Upload any PDF, ask questions — the system retrieves the most relevant content and streams a grounded, accurate answer in real time.

**Live Demo:** https://rag-frontend-topaz.vercel.app

---

## Features

- **Multi-agent retrieval** — RetrievalAgent is a LangGraph state machine that expands queries into 3 variants, runs parallel vector searches via `Runnable.abatch`, deduplicates, reranks with a Voyage AI cross-encoder, and self-corrects — grading context sufficiency and widening the search on a retry if the first pass looks weak
- **Two-stage retrieval** — fast ANN vector search (pgvector IVFFlat) followed by Voyage AI `rerank-2` for precision
- **Real-time streaming** — answers stream token-by-token via Server-Sent Events
- **Source citations** — every response shows which document and page each claim came from, with a match confidence score
- **Eval harness** — LLM-as-judge scoring for faithfulness, answer relevance, and context quality
- **Session-based privacy** — each browser gets a private UUID; your documents are never visible to other users (no login required)
- **Multi-PDF support** — upload multiple documents, query across all of them simultaneously
- **Markdown rendering** — structured responses with headings, bullet lists, code blocks, and bold text
- **Drag-and-drop upload** — drag PDFs anywhere onto the page

---

## Architecture

```
User (Browser)
      │
      ▼
React Frontend (Vercel)
      │  X-Session-ID header (browser-local UUID)
      ▼
FastAPI Backend (Railway)
      │
      ├── POST /upload
      │     PDF → PyPDF2 extraction → 500-word chunks (50-word overlap)
      │     → Voyage AI voyage-3.5 embedding → pgvector storage
      │
      └── POST /query  ──►  Multi-Agent Orchestrator (LangGraph + LangChain)
                                │
                                ├── RetrievalAgent — a LangGraph state machine
                                │     ├── Claude Haiku (structured output): query expansion (3 variants)
                                │     ├── pgvector: parallel ANN search (Runnable.abatch, 3 × top-10)
                                │     ├── deduplicate by chunk text
                                │     ├── Voyage rerank-2: single combined rerank → top 5
                                │     └── Claude Haiku (structured output): grade context sufficiency
                                │           → insufficient? widen search + retry (bounded, max 1 retry)
                                │           → sufficient (or retries exhausted)? proceed
                                │
                                └── SynthesisAgent
                                      ├── Claude Sonnet 4.6 (ChatAnthropic): grounded streaming answer
                                      └── SSE: token stream + [CITATIONS] event
      │
      ▼
PostgreSQL + pgvector (Railway)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, TypeScript, react-markdown, SSE streaming |
| **Backend** | Python 3.11, FastAPI, asyncpg, async/await throughout |
| **Multi-Agent** | LangGraph — RetrievalAgent state machine (self-correcting retrieval loop) + SynthesisAgent, via `langchain-anthropic` |
| **LLM** | Claude Sonnet 4.6 (synthesis) · Claude Haiku 4.5 (query expansion, sufficiency grading, eval judge) |
| **Structured Output** | `with_structured_output` (Pydantic) for query expansion and context-sufficiency grading |
| **Embeddings** | Voyage AI `voyage-3.5` — 1024 dimensions, document/query input types |
| **Reranking** | Voyage AI `rerank-2` cross-encoder |
| **Vector DB** | PostgreSQL + pgvector, IVFFlat cosine index |
| **Privacy** | Browser-local session UUID via `localStorage` — no accounts needed |
| **Deployment** | Vercel (frontend) · Railway Nixpacks (backend + DB) |
| **Testing** | Pytest — 30 tests covering ingestion, retrieval, orchestration (incl. the retry/escalation branch), and API |

---

## Project Structure

```
rag-agent-platform/
├── app/
│   ├── main.py          ← FastAPI app — routes, CORS, lifespan pool, session ID
│   ├── ingestion.py     ← PDF extraction, chunking, Voyage AI embedding, pgvector insert
│   ├── retrieval.py     ← vector_search(), rerank() — used by orchestrator
│   ├── orchestrator.py  ← RetrievalAgent — LangGraph state machine (query expansion
│   │                       + parallel retrieval via Runnable.abatch + rerank +
│   │                       self-correcting sufficiency-grading retry loop)
│   │                       SynthesisAgent (ChatAnthropic streaming + citations)
│   │                       run_pipeline() — entry point for /query
│   ├── eval.py          ← LLM-as-judge: faithfulness, answer_relevance, context_quality
│   ├── agent.py         ← Legacy standalone Claude streaming helper — not imported
│   │                       by the current orchestrator; kept for reference
│   └── auth.py          ← JWT utilities (available but not required on endpoints)
├── frontend/
│   └── src/
│       ├── App.tsx      ← Chat UI, eval panel, drag-and-drop, session management
│       └── App.css      ← Dark theme, markdown styles, score bars, mobile layout
├── tests/
│   ├── test_api.py          ← API endpoint tests (health, upload, query, documents)
│   ├── test_ingestion.py    ← Chunking pipeline tests
│   ├── test_retrieval.py    ← Retrieval tests with mocked asyncpg + embeddings
│   └── test_orchestrator.py ← LCEL chain tests: dedupe, fallback, SSE event format
├── schema.sql           ← pgvector extension, document_chunks table, IVFFlat index
├── railway.json         ← Nixpacks builder config + start command
├── vercel.json          ← Frontend build config (cd frontend && npm run build)
├── requirements.txt
└── .gitignore
```

---

## How It Works

### Document Ingestion

```
PDF uploaded
      ↓
PyPDF2 extracts text page by page
      ↓
Text split into 500-word chunks with 50-word overlap
(overlap ensures sentences at chunk boundaries retain context)
      ↓
Each chunk sent to Voyage AI voyage-3.5 (input_type="document")
→ returns 1024-dimensional semantic vector
      ↓
chunk_text + embedding + session_id stored in PostgreSQL via pgvector
```

### Multi-Agent Query Pipeline

RetrievalAgent is a **LangGraph** state machine — each step is a node, wired with `add_edge`/`add_conditional_edges`, invoked via `.ainvoke()`. The parallel search fan-out within the `parallel_search` node is a LangChain `Runnable` run via `.abatch()`. Both LLM decision steps (query expansion, sufficiency grading) use `with_structured_output` against a Pydantic schema instead of hand-parsed JSON:

```
User question
      ↓
RetrievalAgent  (LangGraph state machine)
  ├── expand_queries     — Haiku (structured output) generates 2 alternative phrasings
  ├── parallel_search    — Runnable.abatch() fans the queries out concurrently to
  │                         pgvector; widens top_k automatically on a retry
  ├── dedupe              — by chunk text, keep highest cosine similarity
  ├── rerank              — Voyage rerank-2 cross-encoder on the combined pool → top 5
  └── grade_sufficiency   — Haiku (structured output) judges: does this context
        │                   likely answer the question?
        ├── insufficient & attempts < 2 → loop back to parallel_search, widened
        └── sufficient, or attempts exhausted → proceed with best-effort context
      ↓
SynthesisAgent
  ├── ChatAnthropic (Claude Sonnet 4.6) streams a grounded markdown answer via SSE
  └── Emits [CITATIONS] event with filename, page, and match score per source
      ↓
Frontend renders streaming tokens + source citation chips
```

This self-correction branch is deliberately small — one bounded retry, not an open-ended agent loop — but it's the same escalate-and-retry shape as a decision engine that widens its search before giving up, rather than a fixed linear pipeline.

### Eval Harness

```
POST /eval  { question, document_ids }
      ↓
RetrievalAgent retrieves context (same pipeline as /query)
      ↓
Claude Sonnet generates non-streaming answer
      ↓
In parallel:
  ├── Claude Haiku judges faithfulness (0–1): does every claim exist in the context?
  └── Claude Haiku judges answer_relevance (0–1): does the answer address the question?
      ↓
context_quality = avg reranker similarity (no extra API call)
      ↓
Returns { faithfulness, answer_relevance, context_quality, overall }
```

### Why Cosine Similarity + Reranking?

Vector search (bi-encoder) embeds the query and each chunk **independently** and measures geometric closeness — fast, but imprecise. The reranker (cross-encoder) sees the query and each chunk **together**, allowing it to judge actual relevance — slower but much more accurate. The two-stage approach uses the fast ANN index for candidate retrieval and the accurate reranker for final selection.

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Anthropic API key
- Voyage AI API key

### Local Setup

```bash
git clone https://github.com/marrivishwanth24/rag-agent-platform
cd rag-agent-platform

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
DATABASE_URL=postgresql://user:password@host:port/dbname
SECRET_KEY=your-secret-key
```

Set up the database:

```bash
psql $DATABASE_URL < schema.sql
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm start
```

---

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF — session-scoped ingestion |
| `POST` | `/query` | Stream an answer via SSE (multi-agent pipeline) |
| `POST` | `/eval` | Run quality evaluation — returns faithfulness, relevance, context scores |
| `GET` | `/documents` | List documents for the current session |

All endpoints read the `X-Session-ID` header to scope data per browser session.

---

## Key Engineering Decisions

**Why two-stage retrieval (vector search + reranker)?**
ANN vector search is O(log n) and very fast but uses bi-encoders that can miss nuanced relevance. The Voyage AI `rerank-2` cross-encoder sees query and chunk together and scores relevance much more accurately. Fetching 3× more candidates then reranking gives the recall of a wide search with the precision of a careful one.

**Why query expansion?**
A single embedding of the user's question might miss relevant chunks phrased differently. Generating 2 alternative phrasings with Claude Haiku and searching all 3 in parallel dramatically increases recall with minimal latency cost (the searches run concurrently).

**Why LangGraph for the retrieval agent, and not a linear LCEL chain or raw SDK calls?**
The retrieval pipeline isn't purely linear — once a sufficiency-grading step exists, there's a real branch: retry with a widened search, or proceed. That's exactly what LangGraph's explicit state graph (nodes + conditional edges) models, and a plain LCEL `|` chain can't express a loop. The pipeline started as an LCEL chain (still true for the `parallel_search` node's fan-out, which uses `Runnable.abatch()`); LangGraph was introduced specifically when a real decision point — "is this context good enough?" — was added. Each node is an independent, testable async function; `tests/test_orchestrator.py` mocks the LLM and retrieval boundaries to verify the graph's wiring, the retry/escalation branch, its bounded termination, and the exact SSE output, without hitting real APIs. The vector store itself stays a direct `asyncpg` + pgvector query (not `langchain_postgres.PGVector`) since the hand-tuned SQL and session/document filtering were already working well — LangChain/LangGraph's value here is in the orchestration and decision layer, not replacing a working data layer.

**Why LLM-as-judge for evaluation?**
Reference-free evaluation — no ground-truth answers needed. Claude Haiku reads the question, answer, and retrieved context and scores faithfulness and relevance independently. This mirrors the RAGAS framework approach and is cheap enough (Haiku) to run on every eval request.

**Why 500-word chunks with 50-word overlap?**
Large chunks lose precision — too much irrelevant content surrounds the answer. Small chunks lose context — sentences get split mid-thought. 500 words with 50-word overlap balances precision and context preservation at the boundary.

**Why session UUIDs instead of accounts?**
For a public demo tool, login friction kills adoption. A UUID stored in `localStorage` gives each visitor a private, isolated document space without any signup. The UUID is validated on every request — no account = no attack surface.

---

## Author

**Vishwanth Marri**
- Email: marrivishwanth24@gmail.com
- LinkedIn: linkedin.com/in/vishwanthmarri/
- GitHub: github.com/marrivishwanth24
- Live Demo: https://rag-frontend-topaz.vercel.app
