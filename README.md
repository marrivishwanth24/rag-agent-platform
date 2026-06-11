# RAG Agent Platform

An end-to-end **AI-powered document Q&A system** built with FastAPI, pgvector, Voyage AI embeddings, and Claude API. Upload any PDF and ask questions — the system retrieves semantically relevant content and generates grounded, accurate answers in real time.

**🚀 Live Demo:** https://rag-frontend-topaz.vercel.app

---

## Features

- **Real semantic search** — Voyage AI `voyage-3.5` embeddings with 1024-dimensional vectors stored in pgvector
- **Retrieval-Augmented Generation** — Claude API generates answers grounded in retrieved document context
- **Real-time streaming** — responses stream token by token via server-sent events
- **Document ingestion pipeline** — PDF upload → text extraction → chunking with overlap → embedding → indexed storage
- **Cosine similarity search** — IVFFlat index on pgvector for fast sub-200ms retrieval
- **OAuth2/JWT authentication** — secure user sessions with RBAC access control
- **Automated test suite** — 16 Pytest tests covering ingestion, retrieval, and API endpoints
- **Full CI/CD** — GitHub Actions → Docker → Railway deployment

---

## Architecture

```
User (Browser)
      ↓
React Frontend (Vercel)
      ↓
FastAPI Backend (Railway)
      ↓
┌─────────────────────────────────┐
│  POST /upload                   │
│  PDF → extract → chunk → embed  │
│  → store in pgvector            │
├─────────────────────────────────┤
│  POST /query                    │
│  question → embed → cosine      │
│  similarity search → top 5      │
│  chunks → Claude API → stream   │
└─────────────────────────────────┘
      ↓
PostgreSQL + pgvector (Railway)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React, TypeScript, real-time SSE streaming |
| **Backend** | Python, FastAPI, async/await |
| **AI Agent** | Claude API (Anthropic), LangChain |
| **Embeddings** | Voyage AI `voyage-3.5` — 1024 dimensions |
| **Vector DB** | PostgreSQL + pgvector, IVFFlat cosine index |
| **Auth** | OAuth2, JWT, RBAC |
| **Cloud** | AWS EC2, Railway, Vercel |
| **DevOps** | Docker, GitHub Actions CI/CD |
| **Testing** | Pytest — 16 tests |

---

## Project Structure

```
rag-agent-platform/
├── app/
│   ├── main.py          ← FastAPI server — routes, CORS, endpoints
│   ├── ingestion.py     ← PDF extraction, chunking, Voyage AI embedding, pgvector storage
│   ├── retrieval.py     ← Query embedding, cosine similarity search, context retrieval
│   ├── agent.py         ← Claude API integration, streaming response generation
│   ├── auth.py          ← OAuth2/JWT authentication, RBAC
│   └── models.py        ← PostgreSQL schema definitions
├── frontend/
│   ├── src/
│   │   └── App.tsx      ← React frontend with real-time streaming UI
│   └── public/
│       └── index.html
├── tests/
│   ├── test_api.py          ← 5 API endpoint tests
│   ├── test_ingestion.py    ← 6 chunking and pipeline tests
│   └── test_retrieval.py    ← 5 embedding and retrieval tests
├── schema.sql           ← PostgreSQL schema with pgvector setup
├── pytest.ini           ← Pytest configuration
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Container configuration
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
(overlap ensures sentences at boundaries don't lose meaning)
      ↓
Each chunk sent to Voyage AI voyage-3.5
→ returns 1024-dimensional semantic vector
      ↓
chunk_text + embedding stored in PostgreSQL via pgvector
```

### Query and Retrieval

```
User asks a question
      ↓
Question embedded with same voyage-3.5 model
(input_type="query" — Voyage optimizes differently for queries)
      ↓
pgvector runs cosine similarity search:
SELECT ... ORDER BY embedding <=> query_vector LIMIT 5
      ↓
Top 5 most semantically similar chunks returned
      ↓
Chunks passed to Claude API as context
      ↓
Claude generates grounded answer
      ↓
Response streamed token by token via SSE
```

### Why Cosine Similarity?

Cosine similarity measures the angle between two vectors. Vectors that point in the same direction (similarity = 1) represent semantically similar text — even if they use different words. This enables true semantic search rather than keyword matching.

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

Create a `.env` file (never commit this):

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

Run the server:

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the API documentation.

---

## Running Tests

```bash
cd rag-agent-platform
source venv/bin/activate
pytest tests/ -v
```

Expected output:

```
tests/test_api.py::test_health_returns_200 PASSED
tests/test_api.py::test_health_returns_status_field PASSED
tests/test_api.py::test_upload_no_file_returns_422 PASSED
tests/test_api.py::test_query_no_body_returns_422 PASSED
tests/test_api.py::test_documents_endpoint_returns_200 PASSED
tests/test_ingestion.py::test_chunk_text_basic PASSED
tests/test_ingestion.py::test_chunk_text_short_input PASSED
tests/test_ingestion.py::test_chunk_text_empty_input PASSED
tests/test_ingestion.py::test_chunk_text_no_empty_chunks PASSED
tests/test_ingestion.py::test_chunk_text_preserves_content PASSED
tests/test_ingestion.py::test_chunk_text_overlap PASSED
tests/test_retrieval.py::test_retrieval_result_has_required_keys PASSED
tests/test_retrieval.py::test_similarity_score_in_valid_range PASSED
tests/test_retrieval.py::test_embedding_dimensions PASSED
tests/test_retrieval.py::test_embedding_values_are_floats PASSED
tests/test_retrieval.py::test_chunk_text_key_name PASSED

16 passed in 1.79s
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF for ingestion |
| `POST` | `/query` | Ask a question about uploaded documents |
| `GET` | `/documents` | List all uploaded documents |

---

## Deployment

### Backend — Railway

```bash
railway login
railway up
```

Set environment variables in Railway dashboard:
- `ANTHROPIC_API_KEY`
- `VOYAGE_API_KEY`
- `DATABASE_URL` (use Railway's `${{Postgres.DATABASE_URL}}` reference)
- `SECRET_KEY`

### Frontend — Vercel

```bash
cd frontend
CI=false npm run build
vercel --prod
```

---

## Key Engineering Decisions

**Why Voyage AI over OpenAI embeddings?**
Anthropic recommends Voyage AI as the embedding partner for Claude-based RAG systems. `voyage-3.5` is optimized for retrieval tasks and uses different `input_type` parameters for documents vs queries — improving retrieval quality.

**Why 500-word chunks with 50-word overlap?**
Large chunks lose precision — the retrieved chunk contains too much irrelevant content alongside the answer. Small chunks lose context — sentences get split mid-thought. 500 words with 50-word overlap balances precision and context preservation.

**Why IVFFlat index?**
Without an index, pgvector scans every row for every query — O(n) at scale. IVFFlat partitions vectors into clusters and searches only relevant clusters — making retrieval fast even with thousands of stored chunks.

---

## Author

**Vishwanth Marri**
- Email: marrivishwanth24@gmail.com
- LinkedIn: linkedin.com/in/vishwanthmarri
- GitHub: github.com/marrivishwanth24
- Live Demo: https://rag-frontend-topaz.vercel.app
