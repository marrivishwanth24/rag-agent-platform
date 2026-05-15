# RAG Agent Platform

A production full-stack AI agent system for semantic document search and conversational Q&A, built with the **Claude API**, **FastAPI**, **pgvector**, and **React**.

**Status:** 
🤖 RAG Agent Platform
Live Demo: https://rag-frontend-topaz.vercel.app
API Docs: https://rag-agent-platform-production.up.railway.app/docs

---

## Architecture

```
User → React Frontend → FastAPI Backend → Claude API (Agent)
                              ↓
                       pgvector (PostgreSQL)
                              ↑
                    Document Ingestion Pipeline
                    (PDF → chunks → embeddings)
```

## Features

- **AI Agent** — Multi-turn conversational Q&A over uploaded documents using Claude API
- **RAG Pipeline** — PDF ingestion, chunking, embedding, and vector indexing into pgvector
- **Sub-200ms retrieval** — Optimized semantic search across concurrent users
- **Streaming responses** — Real-time token streaming from Claude to React frontend
- **OAuth2/JWT auth** — Secure authentication with RBAC access controls
- **S3-backed storage** — Document management via AWS S3

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Tailwind CSS |
| Backend | Python, FastAPI |
| AI / LLM | Claude API (Anthropic), LangChain |
| Vector DB | pgvector (PostgreSQL) |
| Auth | OAuth2, JWT |
| Cloud | AWS EC2, S3, RDS |
| DevOps | Docker, GitHub Actions CI/CD |

## Getting Started

```bash
# Clone the repo
git clone https://github.com/vishwanthmarri/rag-agent-platform
cd rag-agent-platform

# Set up environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY, DATABASE_URL, AWS credentials

# Run with Docker
docker-compose up --build

# Or run locally
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Project Structure

```
rag-agent-platform/
├── app/
│   ├── main.py          # FastAPI application entry point
│   ├── agent.py         # Claude API agent orchestration
│   ├── ingestion.py     # Document ingestion pipeline
│   ├── retrieval.py     # pgvector semantic search
│   ├── auth.py          # OAuth2/JWT authentication
│   └── models.py        # Database models
├── frontend/            # React/TypeScript frontend
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload and ingest a PDF document |
| POST | `/query` | Query the agent with a question |
| GET | `/documents` | List uploaded documents |
| DELETE | `/documents/{id}` | Delete a document |

## Environment Variables

```env
ANTHROPIC_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost/ragdb
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=your_bucket
SECRET_KEY=your_jwt_secret
```
