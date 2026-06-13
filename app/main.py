import os
import re
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

import asyncpg
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import anthropic as _anthropic_sdk

from app.ingestion import ingest_document
from app.retrieval import list_documents
from app.orchestrator import RetrievalAgent, run_pipeline
from app.eval import evaluate
from app.auth import create_access_token, hash_password, verify_password

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
FALLBACK_USER_ID = "00000000-0000-0000-0000-000000000000"


def get_session_id(request: Request) -> str:
    sid = request.headers.get("X-Session-ID", "")
    return sid if _UUID_RE.match(sid) else FALLBACK_USER_ID


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL")
    app.state.pool = None
    if db_url:
        try:
            app.state.pool = await asyncpg.create_pool(db_url)
        except Exception as e:
            print(f"WARNING: Could not connect to database: {e}")
    yield
    if app.state.pool:
        await app.state.pool.close()


app = FastAPI(title="RAG Agent Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def get_pool(request: Request) -> asyncpg.Pool:
    pool = request.app.state.pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    return pool


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = []


class RegisterRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG Agent Platform is running"}


@app.post("/auth/register")
async def register(body: RegisterRequest, request: Request):
    pool = get_pool(request)
    existing = await pool.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    await pool.execute(
        "INSERT INTO users (id, email, password_hash) VALUES ($1, $2, $3)",
        user_id,
        body.email,
        hash_password(body.password),
    )
    token = create_access_token({"sub": user_id, "email": body.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/token")
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
):
    pool = get_pool(request)
    row = await pool.fetchrow(
        "SELECT id, password_hash FROM users WHERE email = $1", form.username
    )
    if not row or not verify_password(form.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": row["id"], "email": form.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    pool = get_pool(request)
    contents = await file.read()
    session_id = get_session_id(request)
    doc_id = await ingest_document(contents, file.filename, session_id, pool)
    return {"document_id": doc_id, "filename": file.filename, "status": "ingested"}


@app.post("/query")
async def query_agent(body: QueryRequest, request: Request):
    pool = get_pool(request)
    return StreamingResponse(
        run_pipeline(
            question=body.question,
            document_ids=body.document_ids,
            user_id=get_session_id(request),
            pool=pool,
        ),
        media_type="text/event-stream",
    )


class EvalRequest(BaseModel):
    question: str
    document_ids: list[str] = []


@app.post("/eval")
async def eval_rag(body: EvalRequest, request: Request):
    """
    Run the full multi-agent pipeline and return a quality scorecard.
    Scores: faithfulness, answer_relevance, context_quality (each 0–1).
    """
    pool = get_pool(request)
    session_id = get_session_id(request)

    # Use the same RetrievalAgent as /query for consistency
    chunks = await RetrievalAgent().retrieve(
        body.question, body.document_ids, session_id, pool
    )

    # Generate a non-streaming answer for the judge to evaluate
    _client = _anthropic_sdk.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    formatted = "\n\n".join(
        f"[Source: {c['filename']}, Page {c.get('page_num', '?')}]\n{c['chunk_text']}"
        for c in chunks
    )
    resp = await _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Context:\n{formatted}\n\n---\n\nQuestion: {body.question}"}],
    )
    answer = resp.content[0].text

    scores = await evaluate(body.question, answer, chunks)
    return {
        "question": body.question,
        "answer": answer,
        "scores": scores,
        "sources": [
            {"filename": c["filename"], "page_num": c["page_num"], "similarity": round(c["similarity"], 2)}
            for c in chunks
        ],
    }


@app.get("/documents")
async def list_user_documents(request: Request):
    pool = get_pool(request)
    docs = await list_documents(get_session_id(request), pool)
    return {"documents": docs}
