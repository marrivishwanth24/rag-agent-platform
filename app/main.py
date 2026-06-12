import os
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
from app.ingestion import ingest_document
from app.retrieval import retrieve_context, list_documents
from app.agent import stream_agent_response
from app.auth import create_access_token, hash_password, verify_password

# Shared user ID for all public (unauthenticated) requests
PUBLIC_USER_ID = "00000000-0000-0000-0000-000000000000"


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
    doc_id = await ingest_document(contents, file.filename, PUBLIC_USER_ID, pool)
    return {"document_id": doc_id, "filename": file.filename, "status": "ingested"}


@app.post("/query")
async def query_agent(body: QueryRequest, request: Request):
    pool = get_pool(request)
    context_chunks = await retrieve_context(
        query=body.question,
        document_ids=body.document_ids,
        user_id=PUBLIC_USER_ID,
        pool=pool,
    )
    return StreamingResponse(
        stream_agent_response(body.question, context_chunks),
        media_type="text/event-stream",
    )


@app.get("/documents")
async def list_user_documents(request: Request):
    pool = get_pool(request)
    docs = await list_documents(PUBLIC_USER_ID, pool)
    return {"documents": docs}
