from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.ingestion import ingest_document
from app.retrieval import retrieve_context
from app.agent import stream_agent_response
from app.auth import get_current_user
import uvicorn

app = FastAPI(title="RAG Agent Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] = []


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and ingest a PDF document into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    contents = await file.read()
    doc_id = await ingest_document(contents, file.filename, current_user["user_id"])

    return {"document_id": doc_id, "filename": file.filename, "status": "ingested"}


@app.post("/query")
async def query_agent(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Query the AI agent with semantic retrieval and streaming response."""
    context_chunks = await retrieve_context(
        query=request.question,
        document_ids=request.document_ids,
        user_id=current_user["user_id"]
    )

    return StreamingResponse(
        stream_agent_response(request.question, context_chunks),
        media_type="text/event-stream"
    )


@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all documents uploaded by the current user."""
    from app.models import get_user_documents
    docs = await get_user_documents(current_user["user_id"])
    return {"documents": docs}


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a document and its vectors from the store."""
    from app.models import delete_document_by_id
    await delete_document_by_id(document_id, current_user["user_id"])
    return {"status": "deleted", "document_id": document_id}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
