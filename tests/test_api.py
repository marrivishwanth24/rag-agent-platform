"""Tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
from app.main import app
from app.auth import create_access_token

client = TestClient(app)


def auth_headers(user_id: str = "test-user-id", email: str = "test@example.com") -> dict:
    token = create_access_token({"sub": user_id, "email": email})
    return {"Authorization": f"Bearer {token}"}


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_status_field():
    response = client.get("/health")
    assert "status" in response.json()


# ── Auth — protected routes reject unauthenticated requests ───────────────────

def test_upload_without_auth_returns_401():
    response = client.post("/upload")
    assert response.status_code == 401


def test_query_without_auth_returns_401():
    response = client.post("/query")
    assert response.status_code == 401


def test_documents_without_auth_returns_401():
    response = client.get("/documents")
    assert response.status_code == 401


# ── Validation — authenticated requests with bad bodies return 422 ─────────────

def test_upload_with_auth_and_no_file_returns_422():
    response = client.post("/upload", headers=auth_headers())
    assert response.status_code == 422


def test_query_with_auth_and_no_body_returns_422():
    response = client.post("/query", headers=auth_headers())
    assert response.status_code == 422


# ── Auth token payload ─────────────────────────────────────────────────────────

def test_created_token_is_a_string():
    token = create_access_token({"sub": "user-1", "email": "a@b.com"})
    assert isinstance(token, str) and len(token) > 0


def test_created_token_has_three_parts():
    """JWT format: header.payload.signature"""
    token = create_access_token({"sub": "user-1", "email": "a@b.com"})
    assert len(token.split(".")) == 3
