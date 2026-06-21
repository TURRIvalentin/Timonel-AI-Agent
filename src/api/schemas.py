"""Pydantic request / response schemas for the Timonel RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for POST /query."""

    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer.")


class SourceDocument(BaseModel):
    """A single retrieved document chunk included in a query response."""

    content: str
    source: str
    page: int | str


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    question: str
    answer: str
    sources: list[SourceDocument]


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    status: str
    pages_loaded: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    vector_store: str  # "ready" | "not_initialized"
