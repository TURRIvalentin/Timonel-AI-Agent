"""HTTP route handlers for the Timonel RAG API.

All handlers delegate to the core layer (src/core/) and the DB repository
(src/db/).  Zero RAG logic lives here — this module is pure HTTP glue.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.runnables import Runnable

from src.api.dependencies import get_qa_chain, get_settings
from src.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from src.config import Settings
from src.core.ingest import ingest as run_ingest
from src.core.retrieval import execute_query, setup_qa_chain
from src.db.repository import save_query_log

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health(request: Request) -> HealthResponse:
    """Return API liveness status and vector-store readiness.

    This endpoint never calls external APIs or paid services.  It is safe
    to use as a Docker / Kubernetes liveness probe.
    """
    chain_ready = getattr(request.app.state, "qa_chain", None) is not None
    return HealthResponse(
        status="ok",
        vector_store="ready" if chain_ready else "not_initialized",
    )


@router.post("/query", response_model=QueryResponse, summary="Answer a question")
def query_endpoint(
    body: QueryRequest,
    chain: Annotated[Runnable[dict[str, Any], dict[str, Any]], Depends(get_qa_chain)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    """Retrieve relevant document chunks and generate a grounded answer.

    Uses the QA chain cached at startup.  The query is logged to the audit
    database after a successful response.

    Raises:
        503: When the vector store is not initialised (chain is None) or the
             underlying chain invocation fails.
        422: When the request body fails validation (e.g. empty question).
    """
    try:
        result = execute_query(body.question, chain)
    except Exception as exc:
        logger.exception("Query execution failed.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    save_query_log(result.question, result.answer, result.sources)

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", "?"),
            )
            for doc in result.sources
        ],
    )


@router.post("/ingest", response_model=IngestResponse, summary="Re-index PDFs")
def ingest_endpoint(
    request: Request,
    cfg: Annotated[Settings, Depends(get_settings)],
) -> IngestResponse:
    """Load PDFs from the configured directory and rebuild the vector store.

    This operation is **synchronous** — the request blocks until indexing
    completes.  For large document libraries this may take several minutes.
    The QA chain is reloaded automatically after a successful ingest so that
    subsequent /query calls use the fresh index without restarting the server.

    Raises:
        503: When the PDF directory does not exist or no PDFs are found.
    """
    try:
        result = run_ingest(cfg)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Reload the chain so /query immediately reflects the new index.
    try:
        request.app.state.qa_chain = setup_qa_chain(cfg)
        logger.info("QA chain reloaded after ingest.")
    except Exception:
        logger.exception("Chain reload after ingest failed — /query may be stale.")
        request.app.state.qa_chain = None

    return IngestResponse(
        status="ok",
        pages_loaded=result.pages_loaded,
        chunks_indexed=result.chunks_indexed,
    )
