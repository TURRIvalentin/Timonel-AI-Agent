"""Timonel RAG — FastAPI application entry point.

Run locally:
    uvicorn src.api.main:app --reload

The interactive Swagger UI is available at http://localhost:8000/docs.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.core.retrieval import setup_qa_chain
from src.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown lifecycle.

    Startup:
    - Configure logging from settings.
    - Initialise the SQLite audit-log database.
    - Attempt to load the QA chain.  If the vector store is not ready
      (e.g. ingest has not been run yet), ``app.state.qa_chain`` is set
      to ``None`` and /query will return 503 until POST /ingest is called.

    Shutdown:
    - No resources to release (ChromaDB and SQLite sessions are stateless).
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    init_db()

    try:
        app.state.qa_chain = setup_qa_chain(settings)
        logger.info("QA chain initialised successfully at startup.")
    except RuntimeError as exc:
        logger.warning("QA chain not available at startup: %s", exc)
        logger.warning("Call POST /ingest to build the vector store, then retry.")
        app.state.qa_chain = None

    yield


app = FastAPI(
    title="Timonel RAG API",
    description=(
        "REST interface for the Timonel nautical-document question-answering system.\n\n"
        "**Typical workflow:**\n"
        "1. `POST /ingest` — index the PDF library (run once, or after adding new files).\n"
        "2. `POST /query` — ask questions grounded in the indexed documents.\n"
        "3. `GET /health` — check service and vector-store status."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins by default for development.
# In production, replace ["*"] with your frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)
