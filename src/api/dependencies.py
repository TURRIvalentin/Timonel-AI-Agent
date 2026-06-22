"""FastAPI dependency factories for the Timonel RAG API."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from langchain_core.runnables import Runnable

from src.config import Settings
from src.config import settings as _settings


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return _settings


def get_qa_chain(request: Request) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Return the cached QA chain from app state, or raise 503.

    The chain is built once during the application lifespan (startup) and
    stored in ``app.state.qa_chain``.  If it is ``None``, the vector store
    has not been initialised yet.
    """
    chain = getattr(request.app.state, "qa_chain", None)
    if chain is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The RAG system is not initialised. "
                "Call POST /ingest to build the vector store first."
            ),
        )
    return chain
