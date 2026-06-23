"""Startup utilities for Timonel RAG.

Contains the auto-ingest chain-initialisation flow in a Streamlit-free
module so that it can be unit-tested without importing any UI framework.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import Runnable

from src.config import Settings
from src.core.ingest import ingest
from src.core.retrieval import setup_qa_chain

logger = logging.getLogger(__name__)


def init_rag_chain(cfg: Settings) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Return a ready QA chain, auto-ingesting from *cfg.pdf_directory* if needed.

    Tries to load the existing vector store first.  When the store is missing
    or empty, discovers PDF files under *cfg.pdf_directory* and runs the full
    ingest pipeline before returning the chain.

    Args:
        cfg: Application settings.

    Returns:
        A LangChain retrieval chain that accepts ``{"input": "<question>"}``
        and returns ``{"answer": str, "context": list[Document]}``.

    Raises:
        RuntimeError: When the vector store is absent/empty AND no PDF files
            are found in *cfg.pdf_directory*.
    """
    try:
        return setup_qa_chain(cfg)
    except RuntimeError as exc:
        if _is_missing_or_empty(exc):
            _auto_ingest(cfg, reason=exc)
            return setup_qa_chain(cfg)
        raise


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_missing_or_empty(exc: Exception) -> bool:
    """Return True when *exc* signals a missing or empty ChromaDB collection."""
    msg = str(exc).lower()
    return "not found" in msg or "empty" in msg


def _auto_ingest(cfg: Settings, reason: Exception) -> None:
    """Run the ingest pipeline or raise a descriptive error if no PDFs exist."""
    pdfs = list(cfg.pdf_directory.rglob("*.pdf")) if cfg.pdf_directory.exists() else []
    if not pdfs:
        raise RuntimeError(
            f"Vector store not initialised and no PDF files found in "
            f"{cfg.pdf_directory}. "
            "Add PDF files to the data/ directory and redeploy."
        ) from reason
    logger.info(
        "Auto-ingest: %d PDF(s) found in %s — building vector store.",
        len(pdfs),
        cfg.pdf_directory,
    )
    ingest(cfg)
