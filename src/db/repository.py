"""Repository functions for query audit logs.

Thin data-access layer that keeps SQL concerns out of the business logic
and UI modules.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from src.db.models import QueryLog
from src.db.session import get_session

logger = logging.getLogger(__name__)


def save_query_log(
    question: str,
    answer: str,
    sources: list[Document],
) -> None:
    """Persist a query and its result to the audit log table.

    Args:
        question: The user's original question.
        answer: The generated answer text.
        sources: Document chunks cited in the answer; source paths are
            extracted from their metadata and joined as a CSV string.
    """
    db = get_session()
    try:
        sources_str = ", ".join(doc.metadata.get("source", "unknown") for doc in sources)
        db.add(QueryLog(question=question, answer=answer, sources=sources_str))
        db.commit()
        logger.debug("Query log saved.")
    except Exception:
        logger.exception("Failed to save query log.")
        db.rollback()
    finally:
        db.close()


def get_recent_logs(limit: int = 5) -> list[QueryLog]:
    """Retrieve the most recent query log entries.

    Args:
        limit: Maximum number of records to return.

    Returns:
        A list of :class:`QueryLog` instances ordered newest-first.
        Returns an empty list on database errors.
    """
    db = get_session()
    try:
        return (
            db.query(QueryLog)
            .order_by(QueryLog.timestamp.desc(), QueryLog.id.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        logger.exception("Failed to retrieve query logs.")
        return []
    finally:
        db.close()
