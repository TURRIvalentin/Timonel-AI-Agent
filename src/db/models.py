"""SQLAlchemy ORM models for Timonel RAG audit logging."""
from __future__ import annotations

import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """Audit record persisted for every RAG query."""

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    question: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(String)
    # Comma-separated list of source file paths cited in the answer.
    sources: Mapped[str] = mapped_column(String)

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id!r}, question={self.question[:30]!r})>"
