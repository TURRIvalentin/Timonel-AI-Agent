"""Tests for src/db/ — repository functions against an in-memory SQLite DB.

The ``patch_db`` fixture (from conftest.py) redirects all DB operations to
an isolated in-memory SQLite instance, so the real database is never touched.
"""

from __future__ import annotations

from langchain_core.documents import Document
from sqlalchemy import inspect

from src.db.models import QueryLog
from src.db.repository import get_recent_logs, save_query_log
from src.db.session import init_db

# ---------------------------------------------------------------------------
# save_query_log
# ---------------------------------------------------------------------------


class TestSaveQueryLog:
    def test_saves_a_single_entry(self, patch_db: None) -> None:
        save_query_log("What is fondeo?", "Fondeo is anchoring.", [])
        logs = get_recent_logs(10)
        assert len(logs) == 1
        assert logs[0].question == "What is fondeo?"
        assert logs[0].answer == "Fondeo is anchoring."

    def test_sources_serialized_as_csv_string(self, patch_db: None) -> None:
        docs = [
            Document(page_content="p1", metadata={"source": "/docs/a.pdf"}),
            Document(page_content="p2", metadata={"source": "/docs/b.pdf"}),
        ]
        save_query_log("Q?", "A.", docs)
        log = get_recent_logs(1)[0]
        assert "/docs/a.pdf" in log.sources
        assert "/docs/b.pdf" in log.sources

    def test_empty_sources_stores_empty_string(self, patch_db: None) -> None:
        save_query_log("Q?", "A.", [])
        log = get_recent_logs(1)[0]
        assert log.sources == ""

    def test_source_falls_back_to_unknown_when_missing(self, patch_db: None) -> None:
        doc = Document(page_content="text", metadata={})  # no "source" key
        save_query_log("Q?", "A.", [doc])
        log = get_recent_logs(1)[0]
        assert "unknown" in log.sources

    def test_multiple_logs_are_all_saved(self, patch_db: None) -> None:
        for i in range(3):
            save_query_log(f"Q{i}?", f"A{i}.", [])
        logs = get_recent_logs(10)
        assert len(logs) == 3

    def test_returns_none_on_success(self, patch_db: None) -> None:
        result = save_query_log("Q?", "A.", [])
        assert result is None


# ---------------------------------------------------------------------------
# get_recent_logs
# ---------------------------------------------------------------------------


class TestGetRecentLogs:
    def test_returns_empty_list_when_no_entries(self, patch_db: None) -> None:
        assert get_recent_logs(5) == []

    def test_respects_limit_parameter(self, patch_db: None) -> None:
        for i in range(5):
            save_query_log(f"Q{i}?", f"A{i}.", [])
        logs = get_recent_logs(3)
        assert len(logs) == 3

    def test_ordered_newest_first(self, patch_db: None) -> None:
        save_query_log("first", "A1", [])
        save_query_log("second", "A2", [])
        save_query_log("third", "A3", [])
        logs = get_recent_logs(3)
        assert logs[0].question == "third"
        assert logs[1].question == "second"
        assert logs[2].question == "first"

    def test_returns_query_log_instances(self, patch_db: None) -> None:
        save_query_log("Q?", "A.", [])
        logs = get_recent_logs(1)
        assert isinstance(logs[0], QueryLog)

    def test_limit_zero_returns_empty(self, patch_db: None) -> None:
        save_query_log("Q?", "A.", [])
        assert get_recent_logs(0) == []

    def test_limit_larger_than_entries_returns_all(self, patch_db: None) -> None:
        save_query_log("Q1?", "A1", [])
        save_query_log("Q2?", "A2", [])
        logs = get_recent_logs(100)
        assert len(logs) == 2


# ---------------------------------------------------------------------------
# init_db — schema creation
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_query_logs_table(self, patch_db: None) -> None:
        """init_db must be idempotent and must create the query_logs table."""
        import src.db.session as session_module

        # Drop the table that patch_db already created, then re-run init_db
        from src.db.models import Base

        Base.metadata.drop_all(bind=session_module._engine)
        init_db()  # should recreate schema without raising

        engine = session_module._engine
        assert engine is not None
        table_names = inspect(engine).get_table_names()
        assert "query_logs" in table_names

    def test_init_db_is_idempotent(self, patch_db: None) -> None:
        """Calling init_db twice must not raise."""
        init_db()
        init_db()  # second call — tables already exist
