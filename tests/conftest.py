"""Shared pytest fixtures for the Timonel RAG test suite.

Key design decisions
--------------------
* ``test_settings`` always passes explicit kwargs so that whatever values
  are in a local ``.env`` file or shell environment cannot bleed into tests.
  Init kwargs have the highest priority in pydantic-settings.

* ``patch_db`` monkeypatches the two module-level globals in
  ``src.db.session`` (``_engine`` / ``_SessionLocal``) rather than
  redirecting the DB path via Settings.  This isolates every test that
  touches the repository layer without needing a real file on disk.
  After each test, pytest's monkeypatch undoes the patch automatically.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import Settings
from src.db.models import Base


@pytest.fixture()
def test_settings(tmp_path: pytest.TempPathFactory) -> Settings:
    """Settings instance wired to ``tmp_path`` directories.

    All values are supplied explicitly via init kwargs so environment
    variables and ``.env`` file contents cannot affect the result.
    """
    return Settings(
        pdf_directory=tmp_path / "pdfs",  # type: ignore[arg-type]
        chroma_db_dir=tmp_path / "chroma",  # type: ignore[arg-type]
        db_path=tmp_path / "test.db",  # type: ignore[arg-type]
        embeddings_provider="huggingface",
        llm_provider="openai",
        openai_api_key="sk-test-dummy",
        chunk_size=200,
        chunk_overlap=20,
        retriever_k=2,
    )


@pytest.fixture()
def patch_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all DB operations to an isolated in-memory SQLite instance.

    Patches ``src.db.session._engine`` and ``src.db.session._SessionLocal``
    so that every call to ``get_session()`` uses the in-memory engine for
    the duration of the test.  The patch is undone automatically after each
    test by pytest's monkeypatch mechanism.
    """
    import src.db.session as session_module

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(session_module, "_SessionLocal", session_factory)
