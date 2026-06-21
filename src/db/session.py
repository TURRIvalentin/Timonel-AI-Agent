"""Database engine and session factory.

The engine is created lazily on first use from the ``settings`` singleton,
so importing this module does not immediately open a database connection.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings as _settings
from src.db.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        db_url = f"sqlite:///{_settings.db_path}"
        logger.debug("Initialising database engine: %s", db_url)
        _engine = create_engine(db_url, connect_args={"check_same_thread": False})
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def init_db() -> None:
    """Create all tables that do not yet exist.

    Safe to call multiple times — SQLAlchemy skips tables that are already
    present.  Creates the parent directory of the SQLite file if needed.
    """
    _settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database ready at %s", _settings.db_path)


def get_session() -> Session:
    """Return a new database session.

    Returns:
        A SQLAlchemy :class:`Session`.  The caller is responsible for
        closing it (or using it as a context manager).
    """
    _get_engine()
    assert _SessionLocal is not None  # guaranteed by _get_engine()
    return _SessionLocal()
