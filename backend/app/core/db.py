"""Database engine / session management with dual-mode support.

`DATABASE_URL` may point at PostgreSQL (production) or SQLite (zero-setup local
development and tests). Engine options are chosen per-dialect.
"""
from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("db")


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


def _engine_kwargs(url: str) -> dict[str, Any]:
    if url.startswith("sqlite"):
        kwargs: dict[str, Any] = {
            "connect_args": {"check_same_thread": False, "timeout": 30},
            "future": True,
        }
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        else:
            db_path = url.split("sqlite:///")[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return kwargs
    return {
        "pool_size": settings.DB_POOL_SIZE,
        "max_overflow": settings.DB_MAX_OVERFLOW,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "future": True,
    }


def build_engine(url: str | None = None, echo: bool | None = None) -> Engine:
    url = url or settings.DATABASE_URL
    engine = create_engine(url, echo=settings.DB_ECHO if echo is None else echo, **_engine_kwargs(url))

    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver callback
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA foreign_keys=ON")
                # WAL needs shared-memory locking, which some network/FUSE mounts
                # do not provide. Fall back rather than failing every connection.
                for mode in ("WAL", "TRUNCATE", "DELETE"):
                    try:
                        cur.execute(f"PRAGMA journal_mode={mode}")
                        break
                    except Exception:  # noqa: BLE001 - probing filesystem capability
                        continue
                try:
                    cur.execute("PRAGMA synchronous=NORMAL")
                except Exception:  # noqa: BLE001
                    pass
            finally:
                cur.close()

    return engine


engine: Engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background workers."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ping(target_engine: Engine | None = None) -> bool:
    try:
        with (target_engine or engine).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - depends on env
        logger.warning("database_ping_failed", error=str(exc))
        return False


def create_all(target_engine: Engine | None = None) -> None:
    """Create the schema directly (used for SQLite dev/test; Alembic owns PG)."""
    import app.models  # noqa: F401  ensure models are imported/registered

    Base.metadata.create_all(bind=target_engine or engine)


def dialect_name(target_engine: Engine | None = None) -> str:
    return (target_engine or engine).dialect.name
