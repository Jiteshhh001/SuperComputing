"""
SQLite database models and session management using SQLAlchemy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class DBSession(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    agent_type = Column(String, nullable=False, default="general")
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


class DBMessage(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    agent_type = Column(String, default="general")
    metadata_json = Column(Text, default="{}")
    sources_json = Column(Text, default="[]")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DBMemory(Base):
    """Long-term memory entries."""

    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    session_id = Column(String, nullable=True, index=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String, default="fact")
    importance = Column(Integer, default=5)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Database Engine Setup ────────────────────────────────────

# Use synchronous SQLite for simplicity (works well for portfolio project)
_db_path = settings.sqlite_url.replace("sqlite+aiosqlite:///", "sqlite:///")
engine = create_engine(_db_path, echo=settings.debug, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get a database session. Use as a context manager or dependency."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise
