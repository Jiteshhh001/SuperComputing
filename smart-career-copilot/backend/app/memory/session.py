"""
Session management — create, list, and manage chat sessions.
"""

from __future__ import annotations

from typing import List, Optional

from app.models.database import DBSession, SessionLocal
from app.models.schemas import AgentType, SessionInfo
from app.utils.helpers import generate_session_id, utc_now
from app.utils.logger import logger


class SessionManager:
    """Manage chat sessions with SQLite persistence."""

    def create_session(
        self,
        agent_type: AgentType = AgentType.GENERAL,
        title: str = "New Conversation",
    ) -> str:
        """Create a new session and return its ID."""
        session_id = generate_session_id()
        now = utc_now()

        try:
            db = SessionLocal()
            db_session = DBSession(
                id=session_id,
                agent_type=agent_type.value,
                title=title,
                created_at=now,
                updated_at=now,
            )
            db.add(db_session)
            db.commit()
            db.close()
            logger.info("Session created: %s (%s)", session_id, agent_type.value)
        except Exception as e:
            logger.error("Failed to create session: %s", str(e))

        return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session info by ID."""
        try:
            db = SessionLocal()
            s = db.query(DBSession).filter(DBSession.id == session_id).first()
            db.close()
            if s:
                return SessionInfo(
                    session_id=s.id,
                    agent_type=AgentType(s.agent_type),
                    title=s.title or "New Conversation",
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
            return None
        except Exception as e:
            logger.error("Failed to get session: %s", str(e))
            return None

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions, most recent first."""
        try:
            db = SessionLocal()
            sessions = (
                db.query(DBSession)
                .order_by(DBSession.updated_at.desc())
                .all()
            )
            db.close()
            return [
                SessionInfo(
                    session_id=s.id,
                    agent_type=AgentType(s.agent_type),
                    title=s.title or "New Conversation",
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
                for s in sessions
            ]
        except Exception as e:
            logger.error("Failed to list sessions: %s", str(e))
            return []

    def update_title(self, session_id: str, first_message: str) -> None:
        """Update session title based on the first user message."""
        title = first_message[:80].strip()
        if len(first_message) > 80:
            title += "..."

        try:
            db = SessionLocal()
            s = db.query(DBSession).filter(DBSession.id == session_id).first()
            if s and s.title == "New Conversation":
                s.title = title
                s.updated_at = utc_now()
                db.commit()
            db.close()
        except Exception as e:
            logger.error("Failed to update title: %s", str(e))

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        try:
            db = SessionLocal()
            db.query(DBSession).filter(DBSession.id == session_id).delete()
            db.commit()
            db.close()
            logger.info("Session deleted: %s", session_id)
        except Exception as e:
            logger.error("Failed to delete session: %s", str(e))
