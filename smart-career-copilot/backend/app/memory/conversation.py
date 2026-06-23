"""
Conversation memory — stores and retrieves chat history per session.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from app.models.database import DBMessage, SessionLocal, init_db
from app.models.schemas import ChatMessage, MessageRole
from app.utils.helpers import generate_id, utc_now
from app.utils.logger import logger


class ConversationMemory:
    """Manage per-session conversation history with SQLite persistence."""

    def __init__(self):
        self._cache: Dict[str, List[ChatMessage]] = {}

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to session history."""
        if session_id not in self._cache:
            self._cache[session_id] = []

        if not message.id:
            message.id = generate_id()
        if not message.timestamp:
            message.timestamp = utc_now()

        self._cache[session_id].append(message)

        # Persist to database
        try:
            db = SessionLocal()
            db_msg = DBMessage(
                id=message.id,
                session_id=session_id,
                role=message.role.value,
                content=message.content,
                agent_type=message.agent_type.value if message.agent_type else "general",
                metadata_json=json.dumps(message.metadata),
                sources_json=json.dumps(message.sources),
                created_at=message.timestamp,
            )
            db.add(db_msg)
            db.commit()
            db.close()
        except Exception as e:
            logger.error("Failed to persist message: %s", str(e))

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Get messages for a session."""
        # Check cache first
        if session_id in self._cache:
            messages = self._cache[session_id]
            if limit:
                return messages[-limit:]
            return messages

        # Load from database
        try:
            db = SessionLocal()
            query = (
                db.query(DBMessage)
                .filter(DBMessage.session_id == session_id)
                .order_by(DBMessage.created_at.asc())
            )
            if limit:
                query = query.limit(limit)

            db_messages = query.all()
            messages = []
            for m in db_messages:
                msg = ChatMessage(
                    id=m.id,
                    role=MessageRole(m.role),
                    content=m.content,
                    timestamp=m.created_at,
                    metadata=json.loads(m.metadata_json or "{}"),
                    sources=json.loads(m.sources_json or "[]"),
                )
                messages.append(msg)

            self._cache[session_id] = messages
            db.close()
            return messages
        except Exception as e:
            logger.error("Failed to load messages: %s", str(e))
            return []

    def get_context_window(
        self,
        session_id: str,
        window_size: int = 10,
    ) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM context."""
        messages = self.get_messages(session_id, limit=window_size)
        return [
            {"role": m.role.value, "content": m.content}
            for m in messages
        ]

    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        self._cache.pop(session_id, None)
        try:
            db = SessionLocal()
            db.query(DBMessage).filter(
                DBMessage.session_id == session_id
            ).delete()
            db.commit()
            db.close()
        except Exception as e:
            logger.error("Failed to clear session: %s", str(e))

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        if session_id in self._cache:
            return len(self._cache[session_id])
        try:
            db = SessionLocal()
            count = (
                db.query(DBMessage)
                .filter(DBMessage.session_id == session_id)
                .count()
            )
            db.close()
            return count
        except Exception:
            return 0
