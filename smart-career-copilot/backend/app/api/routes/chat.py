"""
Chat API routes — main endpoint for agent interaction.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket

from app.agents.orchestrator import run_agent
from app.api.websocket import handle_websocket
from app.memory.conversation import ConversationMemory
from app.memory.session import SessionManager
from app.models.schemas import (
    AgentType,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    SessionInfo,
    SessionListResponse,
)
from app.utils.helpers import generate_id, utc_now
from app.utils.logger import logger

router = APIRouter()
session_manager = SessionManager()
conversation_memory = ConversationMemory()


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a message and get an agent response."""
    # Create or get session
    session_id = request.session_id or session_manager.create_session(
        agent_type=request.agent_type or AgentType.GENERAL
    )

    # Save user message
    metadata = {}
    if request.attachments:
        metadata["attachments"] = request.attachments

    user_msg = ChatMessage(
        id=generate_id(),
        role=MessageRole.USER,
        content=request.message,
        agent_type=request.agent_type,
        timestamp=utc_now(),
        metadata=metadata,
    )
    conversation_memory.add_message(session_id, user_msg)

    # Run agent
    result = await run_agent(
        message=request.message,
        session_id=session_id,
        agent_type=request.agent_type,
        context=request.context,
    )

    # Create assistant message
    assistant_msg = ChatMessage(
        id=generate_id(),
        role=MessageRole.ASSISTANT,
        content=result.get("response", "I couldn't process your request."),
        agent_type=AgentType(result.get("agent_type", "general")),
        timestamp=utc_now(),
        sources=result.get("sources", []),
        metadata=result.get("artifacts", {}),
    )
    conversation_memory.add_message(session_id, assistant_msg)

    # Update session title if first message
    session_manager.update_title(session_id, request.message)

    return ChatResponse(
        message=assistant_msg,
        session_id=session_id,
        agent_used=AgentType(result.get("agent_type", "general")),
        thinking_steps=result.get("thinking_steps", []),
        sources=result.get("sources", []),
        artifacts=result.get("artifacts", {}),
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all chat sessions."""
    sessions = session_manager.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages for a session."""
    messages = conversation_memory.get_messages(session_id)
    return {"session_id": session_id, "messages": [m.model_dump() for m in messages]}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    session_manager.delete_session(session_id)
    conversation_memory.clear_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time streaming."""
    await handle_websocket(websocket, session_id)
