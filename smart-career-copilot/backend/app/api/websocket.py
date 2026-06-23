"""
WebSocket handler for real-time streaming of agent responses.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from app.agents.orchestrator import run_agent
from app.models.schemas import AgentType, StreamChunk
from app.utils.logger import logger


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected: %s", session_id)

    def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)
        logger.info("WebSocket disconnected: %s", session_id)

    async def send_chunk(self, session_id: str, chunk: StreamChunk) -> None:
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(chunk.model_dump_json())

    async def send_json(self, session_id: str, data: Dict[str, Any]) -> None:
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_text(json.dumps(data))


manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket, session_id: str) -> None:
    """Handle WebSocket connection for real-time agent streaming."""
    await manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            message = request.get("message", "")
            agent_type = request.get("agent_type", "general")
            context = request.get("context", {})

            # Send thinking indicator
            await manager.send_chunk(
                session_id,
                StreamChunk(type="thinking", content="Analyzing your request..."),
            )

            try:
                # Run the agent
                result = await run_agent(
                    message=message,
                    session_id=session_id,
                    agent_type=AgentType(agent_type) if agent_type != "general" else None,
                    context=context,
                )

                # Stream the response
                await manager.send_chunk(
                    session_id,
                    StreamChunk(
                        type="token",
                        content=result.get("response", ""),
                        metadata={
                            "agent_type": result.get("agent_type", "general"),
                            "sources": result.get("sources", []),
                            "artifacts": result.get("artifacts", {}),
                            "thinking_steps": result.get("thinking_steps", []),
                        },
                    ),
                )

                # Send completion signal
                await manager.send_chunk(
                    session_id,
                    StreamChunk(type="done", content=""),
                )

            except Exception as e:
                logger.error("Agent error: %s", traceback.format_exc())
                await manager.send_chunk(
                    session_id,
                    StreamChunk(type="error", content=str(e)),
                )

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", traceback.format_exc())
        manager.disconnect(session_id)
