"""
Intent Router — classifies user input and routes to the appropriate agent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from app.config import settings
from app.models.schemas import AgentType
from app.utils.logger import logger


ROUTING_PROMPT = """You are an intent classifier for a career copilot application.
Classify the user's message into exactly ONE of these categories:

- "resume" — Resume related: uploading, analyzing, improving resumes, ATS scores, cover letters, skill extraction
- "research" — Research tasks: web search, information gathering, topic research, summarization
- "coding" — Coding tasks: writing code, debugging, creating projects, running tests, code review
- "interview" — Interview prep: mock interviews, practice questions, feedback, behavioral or technical interview practice
- "general" — General conversation that doesn't fit the above categories

Reply with ONLY the category name, nothing else.

User message: {message}
"""


class Router:
    """Routes user messages to the appropriate agent."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.worker_model,
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    async def classify(
        self,
        message: str,
        forced_agent: Optional[AgentType] = None,
    ) -> AgentType:
        """Classify user intent and return the target agent type."""
        # If agent is explicitly specified, use it
        if forced_agent and forced_agent != AgentType.GENERAL:
            logger.info("Forced routing to: %s", forced_agent.value)
            return forced_agent

        try:
            response = self.llm.invoke(
                ROUTING_PROMPT.format(message=message[:500])
            )
            result = response.content.strip().lower()

            # Map response to AgentType
            mapping = {
                "resume": AgentType.RESUME,
                "research": AgentType.RESEARCH,
                "coding": AgentType.CODING,
                "interview": AgentType.INTERVIEW,
                "general": AgentType.GENERAL,
            }

            agent_type = mapping.get(result, AgentType.GENERAL)
            logger.info("Routed '%s...' -> %s", message[:50], agent_type.value)
            return agent_type

        except Exception as e:
            logger.error("Routing error: %s", str(e))
            return AgentType.GENERAL


router = Router()
