"""
Main Orchestrator Agent — LangGraph-based multi-agent orchestration.
Implements: Reason → Act → Observe → Reflect → Retry.
Routes to specialized worker agents via the Router.
"""

from __future__ import annotations

import traceback
from typing import Any, Annotated, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI

from app.agents.router import router
from app.agents.resume_agent import ResumeAgentRunner
from app.agents.research_agent import ResearchAgentRunner
from app.agents.coding_agent import CodingAgentRunner
from app.agents.interview_agent import InterviewAgentRunner
from app.config import settings
from app.memory.conversation import ConversationMemory
from app.memory.long_term import long_term_memory
from app.models.schemas import AgentType
from app.utils.logger import logger


# ── Agent State ──────────────────────────────────────────────


class AgentState(TypedDict):
    """State passed through the orchestration graph."""
    message: str
    session_id: str
    agent_type: Optional[str]
    forced_agent: Optional[str]
    context: Dict[str, Any]
    response: str
    sources: List[Dict[str, Any]]
    artifacts: Dict[str, Any]
    thinking_steps: List[str]
    iteration: int
    quality_score: float
    error: Optional[str]


# ── Worker Instances ─────────────────────────────────────────

resume_runner = ResumeAgentRunner()
research_runner = ResearchAgentRunner()
coding_runner = CodingAgentRunner()
interview_runner = InterviewAgentRunner()
conversation_memory = ConversationMemory()


# ── Orchestration Functions ──────────────────────────────────


async def reason(state: AgentState) -> Dict[str, Any]:
    """Step 1: Reason — Analyze the request and determine the approach."""
    steps = state.get("thinking_steps", [])
    steps.append("🧠 Reasoning: Analyzing user request...")

    # Classify intent
    forced = None
    if state.get("forced_agent"):
        try:
            forced = AgentType(state["forced_agent"])
        except ValueError:
            pass

    agent_type = await router.classify(state["message"], forced)

    steps.append(f"🎯 Routed to: {agent_type.value} agent")

    return {
        "agent_type": agent_type.value,
        "thinking_steps": steps,
    }


async def act(state: AgentState) -> Dict[str, Any]:
    """Step 2: Act — Execute the appropriate worker agent."""
    agent_type = state.get("agent_type", "general")
    message = state["message"]
    context = state.get("context", {})
    steps = state.get("thinking_steps", [])
    steps.append(f"⚡ Acting: Running {agent_type} agent...")

    try:
        # Get relevant memories for context
        memories = long_term_memory.recall(message, k=3)
        if memories:
            context["memories"] = [m["content"] for m in memories]
            steps.append(f"📚 Retrieved {len(memories)} relevant memories")

        # Get conversation history
        history = conversation_memory.get_context_window(
            state.get("session_id", ""), window_size=6
        )
        if history:
            context["history"] = history

        # Route to worker
        if agent_type == "resume":
            result = await resume_runner.run(message, context)
        elif agent_type == "research":
            result = await research_runner.run(message, context)
        elif agent_type == "coding":
            result = await coding_runner.run(message, context)
        elif agent_type == "interview":
            result = await interview_runner.run(message, context)
        else:
            result = await _general_chat(message, context)

        steps.extend(result.get("thinking_steps", []))

        return {
            "response": result.get("response", ""),
            "sources": result.get("sources", []),
            "artifacts": result.get("artifacts", {}),
            "thinking_steps": steps,
        }

    except Exception as e:
        logger.error("Agent execution error: %s", traceback.format_exc())
        steps.append(f"❌ Error: {str(e)}")
        return {
            "response": f"I encountered an error while processing your request: {str(e)}",
            "error": str(e),
            "thinking_steps": steps,
        }


async def observe(state: AgentState) -> Dict[str, Any]:
    """Step 3: Observe — Check the quality of the response."""
    steps = state.get("thinking_steps", [])
    steps.append("👁️ Observing: Checking response quality...")

    response = state.get("response", "")

    # Simple quality heuristics
    quality_score = 0.5

    if len(response) > 100:
        quality_score += 0.2
    if len(response) > 300:
        quality_score += 0.1
    if state.get("error"):
        quality_score -= 0.3
    if state.get("sources"):
        quality_score += 0.1
    if "```" in response:  # Contains code blocks
        quality_score += 0.05
    if any(marker in response for marker in ["##", "**", "- "]):
        quality_score += 0.05

    quality_score = max(0.0, min(1.0, quality_score))
    steps.append(f"📊 Quality score: {quality_score:.2f}")

    return {
        "quality_score": quality_score,
        "thinking_steps": steps,
    }


async def reflect(state: AgentState) -> Dict[str, Any]:
    """Step 4: Reflect — Decide whether to retry or return."""
    steps = state.get("thinking_steps", [])
    quality = state.get("quality_score", 0.5)
    iteration = state.get("iteration", 0)

    if quality >= 0.6 or iteration >= 2:
        steps.append("✅ Response meets quality threshold")

        # Store in long-term memory
        try:
            long_term_memory.store_memory(
                content=f"Q: {state['message'][:200]}\nA: {state.get('response', '')[:300]}",
                memory_type="conversation",
                session_id=state.get("session_id"),
            )
        except Exception:
            pass

        return {"thinking_steps": steps}
    else:
        steps.append(f"🔄 Quality too low ({quality:.2f}), retrying...")
        return {
            "iteration": iteration + 1,
            "thinking_steps": steps,
        }


# ── General Chat Handler ────────────────────────────────────


async def _general_chat(
    message: str,
    context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Handle general conversation that doesn't fit specific agents."""
    llm = ChatOpenAI(
        model=settings.orchestrator_model,
        temperature=0.7,
        openai_api_key=settings.openai_api_key,
    )

    history_context = ""
    if context and context.get("history"):
        recent = context["history"][-4:]
        history_context = "\n".join(
            f"{m['role']}: {m['content']}" for m in recent
        )
        history_context = f"\nRecent conversation:\n{history_context}\n"

    memory_context = ""
    if context and context.get("memories"):
        memory_context = "\nRelevant past interactions:\n" + "\n".join(
            context["memories"][:3]
        )

    prompt = f"""You are Smart Career Copilot, an AI career development assistant.
You help with resumes, research, coding, and interview preparation.

{history_context}
{memory_context}

User: {message}

Respond helpfully. If the user's request clearly fits one of your specialized agents
(resume, research, coding, interview), mention that they can use that specific agent
for better results."""

    try:
        response = llm.invoke(prompt)
        return {
            "response": response.content,
            "agent_type": "general",
            "thinking_steps": ["Processed as general conversation"],
        }
    except Exception as e:
        return {
            "response": f"Hello! I'm Smart Career Copilot. I can help with resumes, research, coding, and interview prep. How can I assist you today?\n\n(Note: {str(e)})",
            "agent_type": "general",
        }


# ── Main Entry Point ────────────────────────────────────────


async def run_agent(
    message: str,
    session_id: str,
    agent_type: Optional[AgentType] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for the orchestrator.
    Executes the Reason → Act → Observe → Reflect pipeline.
    """
    state: AgentState = {
        "message": message,
        "session_id": session_id,
        "agent_type": None,
        "forced_agent": agent_type.value if agent_type else None,
        "context": context or {},
        "response": "",
        "sources": [],
        "artifacts": {},
        "thinking_steps": [],
        "iteration": 0,
        "quality_score": 0.0,
        "error": None,
    }

    try:
        # Step 1: Reason
        updates = await reason(state)
        state.update(updates)

        for iteration in range(3):  # Max 3 reflection loops
            # Step 2: Act
            updates = await act(state)
            state.update(updates)

            # Step 3: Observe
            updates = await observe(state)
            state.update(updates)

            # Step 4: Reflect
            updates = await reflect(state)
            state.update(updates)

            # Check if done
            if state.get("quality_score", 0) >= 0.6 or iteration >= 2:
                break

    except Exception as e:
        logger.error("Orchestrator error: %s", traceback.format_exc())
        state["response"] = f"I'm sorry, I encountered an error: {str(e)}"
        state["error"] = str(e)

    return {
        "response": state.get("response", ""),
        "agent_type": state.get("agent_type", "general"),
        "sources": state.get("sources", []),
        "artifacts": state.get("artifacts", {}),
        "thinking_steps": state.get("thinking_steps", []),
    }
