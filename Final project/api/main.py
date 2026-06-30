"""
═══════════════════════════════════════════════════════════════════════════════
  FastAPI Backend — REST API for the Personalized Learning Agent
  Serves the Streamlit dashboard and external integrations.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import API_TITLE, API_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="REST API for the LearnFlow AI Personalized Learning Agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy agent initialization ────────────────────────────────────────────

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from src.agent.agent import LearningAgent
        _agent = LearningAgent()
    return _agent


# ── Request/Response Models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    student_id: int
    message: str


class ChatResponse(BaseModel):
    response: str
    student_id: int


class AnalysisResponse(BaseModel):
    student_id: int
    mastery: dict
    gaps: dict
    recommendations: dict
    study_plan: dict
    progress: dict
    summary: str


class StudyPlanRequest(BaseModel):
    student_id: int
    daily_hours: float = 2.0


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "LearnFlow AI API",
        "version": API_VERSION,
        "status": "running",
        "endpoints": [
            "/api/student/{student_id}/analyze",
            "/api/student/{student_id}/mastery",
            "/api/student/{student_id}/gaps",
            "/api/student/{student_id}/recommendations",
            "/api/student/{student_id}/plan",
            "/api/student/{student_id}/progress",
            "/api/chat",
            "/api/students",
        ],
    }


@app.get("/api/students")
async def list_students():
    """Get list of available student IDs."""
    agent = get_agent()
    student_ids = agent.get_student_ids()
    return {"student_ids": student_ids[:100], "total": len(student_ids)}


@app.post("/api/student/{student_id}/analyze")
async def analyze_student(student_id: int):
    """Run full analysis pipeline for a student."""
    try:
        agent = get_agent()
        result = agent.analyze_student(student_id)
        return result
    except Exception as e:
        logger.error(f"Analysis failed for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/student/{student_id}/mastery")
async def get_mastery(student_id: int):
    """Get current mastery scores for a student."""
    try:
        agent = get_agent()
        return agent.get_mastery(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/student/{student_id}/gaps")
async def get_gaps(student_id: int, top_k: int = 5):
    """Get identified weak topics for a student."""
    try:
        agent = get_agent()
        return agent.get_gaps(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/student/{student_id}/recommendations")
async def get_recommendations(student_id: int, top_k: int = 5):
    """Get resource recommendations for a student."""
    try:
        agent = get_agent()
        return agent.get_recommendations(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/student/{student_id}/plan")
async def get_study_plan(student_id: int, request: Optional[StudyPlanRequest] = None):
    """Generate a 7-day study plan."""
    try:
        agent = get_agent()
        daily_hours = request.daily_hours if request and request.daily_hours else 2.0
        return agent.get_study_plan(student_id, daily_hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/student/{student_id}/progress")
async def get_progress(student_id: int):
    """Get progress report for a student."""
    try:
        agent = get_agent()
        return agent.get_progress(student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat with the AI agent about a student's learning."""
    try:
        agent = get_agent()
        response = agent.chat(request.student_id, request.message)
        return ChatResponse(response=response, student_id=request.student_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
