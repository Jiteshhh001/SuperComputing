"""
Interview Coach API routes — mock interviews with feedback and scoring.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agents.interview_agent import InterviewAgentRunner
from app.models.schemas import (
    InterviewConfig,
    InterviewFeedback,
    InterviewQuestion,
    InterviewScorecard,
)
from app.utils.logger import logger

router = APIRouter()
interview_runner = InterviewAgentRunner()


@router.post("/start")
async def start_interview(config: InterviewConfig):
    """Start a new mock interview session."""
    logger.info("Starting %s interview (difficulty=%s)", config.mode, config.difficulty)
    session = await interview_runner.start_session(config)
    return session


@router.post("/answer")
async def submit_answer(
    session_id: str,
    question_id: int,
    answer: str,
):
    """Submit an answer to an interview question and get feedback."""
    feedback = await interview_runner.evaluate_answer(
        session_id=session_id,
        question_id=question_id,
        answer=answer,
    )
    return feedback


@router.get("/question/{session_id}")
async def get_next_question(session_id: str):
    """Get the next interview question."""
    question = await interview_runner.get_next_question(session_id)
    return question


@router.get("/scorecard/{session_id}", response_model=InterviewScorecard)
async def get_scorecard(session_id: str):
    """Get the final interview scorecard."""
    scorecard = await interview_runner.get_scorecard(session_id)
    return scorecard


@router.post("/hint")
async def get_hint(session_id: str, question_id: int):
    """Get a hint for the current question."""
    hint = await interview_runner.get_hint(session_id, question_id)
    return {"hint": hint}
