"""
Interview Coach Agent — mock interviews with feedback and scoring.
Supports: technical, behavioral, and resume-based interviews.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import settings
from app.models.schemas import (
    InterviewConfig,
    InterviewFeedback,
    InterviewMode,
    InterviewQuestion,
    InterviewScorecard,
)
from app.utils.helpers import generate_session_id
from app.utils.logger import logger


class InterviewAgentRunner:
    """Manages mock interview sessions with AI-powered feedback."""

    def __init__(self):
        self._llm = None
        self._sessions: Dict[str, Dict[str, Any]] = {}

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.worker_model,
                temperature=0.4,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    async def start_session(self, config: InterviewConfig) -> Dict[str, Any]:
        """Start a new interview session and generate questions."""
        session_id = generate_session_id()

        # Generate questions
        questions = await self._generate_questions(config)

        self._sessions[session_id] = {
            "config": config.model_dump(),
            "questions": questions,
            "current_index": 0,
            "answers": [],
            "feedback": [],
        }

        return {
            "session_id": session_id,
            "total_questions": len(questions),
            "mode": config.mode.value,
            "difficulty": config.difficulty,
            "first_question": questions[0].model_dump() if questions else None,
        }

    async def _generate_questions(
        self,
        config: InterviewConfig,
    ) -> List[InterviewQuestion]:
        """Generate interview questions based on configuration."""
        resume_context = ""
        if config.resume_context:
            resume_context = f"\n\nCandidate's Resume:\n{config.resume_context[:1500]}"

        topic_context = ""
        if config.topics:
            topic_context = f"\nFocus on topics: {', '.join(config.topics)}"

        mode_descriptions = {
            InterviewMode.TECHNICAL: "technical coding and system design questions",
            InterviewMode.BEHAVIORAL: "behavioral and situational questions using the STAR method",
            InterviewMode.RESUME_BASED: "questions based on the candidate's resume and experience",
        }

        prompt = f"""Generate {config.num_questions} {mode_descriptions.get(config.mode, 'interview')} questions.

Difficulty: {config.difficulty}
{topic_context}
{resume_context}

Return a JSON array of question objects:
[{{
    "id": 1,
    "question": "...",
    "category": "...",
    "difficulty": "{config.difficulty}",
    "hints": ["hint1", "hint2"]
}}]

Return ONLY valid JSON, no markdown."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            data = json.loads(content)
            return [InterviewQuestion(**q) for q in data]
        except Exception as e:
            logger.error("Question generation error: %s", str(e))
            return [
                InterviewQuestion(
                    id=1,
                    question="Tell me about your most challenging project.",
                    category="behavioral",
                    difficulty=config.difficulty,
                    hints=["Use the STAR method", "Focus on your specific contributions"],
                )
            ]

    async def evaluate_answer(
        self,
        session_id: str,
        question_id: int,
        answer: str,
    ) -> InterviewFeedback:
        """Evaluate a candidate's answer and provide feedback."""
        session = self._sessions.get(session_id)
        if not session:
            return InterviewFeedback(
                question_id=question_id,
                score=0,
                strengths=["Session not found"],
            )

        # Find the question
        question = None
        for q in session["questions"]:
            if q.id == question_id:
                question = q
                break

        if not question:
            return InterviewFeedback(
                question_id=question_id,
                score=0,
                strengths=["Question not found"],
            )

        prompt = f"""You are an expert interviewer. Evaluate this answer:

Question: {question.question}
Category: {question.category}
Difficulty: {question.difficulty}

Candidate's Answer: {answer}

Provide detailed evaluation as JSON:
{{
    "question_id": {question_id},
    "score": <1-10>,
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "sample_answer": "A strong example answer..."
}}

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            data = json.loads(content)
            feedback = InterviewFeedback(**data)
        except Exception as e:
            logger.error("Evaluation error: %s", str(e))
            feedback = InterviewFeedback(
                question_id=question_id,
                score=5,
                strengths=["Answer provided"],
                improvements=["Could not evaluate — please try again"],
            )

        # Store feedback
        session["feedback"].append(feedback)
        session["answers"].append({"question_id": question_id, "answer": answer})
        session["current_index"] += 1

        return feedback

    async def get_next_question(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the next question in the session."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        idx = session["current_index"]
        questions = session["questions"]

        if idx >= len(questions):
            return {
                "completed": True,
                "message": "Interview complete! View your scorecard.",
            }

        return {
            "completed": False,
            "question": questions[idx].model_dump(),
            "progress": f"{idx + 1}/{len(questions)}",
        }

    async def get_scorecard(self, session_id: str) -> InterviewScorecard:
        """Generate the final interview scorecard."""
        session = self._sessions.get(session_id)
        if not session:
            return InterviewScorecard(
                overall_score=0,
                total_questions=0,
                questions_answered=0,
            )

        feedback_list = session["feedback"]
        questions = session["questions"]

        if not feedback_list:
            return InterviewScorecard(
                overall_score=0,
                total_questions=len(questions),
                questions_answered=0,
            )

        # Calculate scores by category
        category_scores: Dict[str, List[int]] = {}
        for fb in feedback_list:
            q = next((q for q in questions if q.id == fb.question_id), None)
            if q:
                cat = q.category
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(fb.score)

        avg_category = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        overall = sum(fb.score for fb in feedback_list) / len(feedback_list)

        # Generate improvement plan
        improvement_plan = await self._generate_improvement_plan(
            feedback_list, questions
        )

        return InterviewScorecard(
            overall_score=round(overall, 1),
            total_questions=len(questions),
            questions_answered=len(feedback_list),
            category_scores=avg_category,
            feedback=feedback_list,
            improvement_plan=improvement_plan,
        )

    async def _generate_improvement_plan(
        self,
        feedback: List[InterviewFeedback],
        questions: List[InterviewQuestion],
    ) -> List[str]:
        """Generate a personalized improvement plan based on interview results."""
        feedback_summary = "\n".join(
            f"Q{fb.question_id} (Score: {fb.score}/10): Improvements: {', '.join(fb.improvements)}"
            for fb in feedback
        )

        prompt = f"""Based on this interview performance, create a targeted improvement plan:

{feedback_summary}

Provide 5-7 specific, actionable improvement recommendations.
Return a JSON array of strings: ["recommendation1", "recommendation2", ...]

Return ONLY valid JSON."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            return json.loads(content)
        except Exception:
            return [
                "Practice answering with the STAR method",
                "Prepare specific examples from past experience",
                "Work on quantifying your achievements",
            ]

    async def get_hint(self, session_id: str, question_id: int) -> str:
        """Get a hint for a question."""
        session = self._sessions.get(session_id)
        if not session:
            return "Session not found."

        question = next(
            (q for q in session["questions"] if q.id == question_id), None
        )
        if not question:
            return "Question not found."

        if question.hints:
            return question.hints[0]
        return "Think about a specific example from your experience."

    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle an interview-related chat message."""
        prompt = f"""You are an expert interview coach. Help with this request:

{message}

Provide practical, actionable interview advice.
Include example answers when appropriate.
Use the STAR method for behavioral questions."""

        try:
            response = self.llm.invoke(prompt)
            return {
                "response": response.content,
                "agent_type": "interview",
                "sources": [],
                "thinking_steps": [
                    "Analyzed interview request",
                    "Generated coaching advice",
                ],
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "agent_type": "interview",
            }
