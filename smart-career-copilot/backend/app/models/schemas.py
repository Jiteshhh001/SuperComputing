"""
Pydantic schemas for request/response validation across all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────


class AgentType(str, Enum):
    RESUME = "resume"
    RESEARCH = "research"
    CODING = "coding"
    INTERVIEW = "interview"
    GENERAL = "general"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class InterviewMode(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    RESUME_BASED = "resume_based"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Chat Schemas ──────────────────────────────────────────────


class ChatMessage(BaseModel):
    id: str = Field(default="")
    role: MessageRole
    content: str
    agent_type: Optional[AgentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    agent_type: Optional[AgentType] = None
    attachments: List[Any] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    message: ChatMessage
    session_id: str
    agent_used: AgentType
    thinking_steps: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    type: str  # "token", "thinking", "source", "artifact", "done", "error"
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ── Resume Schemas ────────────────────────────────────────────


class ResumeUploadResponse(BaseModel):
    file_id: str
    filename: str
    parsed_text: str
    sections: Dict[str, str] = Field(default_factory=dict)
    page_count: int = 0


class SkillInfo(BaseModel):
    name: str
    category: str = ""
    proficiency: Optional[str] = None


class ATSScore(BaseModel):
    overall_score: int = Field(..., ge=0, le=100)
    keyword_match: int = Field(..., ge=0, le=100)
    format_score: int = Field(..., ge=0, le=100)
    section_score: int = Field(..., ge=0, le=100)
    details: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class SkillGap(BaseModel):
    required_skills: List[str]
    present_skills: List[str]
    missing_skills: List[str]
    match_percentage: float
    recommendations: List[str] = Field(default_factory=list)


class ResumeAnalysis(BaseModel):
    ats_score: ATSScore
    skills: List[SkillInfo]
    skill_gap: Optional[SkillGap] = None
    improved_bullets: List[Dict[str, str]] = Field(default_factory=list)
    cover_letter: Optional[str] = None


# ── Research Schemas ──────────────────────────────────────────


class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    sources: List[str] = Field(default_factory=lambda: ["web", "arxiv"])
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchSource(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source_type: str = "web"
    relevance_score: float = 0.0


class ResearchReport(BaseModel):
    query: str
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    sources: List[ResearchSource] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)


# ── Coding Schemas ────────────────────────────────────────────


class CodingTask(BaseModel):
    task_description: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="python")
    requirements: List[str] = Field(default_factory=list)


class CodeFile(BaseModel):
    filename: str
    content: str
    language: str = "python"


class TestResult(BaseModel):
    test_name: str
    passed: bool
    output: str = ""
    error: Optional[str] = None


class CodingResult(BaseModel):
    plan: str
    files: List[CodeFile] = Field(default_factory=list)
    test_results: List[TestResult] = Field(default_factory=list)
    review: str = ""
    readme: str = ""
    status: TaskStatus = TaskStatus.COMPLETED


# ── Interview Schemas ─────────────────────────────────────────


class InterviewConfig(BaseModel):
    mode: InterviewMode = InterviewMode.TECHNICAL
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    topics: List[str] = Field(default_factory=list)
    num_questions: int = Field(default=5, ge=1, le=20)
    resume_context: Optional[str] = None


class InterviewQuestion(BaseModel):
    id: int
    question: str
    category: str
    difficulty: str
    hints: List[str] = Field(default_factory=list)


class InterviewFeedback(BaseModel):
    question_id: int
    score: int = Field(..., ge=0, le=10)
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    sample_answer: str = ""


class InterviewScorecard(BaseModel):
    overall_score: float = Field(..., ge=0, le=10)
    total_questions: int
    questions_answered: int
    category_scores: Dict[str, float] = Field(default_factory=dict)
    feedback: List[InterviewFeedback] = Field(default_factory=list)
    improvement_plan: List[str] = Field(default_factory=list)


# ── Session Schemas ───────────────────────────────────────────


class SessionInfo(BaseModel):
    session_id: str
    agent_type: AgentType
    title: str = "New Conversation"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    message_count: int = 0


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int


# ── File Schemas ──────────────────────────────────────────────


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int
    content_type: str
    upload_path: str


# ── Health Check ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
