"""
Unit tests for the Smart Career Copilot backend.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from app.tools.resume_tools import extract_skills, analyze_ats_score, _extract_sections
from app.tools.coding_tools import _validate_path, _workspace_path
from app.models.schemas import (
    AgentType, ChatMessage, ChatRequest, MessageRole,
    ATSScore, InterviewConfig, InterviewMode, CodingTask,
)
from app.utils.helpers import (
    generate_id, generate_session_id, hash_content,
    safe_filename, truncate_text, format_file_size,
)
from app.config import Settings


# ── Utility Tests ────────────────────────────────────────────


class TestHelpers:
    def test_generate_id_uniqueness(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_session_id_length(self):
        sid = generate_session_id()
        assert len(sid) == 12
        assert sid.isalnum()

    def test_hash_content(self):
        h1 = hash_content("hello")
        h2 = hash_content("hello")
        h3 = hash_content("world")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16

    def test_safe_filename(self):
        assert safe_filename("my resume (1).pdf") == "my resume 1.pdf"
        assert safe_filename("../../etc/passwd") == "etcpasswd"
        assert safe_filename("") == "unnamed_file"

    def test_truncate_text(self):
        assert truncate_text("short", 100) == "short"
        assert truncate_text("a" * 200, 50) == "a" * 47 + "..."

    def test_format_file_size(self):
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1048576) == "1.0 MB"


# ── Schema Tests ─────────────────────────────────────────────


class TestSchemas:
    def test_chat_request_validation(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.session_id is None
        assert req.agent_type is None

    def test_chat_request_min_length(self):
        with pytest.raises(Exception):
            ChatRequest(message="")

    def test_agent_type_enum(self):
        assert AgentType.RESUME.value == "resume"
        assert AgentType.RESEARCH.value == "research"
        assert AgentType.CODING.value == "coding"
        assert AgentType.INTERVIEW.value == "interview"

    def test_ats_score_bounds(self):
        score = ATSScore(
            overall_score=85, keyword_match=90,
            format_score=80, section_score=85,
        )
        assert 0 <= score.overall_score <= 100

    def test_interview_config(self):
        config = InterviewConfig(
            mode=InterviewMode.TECHNICAL,
            difficulty="hard",
            topics=["python", "algorithms"],
            num_questions=5,
        )
        assert config.mode == InterviewMode.TECHNICAL
        assert config.num_questions == 5


# ── Tool Tests ───────────────────────────────────────────────


class TestResumeTools:
    def test_extract_skills_python(self):
        text = "Experienced Python developer with React and Docker skills. Strong leadership."
        result = extract_skills.invoke({"resume_text": text})
        assert "Python" in result["technical_skills"]
        assert "React" in result["technical_skills"]
        assert "Docker" in result["technical_skills"]
        assert "Leadership" in result["soft_skills"]

    def test_extract_skills_empty(self):
        result = extract_skills.invoke({"resume_text": "no relevant content here"})
        assert isinstance(result["technical_skills"], list)
        assert isinstance(result["soft_skills"], list)

    def test_ats_analyzer_with_jd(self):
        resume = "Experience in Python development. Education at MIT. Skills: Python, React. Summary: Senior developer."
        jd = "Looking for a Python developer with React experience"
        result = analyze_ats_score.invoke({
            "resume_text": resume,
            "job_description": jd,
        })
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 100

    def test_ats_analyzer_no_jd(self):
        resume = "Experience section. Education section. Skills: Python."
        result = analyze_ats_score.invoke({"resume_text": resume})
        assert result["keyword_match"] == 70  # Default when no JD

    def test_extract_sections(self):
        text = "John Doe\n\nExperience\nWorked at Google\n\nEducation\nMIT CS\n\nSkills\nPython"
        sections = _extract_sections(text)
        assert "experience" in sections
        assert "education" in sections
        assert "skills" in sections


# ── Config Tests ─────────────────────────────────────────────


class TestConfig:
    def test_default_settings(self):
        s = Settings(openai_api_key="test", tavily_api_key="test")
        assert s.orchestrator_model == "gpt-4o"
        assert s.worker_model == "gpt-4o-mini"
        assert s.code_execution_timeout == 30

    def test_cors_origins_parsing(self):
        s = Settings(
            openai_api_key="test",
            tavily_api_key="test",
            cors_origins='["http://localhost:3000"]',
        )
        assert "http://localhost:3000" in s.cors_origin_list


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
