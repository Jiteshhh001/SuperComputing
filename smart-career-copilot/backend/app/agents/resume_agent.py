"""
Resume Tailor Agent — full resume analysis pipeline.
Handles: parse, extract skills, ATS score, JD comparison, rewriting, cover letters.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.document_processor import DocumentProcessor
from app.rag.vectorstore import vectorstore_manager
from app.tools.resume_tools import analyze_ats_score, extract_skills, read_pdf
from app.models.schemas import ATSScore, ResumeAnalysis, SkillGap, SkillInfo
from app.utils.logger import logger


RESUME_COLLECTION = "resume_knowledge"


class ResumeAgentRunner:
    """Orchestrates resume analysis tasks."""

    def __init__(self):
        self._llm = None
        self.doc_processor = DocumentProcessor()

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.worker_model,
                temperature=0.3,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume PDF and store in RAG."""
        result = read_pdf.invoke({"file_path": file_path})

        # Store in vector DB for context
        if result.get("text"):
            docs = self.doc_processor.process_text(
                result["text"],
                metadata={"source": file_path, "type": "resume"},
            )
            vectorstore_manager.add_documents(RESUME_COLLECTION, docs)

        return result

    async def full_analysis(
        self,
        resume_text: str,
        job_description: str = "",
    ) -> ResumeAnalysis:
        """Run full resume analysis pipeline."""
        # Step 1: Extract skills
        skills_result = extract_skills.invoke({"resume_text": resume_text})
        skills = [
            SkillInfo(name=s, category="Technical")
            for s in skills_result.get("technical_skills", [])
        ] + [
            SkillInfo(name=s, category="Soft Skill")
            for s in skills_result.get("soft_skills", [])
        ]

        # Step 2: ATS Score
        ats_result = analyze_ats_score.invoke({
            "resume_text": resume_text,
            "job_description": job_description,
        })
        ats_score = ATSScore(
            overall_score=ats_result.get("overall_score", 0),
            keyword_match=ats_result.get("keyword_match", 0),
            format_score=ats_result.get("format_score", 0),
            section_score=ats_result.get("section_score", 0),
            details=ats_result.get("details", []),
            suggestions=ats_result.get("suggestions", []),
        )

        # Step 3: Skill Gap Analysis (if JD provided)
        skill_gap = None
        if job_description:
            skill_gap = await self._analyze_skill_gap(
                resume_text, job_description, skills_result
            )

        # Step 4: Rewrite bullets
        improved_bullets = await self.rewrite_bullets(resume_text, job_description)

        # Step 5: Generate cover letter (if JD provided)
        cover_letter = None
        if job_description:
            cover_letter = await self.generate_cover_letter(
                resume_text, job_description, "the company"
            )

        return ResumeAnalysis(
            ats_score=ats_score,
            skills=skills,
            skill_gap=skill_gap,
            improved_bullets=improved_bullets,
            cover_letter=cover_letter,
        )

    async def _analyze_skill_gap(
        self,
        resume_text: str,
        job_description: str,
        skills_result: Dict,
    ) -> SkillGap:
        """Compare resume skills against job description requirements."""
        prompt = f"""Analyze the skill gap between this resume and job description.

Resume Skills Found: {json.dumps(skills_result)}

Job Description:
{job_description[:2000]}

Return a JSON object with:
- "required_skills": list of skills required by the job
- "present_skills": list of matching skills from the resume
- "missing_skills": list of skills missing from the resume
- "match_percentage": float 0-100
- "recommendations": list of actionable recommendations

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self.llm.invoke(prompt)
            data = json.loads(response.content.strip().strip("```json").strip("```"))
            return SkillGap(**data)
        except Exception as e:
            logger.error("Skill gap analysis error: %s", str(e))
            return SkillGap(
                required_skills=[],
                present_skills=list(
                    skills_result.get("technical_skills", [])[:5]
                ),
                missing_skills=[],
                match_percentage=0.0,
            )

    async def rewrite_bullets(
        self,
        resume_text: str,
        job_description: str = "",
    ) -> List[Dict[str, str]]:
        """Rewrite resume bullet points for more impact."""
        jd_context = f"\nTarget Job Description:\n{job_description[:1000]}" if job_description else ""

        prompt = f"""You are an expert resume writer. Improve these resume bullet points to be more impactful.
Use strong action verbs, quantify achievements where possible, and align with the target role.

Resume:
{resume_text[:3000]}
{jd_context}

Return a JSON array of objects with "original" and "improved" keys.
Return ONLY valid JSON, no markdown formatting. Return at most 8 bullets."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            return json.loads(content)
        except Exception as e:
            logger.error("Bullet rewrite error: %s", str(e))
            return []

    async def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        company_name: str = "the company",
    ) -> str:
        """Generate a tailored cover letter."""
        prompt = f"""Write a professional, compelling cover letter based on this resume and job description.

Resume:
{resume_text[:2000]}

Job Description:
{job_description[:1500]}

Company: {company_name}

Write a complete cover letter that:
1. Opens with enthusiasm for the role
2. Highlights 2-3 most relevant experiences
3. Shows knowledge of the company/industry
4. Closes with a strong call to action
5. Is concise (3-4 paragraphs)"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error("Cover letter error: %s", str(e))
            return "Error generating cover letter. Please try again."

    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle a resume-related chat message."""
        context = context or {}

        # Check if there's resume context in RAG
        rag_context = ""
        try:
            docs = vectorstore_manager.similarity_search(
                RESUME_COLLECTION, message, k=3
            )
            if docs:
                rag_context = "\n\nRelevant resume context:\n" + "\n".join(
                    d.page_content[:500] for d in docs
                )
        except Exception:
            pass

        frontend_context = ""
        if "resume_analysis" in context:
            frontend_context = f"\n\nPre-calculated ATS Analysis:\n{json.dumps(context['resume_analysis'], indent=2)}"
        elif "resume_text" in context:
            frontend_context = f"\n\nUploaded Resume Full Text:\n{context['resume_text'][:3000]}"

        prompt = f"""You are an expert career advisor and resume consultant.
Help the user with their resume-related request.
{rag_context}{frontend_context}

User request: {message}

Provide detailed, actionable advice. Format your response with clear headings and bullet points."""

        try:
            response = self.llm.invoke(prompt)
            return {
                "response": response.content,
                "agent_type": "resume",
                "sources": [],
                "thinking_steps": ["Analyzed resume context", "Generated advice"],
            }
        except Exception as e:
            return {
                "response": f"I encountered an error: {str(e)}",
                "agent_type": "resume",
            }
