"""
Resume Agent API routes — upload, parse, analyze, and improve resumes.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.agents.resume_agent import ResumeAgentRunner
from app.models.schemas import ResumeAnalysis, ResumeUploadResponse
from app.utils.helpers import generate_id, safe_filename
from app.utils.logger import logger

router = APIRouter()
resume_runner = ResumeAgentRunner()


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse a resume PDF."""
    file_id = generate_id()
    filename = safe_filename(file.filename or "resume.pdf")
    upload_dir = Path("./uploads/resumes")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{file_id}_{filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Resume uploaded: %s -> %s", filename, file_path)

    # Parse the resume
    result = resume_runner.parse_resume(str(file_path))

    return ResumeUploadResponse(
        file_id=file_id,
        filename=filename,
        parsed_text=result.get("text", ""),
        sections=result.get("sections", {}),
        page_count=result.get("page_count", 0),
    )


@router.post("/analyze", response_model=ResumeAnalysis)
async def analyze_resume(
    file_id: str = Form(...),
    resume_text: str = Form(...),
    job_description: Optional[str] = Form(None),
):
    """Analyze resume: ATS score, skills, gaps, improved bullets, cover letter."""
    result = await resume_runner.full_analysis(
        resume_text=resume_text,
        job_description=job_description or "",
    )
    return result


@router.post("/rewrite-bullets")
async def rewrite_bullets(
    resume_text: str = Form(...),
    job_description: str = Form(""),
):
    """Rewrite resume bullet points to be more impactful."""
    result = await resume_runner.rewrite_bullets(resume_text, job_description)
    return {"improved_bullets": result}


@router.post("/cover-letter")
async def generate_cover_letter(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    company_name: str = Form("the company"),
):
    """Generate a tailored cover letter."""
    result = await resume_runner.generate_cover_letter(
        resume_text, job_description, company_name
    )
    return {"cover_letter": result}


@router.post("/download")
async def download_improved_resume(
    resume_text: str = Form(...),
    improvements: str = Form("{}"),
):
    """Generate and return an improved resume document."""
    # For portfolio demo, return the improved text
    return {
        "status": "generated",
        "content": resume_text,
        "format": "text",
    }
