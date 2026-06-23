"""
Resume-specific tools for the Resume Tailor Agent.
- PDF Reader: extract text from resume PDFs
- Skill Extractor: NLP-based skill extraction
- ATS Analyzer: score resume against job descriptions
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from app.utils.logger import logger


@tool
def read_pdf(file_path: str) -> Dict[str, Any]:
    """Read and extract text from a PDF file.

    Args:
        file_path: Path to the PDF file to read.

    Returns:
        Dictionary with extracted text, page count, and sections.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        full_text = ""

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
            full_text += text + "\n"

        # Try to identify sections
        sections = _extract_sections(full_text)

        return {
            "text": full_text.strip(),
            "page_count": len(reader.pages),
            "pages": pages,
            "sections": sections,
        }
    except Exception as e:
        logger.error("PDF read error: %s", str(e))
        return {"text": "", "page_count": 0, "pages": [], "sections": {}, "error": str(e)}


@tool
def extract_skills(resume_text: str) -> Dict[str, List[str]]:
    """Extract skills from resume text, categorized by type.

    Args:
        resume_text: The full text of a resume.

    Returns:
        Dictionary with categorized skills.
    """
    # Technical skill patterns
    technical_patterns = [
        r"python", r"java(?:script)?", r"typescript", r"c\+\+", r"c#", r"ruby",
        r"go(?:lang)?", r"rust", r"swift", r"kotlin", r"php", r"scala",
        r"react(?:\.js)?", r"angular", r"vue(?:\.js)?", r"next\.js", r"node\.js",
        r"django", r"flask", r"fastapi", r"spring", r"express",
        r"sql", r"mongodb", r"postgresql", r"mysql", r"redis", r"elasticsearch",
        r"aws", r"azure", r"gcp", r"docker", r"kubernetes", r"terraform",
        r"git", r"ci/cd", r"jenkins", r"github actions",
        r"machine learning", r"deep learning", r"nlp", r"computer vision",
        r"tensorflow", r"pytorch", r"scikit-learn", r"pandas", r"numpy",
        r"rest(?:\s*api)?", r"graphql", r"microservices", r"agile", r"scrum",
        r"html", r"css", r"sass", r"tailwind",
        r"linux", r"unix", r"bash", r"powershell",
    ]

    soft_skill_patterns = [
        r"leadership", r"communication", r"teamwork", r"problem.?solving",
        r"project management", r"analytical", r"critical thinking",
        r"collaboration", r"mentoring", r"presentation",
    ]

    text_lower = resume_text.lower()
    technical = []
    soft = []

    for pattern in technical_patterns:
        if re.search(r"\b" + pattern + r"\b", text_lower):
            match = re.search(r"\b" + pattern + r"\b", text_lower)
            if match:
                technical.append(match.group().title())

    for pattern in soft_skill_patterns:
        if re.search(r"\b" + pattern + r"\b", text_lower):
            match = re.search(r"\b" + pattern + r"\b", text_lower)
            if match:
                soft.append(match.group().title())

    # Deduplicate
    technical = list(dict.fromkeys(technical))
    soft = list(dict.fromkeys(soft))

    return {
        "technical_skills": technical,
        "soft_skills": soft,
        "total_count": len(technical) + len(soft),
    }


@tool
def analyze_ats_score(
    resume_text: str,
    job_description: str = "",
) -> Dict[str, Any]:
    """Analyze resume for ATS (Applicant Tracking System) compatibility.

    Args:
        resume_text: The full resume text.
        job_description: Optional job description to compare against.

    Returns:
        ATS score breakdown with suggestions.
    """
    scores = {}
    details = []
    suggestions = []

    # 1. Format Score — check for standard sections
    required_sections = [
        "experience", "education", "skills", "summary", "contact",
    ]
    found_sections = []
    text_lower = resume_text.lower()

    for section in required_sections:
        if section in text_lower:
            found_sections.append(section)

    format_score = int((len(found_sections) / len(required_sections)) * 100)
    scores["format_score"] = format_score

    if format_score < 100:
        missing = set(required_sections) - set(found_sections)
        suggestions.append(f"Add missing sections: {', '.join(missing)}")

    # 2. Length check
    word_count = len(resume_text.split())
    if word_count < 200:
        details.append("Resume is too short (< 200 words)")
        suggestions.append("Add more detail to your experience and achievements")
    elif word_count > 1000:
        details.append("Resume is quite long (> 1000 words)")
        suggestions.append("Consider condensing to 1-2 pages")
    else:
        details.append(f"Good length: {word_count} words")

    # 3. Keyword Match (if JD provided)
    keyword_score = 70  # Default if no JD
    if job_description:
        jd_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", job_description.lower()))
        resume_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text_lower))
        # Filter common words
        stop_words = {
            "the", "and", "for", "with", "that", "this", "are", "was", "you",
            "have", "has", "will", "can", "from", "they", "been", "their",
            "more", "about", "would", "which", "when", "make", "than",
        }
        jd_keywords = jd_words - stop_words
        matched = jd_keywords.intersection(resume_words)
        keyword_score = int((len(matched) / max(len(jd_keywords), 1)) * 100)
        keyword_score = min(keyword_score, 100)

        missing_keywords = jd_keywords - resume_words
        if missing_keywords:
            top_missing = list(missing_keywords)[:10]
            suggestions.append(
                f"Add keywords from JD: {', '.join(top_missing)}"
            )

    scores["keyword_match"] = keyword_score

    # 4. Section completeness
    section_score = format_score
    scores["section_score"] = section_score

    # 5. Action verbs check
    action_verbs = [
        "led", "developed", "managed", "created", "implemented",
        "designed", "built", "achieved", "increased", "reduced",
        "improved", "optimized", "delivered", "launched", "spearheaded",
    ]
    found_verbs = [v for v in action_verbs if v in text_lower]
    if len(found_verbs) < 3:
        suggestions.append(
            "Use more action verbs: led, developed, achieved, implemented, etc."
        )

    # 6. Quantification check
    has_numbers = bool(re.search(r"\d+%|\$\d+|\d+\+", resume_text))
    if not has_numbers:
        suggestions.append(
            "Add quantifiable achievements (e.g., 'Increased sales by 25%')"
        )

    # Overall score
    overall = int(
        (scores["keyword_match"] * 0.4)
        + (scores["format_score"] * 0.3)
        + (scores["section_score"] * 0.3)
    )

    return {
        "overall_score": overall,
        "keyword_match": scores["keyword_match"],
        "format_score": scores["format_score"],
        "section_score": scores["section_score"],
        "details": details,
        "suggestions": suggestions,
    }


def _extract_sections(text: str) -> Dict[str, str]:
    """Extract common resume sections from text."""
    section_headers = [
        "summary", "objective", "experience", "work experience",
        "education", "skills", "technical skills", "projects",
        "certifications", "awards", "publications", "interests",
        "contact", "references",
    ]

    sections = {}
    lines = text.split("\n")
    current_section = "header"
    current_content = []

    for line in lines:
        line_stripped = line.strip().lower()
        matched_section = None

        for header in section_headers:
            if line_stripped == header or line_stripped.startswith(header + ":"):
                matched_section = header
                break

        if matched_section:
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = matched_section
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_section] = "\n".join(current_content).strip()

    return sections


# Export all tools
RESUME_TOOLS = [read_pdf, extract_skills, analyze_ats_score]
