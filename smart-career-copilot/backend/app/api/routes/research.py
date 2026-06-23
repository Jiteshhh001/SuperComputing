"""
Research Agent API routes — web search, summarization, and report generation.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agents.research_agent import ResearchAgentRunner
from app.models.schemas import ResearchQuery, ResearchReport
from app.utils.logger import logger

router = APIRouter()
research_runner = ResearchAgentRunner()


@router.post("/query", response_model=ResearchReport)
async def research_query(request: ResearchQuery):
    """Execute a deep research query across multiple sources."""
    logger.info("Research query: %s (depth=%s)", request.query, request.depth)
    result = await research_runner.research(
        query=request.query,
        depth=request.depth,
        sources=request.sources,
        max_results=request.max_results,
    )
    return result


@router.post("/summarize")
async def summarize_url(url: str):
    """Summarize content from a URL."""
    result = await research_runner.summarize_url(url)
    return {"url": url, "summary": result}


@router.post("/export")
async def export_report(report: ResearchReport):
    """Export a research report as markdown."""
    markdown = research_runner.export_to_markdown(report)
    return {"format": "markdown", "content": markdown}
