"""
Coding Agent API routes — plan, generate code, run tests, and review.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.agents.coding_agent import CodingAgentRunner
from app.models.schemas import CodingResult, CodingTask
from app.utils.logger import logger

router = APIRouter()
coding_runner = CodingAgentRunner()


@router.post("/execute", response_model=CodingResult)
async def execute_coding_task(request: CodingTask):
    """Execute a full coding task: plan → code → test → review."""
    logger.info("Coding task: %s", request.task_description[:100])
    result = await coding_runner.execute_task(
        task_description=request.task_description,
        language=request.language,
        requirements=request.requirements,
    )
    return result


@router.post("/plan")
async def create_plan(task_description: str):
    """Create an implementation plan for a coding task."""
    plan = await coding_runner.create_plan(task_description)
    return {"plan": plan}


@router.post("/review")
async def review_code(code: str, language: str = "python"):
    """Review code and provide suggestions."""
    review = await coding_runner.review_code(code, language)
    return {"review": review}


@router.get("/files")
async def list_workspace_files():
    """List files in the coding workspace."""
    files = coding_runner.list_workspace_files()
    return {"files": files}


@router.get("/files/{filename:path}")
async def read_workspace_file(filename: str):
    """Read a file from the coding workspace."""
    content = coding_runner.read_workspace_file(filename)
    return {"filename": filename, "content": content}
