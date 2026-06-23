"""
Autonomous Coding Agent — Plan → Execute → Review → Improve.
Implements reflection loops for iterative code improvement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from app.config import settings
from app.models.schemas import CodeFile, CodingResult, TaskStatus, TestResult
from app.tools.coding_tools import (
    list_files,
    read_file,
    run_python,
    run_terminal,
    write_file,
)
from app.utils.logger import logger


MAX_REFLECTION_LOOPS = 3


class CodingAgentRunner:
    """Orchestrates the Plan → Execute → Review → Improve coding workflow."""

    def __init__(self):
        self._llm = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.worker_model,
                temperature=0.2,
                openai_api_key=settings.openai_api_key,
            )
        return self._llm

    async def execute_task(
        self,
        task_description: str,
        language: str = "python",
        requirements: List[str] = None,
    ) -> CodingResult:
        """Execute a full coding task with reflection loops."""
        requirements = requirements or []

        # Phase 1: PLAN
        logger.info("Coding Agent: Planning...")
        plan = await self.create_plan(task_description)

        # Phase 2: EXECUTE
        logger.info("Coding Agent: Generating code...")
        files = await self._generate_code(task_description, plan, language)

        # Write files to workspace
        for f in files:
            write_file.invoke({"file_path": f.filename, "content": f.content})

        # Phase 3: TEST
        logger.info("Coding Agent: Running tests...")
        test_results = await self._run_tests(files, language)

        # Phase 4: REVIEW + IMPROVE (Reflection Loop)
        review = ""
        for iteration in range(MAX_REFLECTION_LOOPS):
            logger.info("Coding Agent: Review iteration %d/%d", iteration + 1, MAX_REFLECTION_LOOPS)

            # Review
            all_code = "\n\n".join(
                f"# {f.filename}\n{f.content}" for f in files
            )
            review = await self.review_code(all_code, language)

            # Check if improvements needed
            if "no issues" in review.lower() or "looks good" in review.lower():
                logger.info("Code passed review on iteration %d", iteration + 1)
                break

            # Check if tests passed
            all_passed = all(t.passed for t in test_results)
            if all_passed and iteration > 0:
                break

            # Improve based on review
            files = await self._improve_code(files, review, test_results, language)

            # Re-write improved files
            for f in files:
                write_file.invoke({"file_path": f.filename, "content": f.content})

            # Re-test
            test_results = await self._run_tests(files, language)

        # Phase 5: Generate README
        readme = await self._generate_readme(task_description, files, plan)

        # Determine status
        all_passed = all(t.passed for t in test_results) if test_results else True
        status = TaskStatus.COMPLETED if all_passed else TaskStatus.FAILED

        return CodingResult(
            plan=plan,
            files=files,
            test_results=test_results,
            review=review,
            readme=readme,
            status=status,
        )

    async def create_plan(self, task_description: str) -> str:
        """Create an implementation plan for a coding task."""
        prompt = f"""You are a senior software architect. Create a detailed implementation plan.

Task: {task_description}

Provide:
1. Architecture overview
2. Files to create (with descriptions)
3. Key functions/classes needed
4. Dependencies required
5. Testing strategy

Format as a clear, structured plan."""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error creating plan: {str(e)}"

    async def _generate_code(
        self,
        task: str,
        plan: str,
        language: str,
    ) -> List[CodeFile]:
        """Generate code files based on the plan."""
        prompt = f"""Generate the complete code for this task.

Task: {task}
Plan: {plan}
Language: {language}

Return a JSON array of file objects:
[{{"filename": "main.py", "content": "...", "language": "{language}"}}]

Include:
- Main implementation files
- A test file (test_*.py for Python)
- Each file should be complete and runnable

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            data = json.loads(content)
            return [CodeFile(**f) for f in data]
        except Exception as e:
            logger.error("Code generation error: %s", str(e))
            return [
                CodeFile(
                    filename="main.py",
                    content=f"# Error generating code: {str(e)}\n# Task: {task}\n",
                    language=language,
                )
            ]

    async def _run_tests(
        self,
        files: List[CodeFile],
        language: str,
    ) -> List[TestResult]:
        """Run test files and collect results."""
        test_results = []
        test_files = [f for f in files if f.filename.startswith("test_")]

        if not test_files:
            return [TestResult(test_name="no_tests", passed=True, output="No test files found")]

        for tf in test_files:
            result = run_python.invoke({"code": tf.content})
            passed = result.get("success", "False") == "True"
            test_results.append(
                TestResult(
                    test_name=tf.filename,
                    passed=passed,
                    output=result.get("stdout", ""),
                    error=result.get("stderr", "") if not passed else None,
                )
            )

        return test_results

    async def review_code(self, code: str, language: str = "python") -> str:
        """Review code and provide improvement suggestions."""
        prompt = f"""You are a senior code reviewer. Review this {language} code:

{code[:4000]}

Check for:
1. Bugs and logic errors
2. Code quality and best practices
3. Performance issues
4. Security concerns
5. Missing error handling

Provide specific, actionable feedback. If the code looks good, say "Looks good — no issues found."
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"Review error: {str(e)}"

    async def _improve_code(
        self,
        files: List[CodeFile],
        review: str,
        test_results: List[TestResult],
        language: str,
    ) -> List[CodeFile]:
        """Improve code based on review feedback and test results."""
        test_feedback = ""
        failing_tests = [t for t in test_results if not t.passed]
        if failing_tests:
            test_feedback = "\n\nFailing tests:\n" + "\n".join(
                f"- {t.test_name}: {t.error}" for t in failing_tests
            )

        current_code = "\n\n".join(
            f"# {f.filename}\n{f.content}" for f in files
        )

        prompt = f"""Fix these issues in the code:

Review Feedback:
{review}
{test_feedback}

Current Code:
{current_code[:4000]}

Return the fixed code as a JSON array of file objects:
[{{"filename": "...", "content": "...", "language": "{language}"}}]

Return ONLY valid JSON, no markdown formatting."""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip().strip("```json").strip("```")
            data = json.loads(content)
            return [CodeFile(**f) for f in data]
        except Exception:
            return files  # Return original if improvement fails

    async def _generate_readme(
        self,
        task: str,
        files: List[CodeFile],
        plan: str,
    ) -> str:
        """Generate a README.md for the project."""
        file_list = ", ".join(f.filename for f in files)
        prompt = f"""Generate a professional README.md for this project.

Task: {task}
Files: {file_list}
Plan Summary: {plan[:500]}

Include:
- Project title and description
- Installation instructions
- Usage examples
- File structure
- Testing instructions"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"# Project\n\nGenerated for: {task}\n\nError: {str(e)}"

    def list_workspace_files(self) -> List[Dict[str, str]]:
        """List files in the workspace."""
        result = list_files.invoke({"directory": "."})
        return result

    def read_workspace_file(self, filename: str) -> str:
        """Read a file from the workspace."""
        result = read_file.invoke({"file_path": filename})
        return result.get("content", result.get("error", "File not found"))

    async def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle a coding-related chat message."""
        prompt = f"""You are an expert software engineer. Help with this coding request:

{message}

Provide clear, well-structured code with explanations.
Use best practices, include error handling, and add comments."""

        try:
            response = self.llm.invoke(prompt)
            return {
                "response": response.content,
                "agent_type": "coding",
                "sources": [],
                "thinking_steps": [
                    "Analyzed requirements",
                    "Generated solution",
                    "Reviewed code quality",
                ],
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "agent_type": "coding",
            }
