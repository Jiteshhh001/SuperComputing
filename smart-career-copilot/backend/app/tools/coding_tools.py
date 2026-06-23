"""
Coding-specific tools for the Autonomous Coding Agent.
- File Reader: read files from sandboxed workspace
- File Writer: write/create files in workspace
- Python Runner: execute Python code safely
- Terminal Executor: run shell commands in sandbox
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool

from app.config import settings
from app.utils.logger import logger


def _workspace_path() -> Path:
    """Get the absolute workspace path."""
    return Path(settings.workspace_dir).resolve()


def _validate_path(file_path: str) -> Path:
    """Validate that a file path is within the workspace."""
    workspace = _workspace_path()
    workspace.mkdir(parents=True, exist_ok=True)
    resolved = (workspace / file_path).resolve()

    if not str(resolved).startswith(str(workspace)):
        raise ValueError(f"Path escapes workspace: {file_path}")
    return resolved


@tool
def read_file(file_path: str) -> Dict[str, str]:
    """Read a file from the coding workspace.

    Args:
        file_path: Relative path to the file within the workspace.

    Returns:
        Dictionary with filename, content, and line count.
    """
    try:
        resolved = _validate_path(file_path)
        if not resolved.exists():
            return {"error": f"File not found: {file_path}", "content": ""}

        content = resolved.read_text(encoding="utf-8")
        return {
            "filename": file_path,
            "content": content,
            "lines": str(content.count("\n") + 1),
            "size": str(resolved.stat().st_size),
        }
    except Exception as e:
        return {"error": str(e), "content": ""}


@tool
def write_file(file_path: str, content: str) -> Dict[str, str]:
    """Write content to a file in the coding workspace.

    Args:
        file_path: Relative path to the file within the workspace.
        content: The content to write to the file.

    Returns:
        Status of the write operation.
    """
    try:
        resolved = _validate_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        logger.info("File written: %s (%d bytes)", file_path, len(content))
        return {
            "status": "success",
            "filename": file_path,
            "size": str(len(content)),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def run_python(code: str) -> Dict[str, str]:
    """Execute Python code in a sandboxed environment.

    Args:
        code: Python code to execute.

    Returns:
        Dictionary with stdout, stderr, and return code.
    """
    try:
        # Write code to a temp file in workspace
        workspace = _workspace_path()
        workspace.mkdir(parents=True, exist_ok=True)
        temp_file = workspace / "_temp_exec.py"
        temp_file.write_text(code, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=settings.code_execution_timeout,
            cwd=str(workspace),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        # Clean up temp file
        temp_file.unlink(missing_ok=True)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": str(result.returncode),
            "success": str(result.returncode == 0),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {settings.code_execution_timeout}s",
            "return_code": "-1",
            "success": "False",
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": "-1",
            "success": "False",
        }


@tool
def run_terminal(command: str) -> Dict[str, str]:
    """Execute a shell command in the sandboxed workspace.

    Args:
        command: Shell command to execute.

    Returns:
        Dictionary with stdout, stderr, and return code.
    """
    # Block dangerous commands
    blocked = ["rm -rf /", "sudo", "chmod 777", "curl", "wget", "ssh"]
    for b in blocked:
        if b in command.lower():
            return {
                "stdout": "",
                "stderr": f"Blocked command: {b}",
                "return_code": "-1",
                "success": "False",
            }

    try:
        workspace = _workspace_path()
        workspace.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=settings.code_execution_timeout,
            cwd=str(workspace),
        )

        return {
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
            "return_code": str(result.returncode),
            "success": str(result.returncode == 0),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {settings.code_execution_timeout}s",
            "return_code": "-1",
            "success": "False",
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": "-1",
            "success": "False",
        }


@tool
def list_files(directory: str = ".") -> List[Dict[str, str]]:
    """List files in the coding workspace directory.

    Args:
        directory: Relative directory path within the workspace.

    Returns:
        List of file entries with name, type, and size.
    """
    try:
        resolved = _validate_path(directory)
        if not resolved.is_dir():
            return [{"error": f"Not a directory: {directory}"}]

        entries = []
        for item in sorted(resolved.iterdir()):
            if item.name.startswith("_temp_"):
                continue
            entries.append({
                "name": str(item.relative_to(_workspace_path())),
                "type": "directory" if item.is_dir() else "file",
                "size": str(item.stat().st_size) if item.is_file() else "",
            })
        return entries
    except Exception as e:
        return [{"error": str(e)}]


# Export all tools
CODING_TOOLS = [read_file, write_file, run_python, run_terminal, list_files]
