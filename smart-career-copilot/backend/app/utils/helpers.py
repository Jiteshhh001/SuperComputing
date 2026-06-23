"""
Shared utility functions.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def generate_session_id() -> str:
    """Generate a short session identifier."""
    return uuid.uuid4().hex[:12]


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def hash_content(content: str) -> str:
    """Create a SHA-256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def safe_filename(filename: str) -> str:
    """Sanitize a filename for safe filesystem use."""
    # Prevent traversal patterns becoming leading dots
    filename = filename.replace("/", "").replace("\\", "").lstrip(".")
    keepchars = (" ", ".", "_", "-")
    cleaned = "".join(c for c in filename if c.isalnum() or c in keepchars).rstrip()
    return cleaned or "unnamed_file"


def ensure_path(path: str | Path) -> Path:
    """Ensure a directory path exists and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
