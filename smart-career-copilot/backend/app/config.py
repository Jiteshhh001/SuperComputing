"""
Application configuration using pydantic-settings.
All configuration is loaded from environment variables or .env file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Smart Career Copilot."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────
    openai_api_key: str = ""
    tavily_api_key: str = ""

    # ── LLM Models ───────────────────────────────────────────
    orchestrator_model: str = "gpt-4o"
    worker_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Database ─────────────────────────────────────────────
    sqlite_url: str = "sqlite+aiosqlite:///./data/career_copilot.db"
    chroma_persist_dir: str = "./data/chroma"

    # ── Application ──────────────────────────────────────────
    app_name: str = "Smart Career Copilot"
    debug: bool = True
    log_level: str = "INFO"
    cors_origins: str = '["http://localhost:5173","http://localhost:3000"]'

    # ── Security ─────────────────────────────────────────────
    secret_key: str = "change-this-to-a-random-secret-key"

    # ── Coding Agent ─────────────────────────────────────────
    workspace_dir: str = "./workspace"
    code_execution_timeout: int = 30

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        try:
            return json.loads(self.cors_origins)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:5173", "http://localhost:3000"]

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)
        Path("./uploads").mkdir(parents=True, exist_ok=True)


settings = Settings()
