"""
FastAPI application entry point.
Configures CORS, mounts routers, and manages application lifecycle.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import init_db
from app.models.schemas import HealthResponse
from app.utils.logger import logger

from app.api.routes.chat import router as chat_router
from app.api.routes.resume import router as resume_router
from app.api.routes.research import router as research_router
from app.api.routes.coding import router as coding_router
from app.api.routes.interview import router as interview_router
from app.api.routes.files import router as files_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("🚀 Starting %s...", settings.app_name)
    settings.ensure_directories()
    init_db()
    logger.info("✅ Database initialized")
    logger.info("✅ Application ready")
    yield
    logger.info("👋 Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description="A multi-agent AI system for career development",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ─────────────────────────────────────────────

uploads_path = Path("./uploads")
uploads_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

# ── Routers ──────────────────────────────────────────────────

app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(resume_router, prefix="/api/resume", tags=["Resume Agent"])
app.include_router(research_router, prefix="/api/research", tags=["Research Agent"])
app.include_router(coding_router, prefix="/api/coding", tags=["Coding Agent"])
app.include_router(interview_router, prefix="/api/interview", tags=["Interview Agent"])
app.include_router(files_router, prefix="/api/files", tags=["Files"])


# ── Health Check ─────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Application health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services={
            "database": "connected",
            "vectorstore": "ready",
            "llm": "configured" if settings.openai_api_key else "missing_key",
        },
    )


@app.get("/")
async def root():
    """Root endpoint redirect."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "docs": "/docs",
        "health": "/api/health",
    }
