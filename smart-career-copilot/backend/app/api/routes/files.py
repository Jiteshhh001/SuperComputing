"""
File upload/download API routes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.models.schemas import FileUploadResponse
from app.utils.helpers import format_file_size, generate_id, safe_filename
from app.utils.logger import logger

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the server."""
    file_id = generate_id()
    filename = safe_filename(file.filename or "uploaded_file")
    upload_dir = Path("./uploads/general")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{file_id}_{filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = file_path.stat().st_size
    logger.info("File uploaded: %s (%s)", filename, format_file_size(file_size))

    return FileUploadResponse(
        file_id=file_id,
        filename=filename,
        size=file_size,
        content_type=file.content_type or "application/octet-stream",
        upload_path=str(file_path),
    )


@router.get("/list")
async def list_uploaded_files():
    """List all uploaded files."""
    upload_dir = Path("./uploads")
    files = []
    if upload_dir.exists():
        for p in upload_dir.rglob("*"):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    "size_formatted": format_file_size(p.stat().st_size),
                })
    return {"files": files, "total": len(files)}


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    upload_dir = Path("./uploads")
    for p in upload_dir.rglob(f"{file_id}_*"):
        p.unlink()
        logger.info("File deleted: %s", p.name)
        return {"status": "deleted", "file_id": file_id}
    return {"status": "not_found", "file_id": file_id}
