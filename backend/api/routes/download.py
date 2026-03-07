"""
api/routes/download.py
─────────────────────────────────────────────────────────────────────────────
Serves the developer preview installer with correct headers.

This route is registered in the *development* entry point (backend/main.py)
but intentionally NOT in the compiled binary (build/bakup_server.py) — the
installed application should never serve itself for download.

Response headers:
    Content-Type:              application/octet-stream
    Content-Disposition:       attachment; filename="bakup-ai-installer.exe"
    Content-Length:            <file size in bytes>
    X-Content-Type-Options:    nosniff
    Cache-Control:             no-cache
"""

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

# Resolve: download.py → routes/ → api/ → backend/ → project root → downloads/
_BACKEND_DIR   = Path(__file__).resolve().parent.parent.parent   # backend/
_PROJECT_ROOT  = _BACKEND_DIR.parent                              # bakup.ai/
_DOWNLOAD_DIR  = _PROJECT_ROOT / "downloads"
_INSTALLER_NAME = "bakup-ai-installer.exe"


@router.head("/download", tags=["download"])
@router.get("/download", tags=["download"])
async def download_preview():
    """
    Download the developer preview installer.

    GET  — streams the full .exe binary (for browser downloads).
    HEAD — returns headers only (for CDN probes and link checkers).
    """
    installer_path = _DOWNLOAD_DIR / _INSTALLER_NAME

    if not installer_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": "Installer not available. Run the build pipeline first.",
                "expected_path": str(installer_path),
            },
        )

    file_size = os.path.getsize(installer_path)

    return FileResponse(
        path=str(installer_path),
        media_type="application/octet-stream",
        filename=_INSTALLER_NAME,
        headers={
            "Content-Length":           str(file_size),
            "X-Content-Type-Options":   "nosniff",
            "Cache-Control":            "no-cache",
        },
    )
