"""
api/routes/download.py
─────────────────────────────────────────────────────────────────────────────
Serves the developer preview installer with correct headers.

Returns:
    Content-Type: application/octet-stream
    Content-Disposition: attachment; filename="bakup-ai-installer.exe"
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

# Resolve path relative to backend/ → project root → downloads/
_DOWNLOAD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "downloads"
_INSTALLER_NAME = "bakup-ai-installer.exe"


@router.get("/download", tags=["download"])
async def download_preview():
    """
    Download the developer preview installer.
    Serves the Windows installer with proper headers.
    """
    installer_path = _DOWNLOAD_DIR / _INSTALLER_NAME

    if not installer_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Installer not available. Run the build pipeline first."},
        )

    return FileResponse(
        path=str(installer_path),
        media_type="application/octet-stream",
        filename=_INSTALLER_NAME,
        headers={
            "Content-Disposition": f'attachment; filename="{_INSTALLER_NAME}"',
        },
    )
