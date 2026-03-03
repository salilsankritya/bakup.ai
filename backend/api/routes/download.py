"""
api/routes/download.py
─────────────────────────────────────────────────────────────────────────────
Serves the developer preview ZIP with correct headers.

Returns:
    Content-Type: application/zip
    Content-Disposition: attachment; filename="bakup-preview-0.1.0.zip"
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter()

# Resolve path relative to backend/ → project root → downloads/
_DOWNLOAD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "downloads"
_ZIP_NAME = "bakup-preview-0.1.0.zip"


@router.get("/download", tags=["download"])
async def download_preview():
    """
    Download the developer preview archive.
    Serves the ZIP with Content-Type: application/zip and
    Content-Disposition: attachment headers.
    """
    zip_path = _DOWNLOAD_DIR / _ZIP_NAME

    if not zip_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Download not available. Build the preview first."},
        )

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=_ZIP_NAME,
        headers={
            "Content-Disposition": f'attachment; filename="{_ZIP_NAME}"',
        },
    )
