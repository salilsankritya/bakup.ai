"""
api/routes/recent.py
─────────────────────────────────────────────────────────────────────────────
GET    /recent           — list recent projects (with availability)
DELETE /recent/{ns}       — remove a single entry
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.recent_projects import list_projects, remove_project

router = APIRouter()


@router.get("/recent", tags=["recent"])
async def get_recent_projects():
    """Return recent projects (newest first), max 10."""
    return list_projects()


@router.delete("/recent/{namespace}", tags=["recent"])
async def delete_recent_project(namespace: str):
    """Remove a project from the recent list by namespace."""
    if not remove_project(namespace):
        raise HTTPException(status_code=404, detail="Project not found in recent list.")
    return {"status": "ok", "message": "Removed from recent list."}
