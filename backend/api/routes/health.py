"""
api/routes/health.py — Liveness, readiness, and LLM status probe.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["ops"])
async def health() -> dict:
    """
    Returns 200 when the service is up.
    Includes LLM status so the UI can show a single accurate indicator.
    """
    from core.llm.llm_service import get_llm_service
    svc = get_llm_service()
    llm_status, llm_message = svc.health_check()

    return {
        "status":      "ok",
        "version":     "0.1.0",
        "llm_status":  llm_status.value,   # "ready" | "not_configured" | "error"
        "llm_message": llm_message,
    }
