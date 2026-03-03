"""
api/routes/llm_config.py
─────────────────────────────────────────────────────────────────────────────
REST endpoints for LLM provider configuration and status.

Endpoints:
  GET  /llm/status   — Current LLM health (safe to poll from UI)
  GET  /llm/config   — Current config with API key masked
  POST /llm/config   — Save a new config and test connectivity
  POST /llm/test     — Test connectivity without saving
  DELETE /llm/config — Reset to unconfigured state
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.llm.config_store import (
    LLMConfig,
    PROVIDERS,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    get_config_public,
    invalidate_cache,
    load_config,
    save_config,
)
from core.llm.llm_service import LLMStatus, get_llm_service

router = APIRouter(prefix="/llm", tags=["llm"])


# ── Request / response models ──────────────────────────────────────────────────

class LLMConfigRequest(BaseModel):
    provider:          str
    model:             str
    api_key:           str  = ""
    azure_endpoint:    str  = ""
    azure_api_version: str  = "2024-02-01"
    ollama_base_url:   str  = DEFAULT_OLLAMA_URL

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in PROVIDERS:
            raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must not be empty")
        return v


class LLMStatusResponse(BaseModel):
    status:   str       # "ready" | "not_configured" | "error"
    provider: str
    model:    str
    message:  str


class LLMConfigResponse(BaseModel):
    provider:          str
    model:             str
    api_key_set:       bool
    api_key_preview:   str
    azure_endpoint:    str
    azure_api_version: str
    ollama_base_url:   str
    configured:        bool
    default_models:    dict


class SaveConfigResponse(BaseModel):
    status:   str
    message:  str
    provider: str
    model:    str


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=LLMStatusResponse)
async def llm_status() -> LLMStatusResponse:
    """
    Poll this to update the UI health indicator.
    Safe to call frequently — performs a connectivity check only when configured.
    """
    svc = get_llm_service()
    status, message = svc.health_check()
    cfg = load_config()
    return LLMStatusResponse(
        status   = status.value,
        provider = cfg.provider,
        model    = cfg.model,
        message  = message,
    )


@router.get("/config", response_model=LLMConfigResponse)
async def get_config() -> LLMConfigResponse:
    """Return current config. API key is always masked."""
    public = get_config_public()
    return LLMConfigResponse(**public, default_models=DEFAULT_MODELS)


@router.post("/config", response_model=SaveConfigResponse)
async def save_llm_config(body: LLMConfigRequest) -> SaveConfigResponse:
    """
    Save LLM config and immediately test connectivity.

    The API key (if provided) is stored locally on this machine only.
    It is never logged and never transmitted except to the selected provider.
    """
    cfg = LLMConfig(
        provider          = body.provider,
        model             = body.model,
        api_key           = body.api_key,
        azure_endpoint    = body.azure_endpoint,
        azure_api_version = body.azure_api_version,
        ollama_base_url   = body.ollama_base_url,
        configured        = False,   # will be set True inside save_config()
    )

    # Validate provider-specific requirements before saving.
    if body.provider in ("openai", "azure_openai") and not body.api_key:
        raise HTTPException(
            status_code=422,
            detail=f"An API key is required for the '{body.provider}' provider.",
        )
    if body.provider == "azure_openai" and not body.azure_endpoint:
        raise HTTPException(
            status_code=422,
            detail="An Azure endpoint URL is required for the 'azure_openai' provider.",
        )

    # Test connectivity before persisting.
    svc = get_llm_service()
    try:
        svc._ping_provider(cfg)   # raises on unreachable / bad key
    except Exception as exc:
        from core.llm.llm_service import _redact_key
        safe = _redact_key(str(exc), body.api_key)
        raise HTTPException(status_code=422, detail=f"Connection test failed: {safe}")

    # Persist — this also sets cfg.configured = True.
    save_config(cfg)
    invalidate_cache()

    # Bust the health-check ping cache so the UI immediately shows the new status.
    from core.llm.llm_service import _health_cache
    _health_cache.clear()

    return SaveConfigResponse(
        status   = "saved",
        message  = (
            "Configuration saved. "
            "Your API key is stored locally on this machine only and is never transmitted "
            "anywhere except the selected provider's API endpoint."
        ),
        provider = cfg.provider,
        model    = cfg.model,
    )


@router.post("/test", response_model=LLMStatusResponse)
async def test_connection(body: LLMConfigRequest) -> LLMStatusResponse:
    """
    Test LLM connectivity without saving config.
    Useful for verifying credentials before committing.
    """
    cfg = LLMConfig(
        provider          = body.provider,
        model             = body.model,
        api_key           = body.api_key,
        azure_endpoint    = body.azure_endpoint,
        azure_api_version = body.azure_api_version,
        ollama_base_url   = body.ollama_base_url,
        configured        = True,
    )
    svc = get_llm_service()
    try:
        svc._ping_provider(cfg)
        return LLMStatusResponse(
            status   = LLMStatus.READY.value,
            provider = cfg.provider,
            model    = cfg.model,
            message  = f"Connection successful — {cfg.provider} / {cfg.model}",
        )
    except Exception as exc:
        from core.llm.llm_service import _redact_key
        safe = _redact_key(str(exc), body.api_key)
        return LLMStatusResponse(
            status   = LLMStatus.ERROR.value,
            provider = cfg.provider,
            model    = cfg.model,
            message  = safe,
        )


@router.delete("/config")
async def reset_config() -> dict:
    """Remove LLM config and return to unconfigured state."""
    import os
    from core.llm.config_store import _config_path
    from core.llm.llm_service import _health_cache
    path = _config_path()
    if path.exists():
        path.unlink()
    invalidate_cache()
    _health_cache.clear()
    return {"status": "reset", "message": "LLM configuration cleared."}
