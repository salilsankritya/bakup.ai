"""
api/routes/llm_config.py
─────────────────────────────────────────────────────────────────────────────
REST endpoints for LLM provider configuration and status.

Endpoints:
  GET  /llm/status        — Current LLM health (safe to poll from UI)
  GET  /llm/config        — Current config with API key masked
  POST /llm/config        — Save a new config and test connectivity
  POST /llm/test          — Test connectivity without saving
  DELETE /llm/config      — Reset to unconfigured state
  GET  /llm/providers     — List all supported providers and their models
  GET  /llm/ollama-models — Auto-detect locally installed Ollama models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.llm.config_store import (
    LLMConfig,
    PROVIDERS,
    DEFAULT_MODELS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_LLM_PARAMS,
    PROVIDER_MODELS,
    PROVIDER_INFO,
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
    api_key:           str   = ""
    azure_endpoint:    str   = ""
    azure_api_version: str   = "2024-02-01"
    ollama_base_url:   str   = DEFAULT_OLLAMA_URL
    # Generation parameters (optional — defaults from config_store)
    temperature:       Optional[float] = None
    num_predict:       Optional[int]   = None
    num_ctx:           Optional[int]   = None
    timeout:           Optional[int]   = None

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
    temperature:       float
    num_predict:       int
    num_ctx:           int
    timeout:           int


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
        temperature       = body.temperature  if body.temperature  is not None else DEFAULT_LLM_PARAMS["temperature"],
        num_predict       = body.num_predict   if body.num_predict   is not None else DEFAULT_LLM_PARAMS["num_predict"],
        num_ctx           = body.num_ctx        if body.num_ctx        is not None else DEFAULT_LLM_PARAMS["num_ctx"],
        timeout           = body.timeout        if body.timeout        is not None else DEFAULT_LLM_PARAMS["timeout"],
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
        temperature       = body.temperature  if body.temperature  is not None else DEFAULT_LLM_PARAMS["temperature"],
        num_predict       = body.num_predict   if body.num_predict   is not None else DEFAULT_LLM_PARAMS["num_predict"],
        num_ctx           = body.num_ctx        if body.num_ctx        is not None else DEFAULT_LLM_PARAMS["num_ctx"],
        timeout           = body.timeout        if body.timeout        is not None else DEFAULT_LLM_PARAMS["timeout"],
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


@router.get("/providers")
async def list_providers() -> dict:
    """
    Return all supported providers with their display labels,
    available model choices, and default model.

    The UI uses this to populate the provider & model dropdowns.
    """
    return {
        "providers": [
            {
                "id":           pid,
                "label":        PROVIDER_INFO[pid]["label"],
                "needs_key":    PROVIDER_INFO[pid]["needs_key"] == "true",
                "default_model": DEFAULT_MODELS.get(pid, ""),
                "models":       PROVIDER_MODELS.get(pid, []),
            }
            for pid in PROVIDERS
        ],
        "defaults": DEFAULT_MODELS,
        "default_params": DEFAULT_LLM_PARAMS,
    }


@router.get("/ollama-models")
async def list_ollama_models() -> dict:
    """
    Auto-detect locally installed Ollama models by querying the
    Ollama API at the configured (or default) base URL.

    Returns an empty list if Ollama is not running.
    """
    from core.llm.providers.ollama_provider import list_models

    cfg = load_config()
    models = list_models(cfg)
    return {
        "models": models,
        "ollama_running": len(models) > 0,
    }
