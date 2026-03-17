"""
core/llm/config_store.py
─────────────────────────────────────────────────────────────────────────────
Secure local storage for LLM provider configuration.

Security model:
  - Configuration is written to a single JSON file on the local machine.
  - The API key is stored in this file at rest (no additional encryption by
    default — the file is only readable by the local user via OS permissions).
  - The API key is NEVER written to any log file.
  - The API key is NEVER transmitted anywhere except the chosen provider's
    own API endpoint (OpenAI, Azure OpenAI, Anthropic, or a local Ollama
    instance).
  - get_config_public() returns a sanitised view — key is masked — safe for
    API responses and logging.

Storage location: <BAKUP_MODEL_CACHE_DIR>/bakup_llm_config.json
  (Defaults to  model-weights/bakup_llm_config.json)
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Supported providers & models ───────────────────────────────────────────────

PROVIDERS = ("openai", "anthropic", "azure_openai", "ollama")

DEFAULT_MODELS: Dict[str, str] = {
    "openai":       "gpt-4o",
    "anthropic":    "claude-sonnet-4-20250514",
    "azure_openai": "gpt-4o",
    "ollama":       "gemma3:4b",
}

# Model choices shown in the UI per provider.
PROVIDER_MODELS: Dict[str, List[Dict[str, str]]] = {
    "openai": [
        {"label": "GPT-4o",         "value": "gpt-4o"},
        {"label": "GPT-4.1",        "value": "gpt-4.1"},
        {"label": "GPT-4.1 Mini",   "value": "gpt-4.1-mini"},
        {"label": "GPT-4o Mini",    "value": "gpt-4o-mini"},
        {"label": "GPT-4.1 Nano",   "value": "gpt-4.1-nano"},
    ],
    "anthropic": [
        {"label": "Claude Sonnet 4",  "value": "claude-sonnet-4-20250514"},
        {"label": "Claude Opus 4",    "value": "claude-opus-4-20250514"},
        {"label": "Claude Haiku 3.5", "value": "claude-3-5-haiku-20241022"},
    ],
    "azure_openai": [
        {"label": "GPT-4o",       "value": "gpt-4o"},
        {"label": "GPT-4.1",      "value": "gpt-4.1"},
        {"label": "GPT-4.1 Mini", "value": "gpt-4.1-mini"},
    ],
    "ollama": [
        {"label": "Gemma 3 4B",   "value": "gemma3:4b"},
        {"label": "LLaMA 3",      "value": "llama3"},
        {"label": "Mistral",      "value": "mistral"},
        {"label": "CodeLlama",    "value": "codellama"},
    ],
}

# Provider display metadata for the UI.
PROVIDER_INFO: Dict[str, Dict[str, str]] = {
    "openai":       {"label": "OpenAI",                "needs_key": "true"},
    "anthropic":    {"label": "Anthropic (Claude)",     "needs_key": "true"},
    "azure_openai": {"label": "Azure OpenAI",           "needs_key": "true"},
    "ollama":       {"label": "Ollama (local, no key)",  "needs_key": "false"},
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# ── Default LLM generation parameters ─────────────────────────────────────────

DEFAULT_LLM_PARAMS: Dict[str, Any] = {
    "temperature":  0.1,
    "num_predict":  1024,   # max tokens to generate
    "num_ctx":      8192,   # context window (Ollama only)
    "timeout":      300,    # request timeout in seconds
}


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    Mutable configuration for the active LLM provider.

    Fields:
        provider         One of "openai" | "anthropic" | "azure_openai" | "ollama"
        model            Model name / deployment name
        api_key          OpenAI, Azure, or Anthropic key — empty for Ollama
        azure_endpoint   Required for azure_openai (e.g. https://my.openai.azure.com/)
        azure_api_version  Azure API version string
        ollama_base_url  Base URL for local Ollama server
        configured       False until the user saves a valid config
        temperature      Generation temperature (0.0–2.0)
        num_predict      Max tokens to generate per request
        num_ctx          Context window size (Ollama only)
        timeout          Request timeout in seconds
    """
    provider:          str   = "ollama"
    model:             str   = "gemma3:4b"
    api_key:           str   = ""          # Never logged, never returned in public view
    azure_endpoint:    str   = ""
    azure_api_version: str   = "2024-02-01"
    ollama_base_url:   str   = DEFAULT_OLLAMA_URL
    configured:        bool  = False
    # Generation parameters (persisted so user can tune from UI)
    temperature:       float = DEFAULT_LLM_PARAMS["temperature"]
    num_predict:       int   = DEFAULT_LLM_PARAMS["num_predict"]
    num_ctx:           int   = DEFAULT_LLM_PARAMS["num_ctx"]
    timeout:           int   = DEFAULT_LLM_PARAMS["timeout"]


# ── Singleton in-memory cache ──────────────────────────────────────────────────

_cached_config: Optional[LLMConfig] = None


def _config_path() -> Path:
    cache_dir = Path(os.environ.get("BAKUP_MODEL_CACHE_DIR", "model-weights"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "bakup_llm_config.json"


def load_config() -> LLMConfig:
    """Load config from disk. Returns default (unconfigured) if file does not exist."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    path = _config_path()
    if not path.exists():
        _cached_config = LLMConfig()
        return _cached_config

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _cached_config = LLMConfig(
            provider          = raw.get("provider",          "ollama"),
            model             = raw.get("model",             "gemma3:4b"),
            api_key           = raw.get("api_key",           ""),
            azure_endpoint    = raw.get("azure_endpoint",    ""),
            azure_api_version = raw.get("azure_api_version", "2024-02-01"),
            ollama_base_url   = raw.get("ollama_base_url",   DEFAULT_OLLAMA_URL),
            configured        = raw.get("configured",        False),
            temperature       = float(raw.get("temperature", DEFAULT_LLM_PARAMS["temperature"])),
            num_predict       = int(raw.get("num_predict",   DEFAULT_LLM_PARAMS["num_predict"])),
            num_ctx           = int(raw.get("num_ctx",       DEFAULT_LLM_PARAMS["num_ctx"])),
            timeout           = int(raw.get("timeout",       DEFAULT_LLM_PARAMS["timeout"])),
        )
    except Exception as exc:
        # Corrupt config — start fresh, but don't crash.
        print(f"bakup: could not read LLM config ({exc}) — using defaults.")
        _cached_config = LLMConfig()

    return _cached_config


def save_config(cfg: LLMConfig) -> None:
    """
    Persist config to disk.

    The file is created with owner-read-only permissions (0600) on POSIX.
    On Windows the directory ACL provides equivalent protection.

    IMPORTANT: api_key is written to disk. The file must stay on the user's
    local machine and must not be committed to version control.
    """
    global _cached_config
    path = _config_path()

    payload = asdict(cfg)
    payload["configured"] = True

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Restrict file permissions to owner-only on POSIX systems.
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # Windows — skip silently

    cfg.configured = True
    _cached_config = cfg


def get_config_public() -> dict:
    """
    Return a sanitised dict safe to include in API responses and logs.
    The API key is masked — only the last 4 characters are shown.
    """
    cfg = load_config()
    masked_key = _mask_key(cfg.api_key)
    return {
        "provider":          cfg.provider,
        "model":             cfg.model,
        "api_key_set":       bool(cfg.api_key),
        "api_key_preview":   masked_key,
        "azure_endpoint":    cfg.azure_endpoint,
        "azure_api_version": cfg.azure_api_version,
        "ollama_base_url":   cfg.ollama_base_url,
        "configured":        cfg.configured,
        "temperature":       cfg.temperature,
        "num_predict":       cfg.num_predict,
        "num_ctx":           cfg.num_ctx,
        "timeout":           cfg.timeout,
    }


def invalidate_cache() -> None:
    """Force re-read from disk on next access (call after save)."""
    global _cached_config
    _cached_config = None


# ── Internal helpers ───────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 4:
        return "****"
    return "****" + key[-4:]
