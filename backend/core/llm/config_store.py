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
    own API endpoint (OpenAI, Azure OpenAI, or a local Ollama instance).
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
from typing import Optional


# ── Supported providers ────────────────────────────────────────────────────────

PROVIDERS = ("openai", "anthropic", "azure_openai", "ollama")

DEFAULT_MODELS = {
    "openai":       "gpt-4o-mini",
    "anthropic":    "claude-sonnet-4-20250514",
    "azure_openai": "gpt-4o-mini",
    "ollama":       "llama3",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    Mutable configuration for the active LLM provider.

    Fields:
        provider         One of "openai" | "azure_openai" | "ollama"
        model            Model name / deployment name
        api_key          OpenAI or Azure key — empty string for Ollama
        azure_endpoint   Required for azure_openai (e.g. https://my.openai.azure.com/)
        azure_api_version  Azure API version string
        ollama_base_url  Base URL for local Ollama server
        configured       False until the user saves a valid config
    """
    provider:          str  = "ollama"
    model:             str  = "llama3"
    api_key:           str  = ""          # Never logged, never returned in public view
    azure_endpoint:    str  = ""
    azure_api_version: str  = "2024-02-01"
    ollama_base_url:   str  = DEFAULT_OLLAMA_URL
    configured:        bool = False


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
            model             = raw.get("model",             "llama3"),
            api_key           = raw.get("api_key",           ""),
            azure_endpoint    = raw.get("azure_endpoint",    ""),
            azure_api_version = raw.get("azure_api_version", "2024-02-01"),
            ollama_base_url   = raw.get("ollama_base_url",   DEFAULT_OLLAMA_URL),
            configured        = raw.get("configured",        False),
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
