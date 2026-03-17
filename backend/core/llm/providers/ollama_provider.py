"""
core/llm/providers/ollama_provider.py
──────────────────────────────────────────────────────────────────────────────
Adapter for local Ollama server.

No extra dependencies — uses urllib only.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from core.llm.config_store import DEFAULT_OLLAMA_URL

log = logging.getLogger("bakup.llm.ollama")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base(cfg) -> str:
    return (getattr(cfg, "ollama_base_url", None) or DEFAULT_OLLAMA_URL).rstrip("/")


def _get_options(cfg) -> dict:
    """Build Ollama options dict from config, with env-var overrides."""
    return {
        "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", str(cfg.temperature))),
        "num_predict": int(os.environ.get("BAKUP_LLM_MAX_TOKENS",    str(cfg.num_predict))),
        "num_ctx":     int(os.environ.get("BAKUP_OLLAMA_CTX",         str(cfg.num_ctx))),
    }


def _get_timeout(cfg) -> int:
    return int(os.environ.get("BAKUP_OLLAMA_TIMEOUT", str(cfg.timeout)))


# ── Core functions ─────────────────────────────────────────────────────────────

def call(cfg, user_message: str, system_prompt: str) -> str:
    """Send a chat completion request to Ollama and return the text."""
    base = _base(cfg)
    url  = f"{base}/api/chat"

    payload = {
        "model":  cfg.model,
        "stream": False,
        "options": _get_options(cfg),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }
    timeout = _get_timeout(cfg)
    log.debug("POST %s  model=%s  timeout=%ds  options=%s", url, cfg.model, timeout, payload["options"])

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    return data.get("message", {}).get("content", "")


def call_with_tools(cfg, messages: list, tools: list) -> dict:
    """Send a chat request with tool definitions to Ollama."""
    base = _base(cfg)
    url  = f"{base}/api/chat"

    payload: dict = {
        "model":   cfg.model,
        "stream":  False,
        "options": _get_options(cfg),
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    timeout = _get_timeout(cfg)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    msg = data.get("message", {})
    content = msg.get("content", "")
    tool_calls = []

    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        tool_calls.append({
            "id":        tc.get("id", f"call_{len(tool_calls)}"),
            "name":      fn.get("name", ""),
            "arguments": fn.get("arguments", {}),
        })

    return {"content": content, "tool_calls": tool_calls}


def ping(cfg) -> None:
    """Verify Ollama is reachable (uses /api/tags endpoint)."""
    base = _base(cfg)
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {base}. "
            f"Is Ollama running? (ollama serve)  Detail: {exc}"
        )


def list_models(cfg=None) -> List[Dict[str, Any]]:
    """
    Query Ollama /api/tags to discover locally installed models.

    Returns a list of dicts with keys: name, size, modified_at, details.
    Returns an empty list on any failure (Ollama not running, etc.).
    """
    base_url = DEFAULT_OLLAMA_URL
    if cfg and getattr(cfg, "ollama_base_url", None):
        base_url = cfg.ollama_base_url.rstrip("/")
    else:
        base_url = os.environ.get("BAKUP_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        log.debug("Could not list Ollama models: %s", exc)
        return []

    models = []
    for m in data.get("models", []):
        models.append({
            "name":        m.get("name", ""),
            "size":        m.get("size", 0),
            "modified_at": m.get("modified_at", ""),
            "details":     m.get("details", {}),
        })

    log.debug("Discovered %d Ollama models", len(models))
    return models
