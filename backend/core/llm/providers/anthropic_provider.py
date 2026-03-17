"""
core/llm/providers/anthropic_provider.py
──────────────────────────────────────────────────────────────────────────────
Adapter for Anthropic Claude API.

No extra dependencies — uses urllib only (Messages API).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger("bakup.llm.anthropic")

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def _get_params(cfg) -> dict:
    """Build generation parameters from config with env-var overrides."""
    return {
        "max_tokens":  int(os.environ.get("BAKUP_LLM_MAX_TOKENS",    str(cfg.num_predict))),
        "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", str(cfg.temperature))),
    }


def _headers(api_key: str) -> dict:
    return {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }


# ── Core functions ─────────────────────────────────────────────────────────────

def call(cfg, user_message: str, system_prompt: str) -> str:
    """Send a message to the Anthropic Messages API and return the text."""
    params = _get_params(cfg)
    timeout = int(os.environ.get("BAKUP_OLLAMA_TIMEOUT", str(cfg.timeout)))

    payload = {
        "model":       cfg.model,
        "max_tokens":  params["max_tokens"],
        "temperature": params["temperature"],
        "system":      system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    log.debug("Anthropic call: model=%s  max_tokens=%d  temperature=%.2f",
              cfg.model, params["max_tokens"], params["temperature"])

    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode(),
        headers=_headers(cfg.api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    # Extract text from content blocks
    text_parts = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "".join(text_parts)


def call_with_tools(cfg, messages: list, tools: list) -> dict:
    """Send a chat request with tool definitions to Anthropic."""
    params = _get_params(cfg)
    timeout = int(os.environ.get("BAKUP_OLLAMA_TIMEOUT", str(cfg.timeout)))

    # Separate system message from conversation messages
    system_text = ""
    conv_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"] if isinstance(m["content"], str) else str(m["content"])
        else:
            conv_messages.append(m)

    payload: dict = {
        "model":       cfg.model,
        "max_tokens":  params["max_tokens"],
        "temperature": params["temperature"],
        "messages":    conv_messages,
    }
    if system_text:
        payload["system"] = system_text
    if tools:
        payload["tools"] = tools

    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode(),
        headers=_headers(cfg.api_key),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    # Parse Anthropic response
    content_text = ""
    tool_calls = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            content_text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id":        block.get("id", ""),
                "name":      block.get("name", ""),
                "arguments": block.get("input", {}),
            })

    return {"content": content_text, "tool_calls": tool_calls}


def ping(cfg) -> None:
    """Validate Anthropic API key with a minimal request."""
    payload = {
        "model":      cfg.model,
        "max_tokens": 1,
        "messages":   [{"role": "user", "content": "ping"}],
    }
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode(),
        headers=_headers(cfg.api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass  # 200 = key is valid
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("Invalid Anthropic API key.")
        # Other errors (rate limit, etc.) still mean the key is working
        if e.code >= 500:
            raise
