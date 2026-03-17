"""
core/llm/providers/openai_provider.py
──────────────────────────────────────────────────────────────────────────────
Adapter for OpenAI and Azure OpenAI APIs.

Requires: pip install openai
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import Any, Dict

log = logging.getLogger("bakup.llm.openai")


def _get_params(cfg) -> dict:
    """Build generation parameters from config with env-var overrides."""
    return {
        "max_tokens":  int(os.environ.get("BAKUP_LLM_MAX_TOKENS",    str(cfg.num_predict))),
        "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", str(cfg.temperature))),
    }


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def call(cfg, user_message: str, system_prompt: str) -> str:
    """Send a chat completion request to OpenAI."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    params = _get_params(cfg)
    log.debug("OpenAI call: model=%s  max_tokens=%d  temperature=%.2f",
              cfg.model, params["max_tokens"], params["temperature"])

    client = OpenAI(api_key=cfg.api_key)
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        **params,
    )
    return completion.choices[0].message.content or ""


def call_with_tools(cfg, messages: list, tools: list) -> dict:
    """OpenAI chat with tool definitions."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    params = _get_params(cfg)
    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        **params,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    client = OpenAI(api_key=cfg.api_key)
    completion = client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message

    result: dict = {"content": msg.content or "", "tool_calls": []}
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = _json.loads(tc.function.arguments)
            except (ValueError, TypeError):
                args = {}
            result["tool_calls"].append({
                "id":        tc.id,
                "name":      tc.function.name,
                "arguments": args,
            })
    return result


def ping(cfg) -> None:
    """Validate OpenAI API key with a cheap models.list() call."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=cfg.api_key)
    client.models.list()


# ── Azure OpenAI ───────────────────────────────────────────────────────────────

def call_azure(cfg, user_message: str, system_prompt: str) -> str:
    """Send a chat completion request to Azure OpenAI."""
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    if not cfg.azure_endpoint:
        raise ValueError("Azure endpoint URL is required for azure_openai provider.")

    params = _get_params(cfg)
    log.debug("Azure call: model=%s  endpoint=%s  max_tokens=%d",
              cfg.model, cfg.azure_endpoint, params["max_tokens"])

    client = AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        api_version=cfg.azure_api_version or "2024-02-01",
    )
    completion = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        **params,
    )
    return completion.choices[0].message.content or ""


def call_azure_with_tools(cfg, messages: list, tools: list) -> dict:
    """Azure OpenAI chat with tool definitions."""
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    if not cfg.azure_endpoint:
        raise ValueError("Azure endpoint URL is required.")

    params = _get_params(cfg)
    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        **params,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    client = AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        api_version=cfg.azure_api_version or "2024-02-01",
    )
    completion = client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message

    result: dict = {"content": msg.content or "", "tool_calls": []}
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = _json.loads(tc.function.arguments)
            except (ValueError, TypeError):
                args = {}
            result["tool_calls"].append({
                "id":        tc.id,
                "name":      tc.function.name,
                "arguments": args,
            })
    return result


def ping_azure(cfg) -> None:
    """Validate Azure OpenAI credentials."""
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    if not cfg.azure_endpoint:
        raise ValueError("Azure endpoint URL is required.")
    client = AzureOpenAI(
        api_key=cfg.api_key,
        azure_endpoint=cfg.azure_endpoint,
        api_version=cfg.azure_api_version or "2024-02-01",
    )
    client.models.list()
