"""
core/llm/providers
──────────────────────────────────────────────────────────────────────────────
Provider adapter modules for bakup.ai LLM abstraction.

Each adapter exposes three functions:
    call(cfg, user_message, system_prompt) -> str
    call_with_tools(cfg, messages, tools) -> dict
    ping(cfg) -> None   (raises on failure)
"""

from __future__ import annotations

from core.llm.providers.ollama_provider import (
    call as ollama_call,
    call_with_tools as ollama_call_with_tools,
    ping as ollama_ping,
    list_models as ollama_list_models,
)
from core.llm.providers.openai_provider import (
    call as openai_call,
    call_with_tools as openai_call_with_tools,
    ping as openai_ping,
)
from core.llm.providers.anthropic_provider import (
    call as anthropic_call,
    call_with_tools as anthropic_call_with_tools,
    ping as anthropic_ping,
)

__all__ = [
    "ollama_call", "ollama_call_with_tools", "ollama_ping", "ollama_list_models",
    "openai_call", "openai_call_with_tools", "openai_ping",
    "anthropic_call", "anthropic_call_with_tools", "anthropic_ping",
]
