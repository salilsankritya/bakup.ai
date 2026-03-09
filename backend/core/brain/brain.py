"""
core/brain/brain.py
─────────────────────────────────────────────────────────────────────────────
Brain Controller — LLM-orchestrated project intelligence.

Architecture:
    ┌───────────────┐
    │  User Query   │
    └───────┬───────┘
            │
    ┌───────▼───────────┐     LLM configured?
    │  Brain Controller │──── No ──► Fallback pipeline (rag.py)
    └───────┬───────────┘
            │ Yes
    ┌───────▼───────────┐
    │  LLM (any provider)│
    │  + tool schemas    │
    └───────┬───────────┘
            │ tool_calls[]
    ┌───────▼───────────┐
    │  Execute tools    │◄── up to max_tool_calls iterations
    │  Feed results     │
    │  back to LLM      │
    └───────┬───────────┘
            │ final answer
    ┌───────▼───────────┐
    │  BrainResponse    │
    └───────────────────┘

The brain is the central reasoning engine.  When an LLM is configured it:
  1. Receives the user query + session context + tool schemas.
  2. Decides which tools to call (search_logs, search_code, etc.).
  3. Receives tool results and may call more tools.
  4. Produces a final reasoned answer grounded in tool evidence.

When NO LLM is configured the brain delegates to the existing deterministic
pipeline (planner → agent → extractive/LLM answer) in rag.py.

Safety:
  - max_tool_calls limits total tool invocations per query (default 5).
  - Tools are read-only — they never mutate project data.
  - The LLM API key is never stored on the brain; it is read from
    config_store at call time via LLMService.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.brain.tools import (
    TOOLS,
    TOOL_MAP,
    ToolResult,
    execute_tool,
    get_tool_schemas_openai,
    get_tool_schemas_anthropic,
)

logger = logging.getLogger("bakup.brain")


# ── Configuration ──────────────────────────────────────────────────────────────

MAX_TOOL_CALLS = int(os.environ.get("BAKUP_MAX_TOOL_CALLS", "5"))


# ── Response model ─────────────────────────────────────────────────────────────

@dataclass
class BrainResponse:
    """
    The output of the brain controller.

    Carries the final answer, debug trace, and metadata about how the
    answer was produced (which tools were called, LLM provider, etc.).
    """
    answer: str
    mode: str                            # "brain" | "fallback" | "greeting" | etc.
    confidence: float = 0.0
    no_data: bool = False
    provider: str = "none"
    model: str = "none"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: float = 0.0
    error: Optional[str] = None


# ── Brain controller ──────────────────────────────────────────────────────────

def process_query(
    question: str,
    namespace: str,
    top_k: int = 8,
    debug: bool = False,
    *,
    session_context: str = "",
    pre_classified: Optional[str] = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> BrainResponse:
    """
    Central entry point for answering a user question.

    When an LLM is configured and supports function-calling:
      → LLM-orchestrated tool loop (brain mode).

    When no LLM is configured (or the provider doesn't support tools):
      → Deterministic pipeline via rag.answer_question() (fallback mode).

    Args:
        question:        The user's question.
        namespace:       Project namespace from indexing.
        top_k:           Number of retrieval candidates per tool call.
        debug:           Attach reasoning trace to the response.
        session_context: Formatted prior conversation context.
        pre_classified:  Pre-computed query category (skips classifier).
        max_tool_calls:  Maximum tool invocations per query.

    Returns:
        BrainResponse with the final answer, tool call trace, and metadata.
    """
    t0 = time.perf_counter()
    trace: List[Dict[str, Any]] = []

    def _trace(step: str, message: str, **data):
        entry = {
            "step": step,
            "message": message,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if data:
            entry["data"] = data
        trace.append(entry)
        print(f"  [bakup:brain] {step}: {message}")

    # ── Check if LLM is configured and supports tool calling ─────────────────
    from core.llm.config_store import load_config
    cfg = load_config()

    if not cfg.configured:
        _trace("fallback", "No LLM configured — using deterministic pipeline")
        return _fallback_pipeline(
            question, namespace, top_k, debug,
            pre_classified=pre_classified,
            trace=trace, t0=t0,
        )

    # Check if this provider supports tool/function calling
    if not _provider_supports_tools(cfg.provider):
        _trace("fallback", f"Provider {cfg.provider} does not support tool calling — using deterministic pipeline")
        return _fallback_pipeline(
            question, namespace, top_k, debug,
            pre_classified=pre_classified,
            trace=trace, t0=t0,
        )

    # ── Handle non-project queries without tools ─────────────────────────────
    if pre_classified in ("greeting", "off_topic", "conversational"):
        _trace("shortcircuit", f"Non-project query ({pre_classified}) — no tools needed")
        return _handle_non_project(question, pre_classified, trace, t0)

    # ── LLM-orchestrated tool loop ──────────────────────────────────────────
    _trace("brain_start", f"Starting LLM-orchestrated reasoning (provider={cfg.provider}, model={cfg.model})")

    try:
        return _run_tool_loop(
            question=question,
            namespace=namespace,
            cfg=cfg,
            session_context=session_context,
            max_tool_calls=max_tool_calls,
            top_k=top_k,
            debug=debug,
            trace=trace,
            t0=t0,
        )
    except Exception as exc:
        _trace("brain_error", f"Brain error: {exc} — falling back to deterministic pipeline")
        logger.warning("Brain tool loop failed: %s — falling back", exc)
        return _fallback_pipeline(
            question, namespace, top_k, debug,
            pre_classified=pre_classified,
            trace=trace, t0=t0,
        )


# ── Tool loop ─────────────────────────────────────────────────────────────────

def _run_tool_loop(
    question: str,
    namespace: str,
    cfg,
    session_context: str,
    max_tool_calls: int,
    top_k: int,
    debug: bool,
    trace: list,
    t0: float,
) -> BrainResponse:
    """
    Run the LLM tool-call loop.

    1. Build the initial message with system prompt + user question.
    2. Send to LLM with tool schemas.
    3. If LLM returns tool_calls → execute them, feed results back.
    4. Repeat until LLM produces a final text answer or budget exhausted.
    """
    from core.llm.llm_service import get_llm_service
    from core.brain.prompt_templates import build_brain_system_prompt

    svc = get_llm_service()
    system_prompt = build_brain_system_prompt(session_context)

    # Build messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_calls_made: List[Dict[str, Any]] = []
    calls_remaining = max_tool_calls

    def _trace_step(step: str, message: str, **data):
        entry = {
            "step": step,
            "message": message,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if data:
            entry["data"] = data
        trace.append(entry)
        print(f"  [bakup:brain] {step}: {message}")

    for iteration in range(max_tool_calls + 1):
        _trace_step("llm_call", f"Iteration {iteration + 1} — calling LLM with {len(messages)} messages")

        # Call LLM with tools
        response = svc.call_with_tools(
            cfg=cfg,
            messages=messages,
            tools=_get_tool_schemas_for_provider(cfg.provider),
        )

        # Check if the LLM wants to call tools
        if response.get("tool_calls") and calls_remaining > 0:
            tool_calls = response["tool_calls"]
            _trace_step("tool_decision", f"LLM requested {len(tool_calls)} tool call(s) ({calls_remaining} remaining)")

            # Add the assistant message (with tool_calls) to conversation
            messages.append(_build_assistant_message(response, cfg.provider))

            for tc in tool_calls:
                if calls_remaining <= 0:
                    _trace_step("budget_exceeded", "Tool call budget exhausted")
                    break

                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_id = tc.get("id", "")

                # Inject namespace if not provided
                if "namespace" in {p.name for p in TOOL_MAP.get(tool_name, type("", (), {"params": []})()).params if hasattr(TOOL_MAP.get(tool_name), "params")} if tool_name in TOOL_MAP else False:
                    pass
                if tool_name in TOOL_MAP:
                    # Auto-inject namespace for tools that need it
                    tool_params = {p.name for p in TOOL_MAP[tool_name].params}
                    if "namespace" in tool_params and "namespace" not in tool_args:
                        tool_args["namespace"] = namespace
                    # Auto-inject top_k default
                    if "top_k" in tool_params and "top_k" not in tool_args:
                        tool_args["top_k"] = top_k

                _trace_step("tool_exec", f"Executing {tool_name}({json.dumps(tool_args, default=str)[:200]})")

                result = execute_tool(tool_name, tool_args)
                calls_remaining -= 1

                tool_calls_made.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result_summary": _summarise_result(result.result),
                    "ms": result.ms,
                    "error": result.error,
                })

                if result.error:
                    _trace_step("tool_error", f"{tool_name} failed: {result.error}")
                else:
                    _trace_step("tool_done", f"{tool_name} completed ({result.ms:.0f}ms)")

                # Add tool result to messages
                messages.append(_build_tool_result_message(
                    tool_id, tool_name, result, cfg.provider,
                ))

        else:
            # LLM produced a final text answer (no more tool calls)
            answer = response.get("content", "").strip()
            if not answer:
                answer = "I was unable to produce an answer from the available tools."

            _trace_step("answer", f"Final answer received ({len(answer)} chars)")

            # Context size reporting for observability
            total_context_chars = sum(len(json.dumps(m.get("content", ""), default=str)) for m in messages)
            _trace_step("context_size", f"Total LLM context: {total_context_chars} chars across {len(messages)} messages")

            total_ms = round((time.perf_counter() - t0) * 1000, 1)

            return BrainResponse(
                answer=answer,
                mode="brain",
                confidence=_estimate_confidence(tool_calls_made),
                provider=cfg.provider,
                model=cfg.model,
                tool_calls=tool_calls_made,
                reasoning_trace=trace if debug else [],
                total_ms=total_ms,
            )

    # Budget fully exhausted — ask LLM for final answer without tools
    _trace_step("budget_final", "Tool budget exhausted — requesting final answer")
    messages.append({
        "role": "user",
        "content": "You have used all available tool calls. Please provide your final answer based on the evidence collected so far.",
    })
    response = svc.call_with_tools(cfg=cfg, messages=messages, tools=[])
    answer = response.get("content", "").strip() or "Analysis incomplete — tool budget exhausted."

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BrainResponse(
        answer=answer,
        mode="brain",
        confidence=_estimate_confidence(tool_calls_made),
        provider=cfg.provider,
        model=cfg.model,
        tool_calls=tool_calls_made,
        reasoning_trace=trace if debug else [],
        total_ms=total_ms,
    )


# ── Provider helpers ──────────────────────────────────────────────────────────

_TOOL_PROVIDERS = {"openai", "azure_openai", "anthropic", "ollama"}


def _provider_supports_tools(provider: str) -> bool:
    """Check if a provider supports function/tool calling."""
    return provider in _TOOL_PROVIDERS


def _get_tool_schemas_for_provider(provider: str) -> list:
    """Return tool schemas in the format expected by the provider."""
    if provider == "anthropic":
        return get_tool_schemas_anthropic()
    # OpenAI, Azure OpenAI, and Ollama all use OpenAI format
    return get_tool_schemas_openai()


def _build_assistant_message(response: dict, provider: str) -> dict:
    """Build the assistant message entry for the conversation history."""
    msg: dict = {"role": "assistant"}

    if provider == "anthropic":
        # Anthropic format: content is a list of blocks
        content_blocks = []
        if response.get("content"):
            content_blocks.append({"type": "text", "text": response["content"]})
        for tc in response.get("tool_calls", []):
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc.get("name", ""),
                "input": tc.get("arguments", {}),
            })
        msg["content"] = content_blocks
    else:
        # OpenAI format
        msg["content"] = response.get("content", "")
        if response.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for tc in response["tool_calls"]
            ]

    return msg


def _build_tool_result_message(
    tool_id: str,
    tool_name: str,
    result: ToolResult,
    provider: str,
) -> dict:
    """Build the tool result message for feeding back into the conversation."""
    # Truncate result to avoid context overflow
    result_json = json.dumps(result.result, default=str)
    if len(result_json) > 4000:
        result_json = result_json[:4000] + '..."}'

    if result.error:
        content = json.dumps({"error": result.error})
    else:
        content = result_json

    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            ],
        }
    else:
        # OpenAI format
        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": content,
        }


# ── Fallback pipeline ─────────────────────────────────────────────────────────

def _fallback_pipeline(
    question: str,
    namespace: str,
    top_k: int,
    debug: bool,
    *,
    pre_classified: Optional[str] = None,
    trace: list,
    t0: float,
) -> BrainResponse:
    """
    Delegate to the existing deterministic pipeline in rag.py.

    This is the fallback when no LLM is configured or when the LLM
    provider does not support tool calling.
    """
    from core.retrieval.rag import answer_question

    rag_result = answer_question(
        question=question,
        namespace=namespace,
        top_k=top_k,
        debug=debug,
        pre_classified=pre_classified,
    )

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BrainResponse(
        answer=rag_result.answer,
        mode=f"fallback:{rag_result.mode}",
        confidence=rag_result.confidence,
        no_data=rag_result.no_data,
        sources=[
            {
                "file": s.file,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "excerpt": s.excerpt,
                "confidence": s.confidence,
                "confidence_label": s.confidence_label,
                "source_type": s.source_type,
            }
            for s in rag_result.sources
        ],
        reasoning_trace=trace + (rag_result.debug_trace or []),
        total_ms=total_ms,
    )


def _handle_non_project(
    question: str,
    category: str,
    trace: list,
    t0: float,
) -> BrainResponse:
    """Handle greeting / off_topic / conversational without tools."""
    from core.retrieval.rag import answer_question

    result = answer_question(
        question=question,
        namespace="_",
        top_k=1,
        debug=False,
        pre_classified=category,
    )

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BrainResponse(
        answer=result.answer,
        mode=category,
        confidence=result.confidence,
        no_data=result.no_data,
        total_ms=total_ms,
        reasoning_trace=trace,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _estimate_confidence(tool_calls: list) -> float:
    """Estimate answer confidence from tool call results."""
    if not tool_calls:
        return 0.5

    successful = sum(1 for tc in tool_calls if not tc.get("error"))
    total = len(tool_calls)

    if total == 0:
        return 0.5

    # Base confidence from tool success rate
    base = successful / total

    # Boost for having multiple evidence sources
    if successful >= 3:
        base = min(1.0, base + 0.1)

    return round(base, 2)


def _summarise_result(result: dict) -> str:
    """Create a short summary of a tool result for the trace."""
    if not result:
        return "empty"

    parts = []
    for key in ["logs_found", "code_found", "chunks_found", "available",
                 "answered", "clusters", "total_errors", "has_cross_analysis"]:
        if key in result:
            parts.append(f"{key}={result[key]}")

    return ", ".join(parts[:5]) if parts else f"{len(result)} keys"


# ── Debug / introspection ────────────────────────────────────────────────────

# In-memory store for the most recent brain invocation (per namespace).
# Bounded to prevent unbounded memory growth.
_MAX_DEBUG_CACHE = 50
_last_brain_result: Dict[str, BrainResponse] = {}


def store_debug_result(namespace: str, result: BrainResponse) -> None:
    """Store the most recent brain result for debug inspection (LRU-evicted)."""
    # Evict oldest entries if cache is full
    while len(_last_brain_result) >= _MAX_DEBUG_CACHE:
        oldest = next(iter(_last_brain_result))
        del _last_brain_result[oldest]
    _last_brain_result[namespace] = result


def get_debug_result(namespace: str) -> Optional[BrainResponse]:
    """Retrieve the most recent brain result for a namespace."""
    return _last_brain_result.get(namespace)
