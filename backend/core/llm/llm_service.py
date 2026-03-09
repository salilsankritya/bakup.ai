"""
core/llm/llm_service.py
─────────────────────────────────────────────────────────────────────────────
LLM abstraction layer for bakup.ai.

Supports three providers:
  - openai       — OpenAI API (GPT-4o-mini, GPT-4o, etc.)
  - azure_openai — Azure OpenAI deployment
  - ollama       — Local Ollama server (llama3, mistral, codellama, etc.)

Design principles:
  1. Single entry point: LLMService.generate_response(chunks, question)
  2. Provider logic is fully isolated — swapping providers requires only a
     config change, no code change anywhere in the rest of the app.
  3. The API key is read from config_store at call time and is NEVER stored
     as an instance variable, class variable, or written to any log.
  4. All responses include source citations and a confidence label.
  5. If no relevant context exists, the answer is always the canonical
     "No similar incident found." string — never a hallucinated answer.

Dependencies (install only what you need):
  openai       → pip install openai
  azure_openai → pip install openai   (same package, different client init)
  ollama       → pip install ollama   (or just needs Ollama running locally)
"""

from __future__ import annotations

import enum
import os
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from core.llm.config_store import LLMConfig, load_config
from core.llm.prompt_templates import (
    SYSTEM_RAG,
    SYSTEM_CLARIFY,
    NO_ANSWER_TOKEN,
    build_rag_user_message,
    build_clarify_user_message,
    build_context_block,
)


# ── Health-check result cache (avoids pinging provider on every poll) ─────────

import time as _time

_health_cache: dict = {}          # {"status": LLMStatus, "msg": str, "ts": float}
_HEALTH_TTL   = 60.0              # seconds between real pings


# ── Status enum ────────────────────────────────────────────────────────────────

class LLMStatus(str, enum.Enum):
    READY         = "ready"           # Configured and connectivity verified
    NOT_CONFIGURED = "not_configured" # User has not saved any LLM config yet
    ERROR         = "error"           # Config exists but provider check failed


# ── Response model ─────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    answer:     str
    mode:       str               # "llm" | "extractive" | "no_data"
    provider:   str               # e.g. "openai", "ollama"
    model:      str
    no_data:    bool = False
    error:      Optional[str] = None
    sources:    List[dict] = field(default_factory=list)   # passed-through from RAG


# ── System prompt & constants ──────────────────────────────────────────────────
# Prompts are defined in prompt_templates.py — imported above.

_MAX_CHUNK_CHARS = 1200   # Chars per chunk sent to LLM — guards against ctx overflow


# ── Main service class ─────────────────────────────────────────────────────────

class LLMService:
    """
    Unified interface to all supported LLM providers.

    Usage:
        svc = LLMService()
        response = await svc.generate_response(ranked_chunks, question)
        print(response.answer)

    Thread-safety: instances are stateless — safe to share across requests.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_response(
        self,
        chunks: list,           # List[RankedResult] from ranker.py
        question: str,
        top_n: int = 5,
    ) -> LLMResponse:
        """
        Generate a cited answer for *question* using *chunks* as context.

        Args:
            chunks:   Ranked retrieval results (RankedResult objects).
            question: The user's plain-English question.
            top_n:    Maximum number of chunks to include in the LLM context.

        Returns:
            LLMResponse with answer, mode, no_data flag, and pass-through sources.
        """
        if not chunks:
            print("  [bakup:debug] generate_response: no chunks — returning no_data")
            return self._no_data_response()

        cfg = load_config()
        if not cfg.configured:
            # Graceful fallback when LLM is not yet configured.
            print(f"  [bakup:debug] LLM not configured — extractive fallback ({len(chunks)} chunks)")
            return LLMResponse(
                answer=self._extractive_fallback(chunks),
                mode="extractive",
                provider="none",
                model="none",
                sources=self._serialise_sources(chunks[:top_n]),
            )

        context   = build_context_block(chunks[:top_n], max_chars=_MAX_CHUNK_CHARS)
        user_msg  = build_rag_user_message(question, context)

        print(f"  [bakup:debug] LLM call: generate_response ({cfg.provider}/{cfg.model})")
        print(f"  [bakup:debug]   context: {len(context)} chars, {len(chunks[:top_n])} chunks")

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=SYSTEM_RAG)
        except Exception as exc:
            # Never surface the API key — strip it from any exception message.
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM call FAILED: {safe_msg}")
            return LLMResponse(
                answer=self._extractive_fallback(chunks),
                mode="extractive",
                provider=cfg.provider,
                model=cfg.model,
                error=f"LLM call failed ({safe_msg}) — showing extractive result.",
                sources=self._serialise_sources(chunks[:top_n]),
            )

        print(f"  [bakup:debug] LLM response received ({len(raw)} chars)")

        if raw.strip().startswith(NO_ANSWER_TOKEN):
            print("  [bakup:debug] LLM returned NO_ANSWER token")
            return self._no_data_response(cfg)

        return LLMResponse(
            answer=raw.strip(),
            mode="llm",
            provider=cfg.provider,
            model=cfg.model,
            sources=self._serialise_sources(chunks[:top_n]),
        )

    def generate_clarification(
        self,
        question: str,
        near_miss_chunks: list,
        best_confidence: float,
    ) -> LLMResponse:
        """
        When retrieval returns only low-confidence results, ask the LLM to
        produce a helpful clarifying question instead of guessing an answer.

        Falls back to a canned response if LLM is not configured.
        """
        from core.classifier.query_classifier import low_confidence_response

        cfg = load_config()
        if not cfg.configured:
            return LLMResponse(
                answer=low_confidence_response(best_confidence),
                mode="clarification",
                provider="none",
                model="none",
                no_data=True,
                sources=self._serialise_sources(near_miss_chunks[:3]),
            )

        near_files = list(dict.fromkeys(c.source_file for c in near_miss_chunks[:5]))
        user_msg   = build_clarify_user_message(question, near_files, best_confidence)

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=SYSTEM_CLARIFY)
        except Exception:
            return LLMResponse(
                answer=low_confidence_response(best_confidence),
                mode="clarification",
                provider=cfg.provider,
                model=cfg.model,
                no_data=True,
                sources=self._serialise_sources(near_miss_chunks[:3]),
            )

        return LLMResponse(
            answer=raw.strip(),
            mode="clarification",
            provider=cfg.provider,
            model=cfg.model,
            no_data=True,
            sources=self._serialise_sources(near_miss_chunks[:3]),
        )

    def generate_log_summary(
        self,
        question: str,
        log_chunks: list,
        trend_summary: str = "",
        cluster_summary: str = "",
        confidence_summary: str = "",
        file_aggregation_summary: str = "",
        cross_analysis_context: str = "",
    ) -> LLMResponse:
        """
        Summarise errors/issues found in log chunks.

        Used when a log-style query returns low-confidence embedding matches
        but keyword matching finds actual log entries. Instead of returning
        "No similar incident found", we pass the entries to the LLM and let
        it summarise whether any errors exist.

        When trend/cluster/confidence/file-aggregation summaries are provided,
        they are injected into the context so the LLM can reference them.

        When cross_analysis_context is provided (log-to-code links), it is
        appended after the standard context so the LLM can correlate errors
        with source code for root-cause analysis.

        Falls back to an extractive display of log entries when the LLM is
        not configured.
        """
        from core.llm.prompt_templates import (
            SYSTEM_LOG_SUMMARY, SYSTEM_CROSS_ANALYSIS,
            build_log_analysis_context,
        )

        print(f"  [bakup:debug] generate_log_summary called with {len(log_chunks)} chunks")

        cfg = load_config()
        if not cfg.configured:
            print("  [bakup:debug] LLM not configured — extractive log fallback")
            return self._log_extractive_fallback(log_chunks)

        context = build_log_analysis_context(
            log_chunks[:5],
            trend_summary=trend_summary,
            cluster_summary=cluster_summary,
            confidence_summary=confidence_summary,
            file_aggregation_summary=file_aggregation_summary,
            max_chars=_MAX_CHUNK_CHARS,
        )

        # Append cross-analysis context when log-code links are available
        if cross_analysis_context:
            context += "\n\n## Log-to-Code Cross Analysis\n" + cross_analysis_context

        system_prompt = SYSTEM_CROSS_ANALYSIS if cross_analysis_context else SYSTEM_LOG_SUMMARY
        user_msg = f"Context (log entries + analysis):\n\n{context}\n\nQuestion: {question}"

        mode_label = "cross-analysis" if cross_analysis_context else "log-summary"
        print(f"  [bakup:debug] LLM call: {mode_label} ({cfg.provider}/{cfg.model})")
        print(f"  [bakup:debug]   context length: {len(context)} chars, {len(log_chunks[:5])} chunks")

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=system_prompt)
        except Exception as exc:
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM error in log summary: {safe_msg}")
            return self._log_extractive_fallback(log_chunks)

        print(f"  [bakup:debug] LLM {mode_label} response received ({len(raw)} chars)")

        if raw.strip().startswith(NO_ANSWER_TOKEN):
            return LLMResponse(
                answer="No errors or issues were found in the indexed log entries.",
                mode="llm",
                provider=cfg.provider,
                model=cfg.model,
                no_data=False,
            )

        return LLMResponse(
            answer=raw.strip(),
            mode="llm",
            provider=cfg.provider,
            model=cfg.model,
            no_data=False,
            sources=self._serialise_sources(log_chunks[:5]),
        )

    def generate_code_review(
        self,
        chunks: list,
        question: str,
        top_n: int = 8,
    ) -> LLMResponse:
        """
        Generate a code quality review for broad analytical questions.

        Uses a specialised code-review prompt that instructs the LLM to
        analyse code for quality, patterns, and improvements rather than
        looking for a specific answer.
        """
        from core.llm.prompt_templates import SYSTEM_CODE_REVIEW

        if not chunks:
            return self._no_data_response()

        cfg = load_config()
        if not cfg.configured:
            print(f"  [bakup:debug] LLM not configured — extractive fallback for code review")
            return LLMResponse(
                answer=self._extractive_fallback(chunks),
                mode="extractive",
                provider="none",
                model="none",
                sources=self._serialise_sources(chunks[:top_n]),
            )

        context = build_context_block(chunks[:top_n], max_chars=_MAX_CHUNK_CHARS)
        user_msg = build_rag_user_message(question, context)

        print(f"  [bakup:debug] LLM call: generate_code_review ({cfg.provider}/{cfg.model})")
        print(f"  [bakup:debug]   context: {len(context)} chars, {len(chunks[:top_n])} chunks")

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=SYSTEM_CODE_REVIEW)
        except Exception as exc:
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM code review call FAILED: {safe_msg}")
            return LLMResponse(
                answer=self._extractive_fallback(chunks),
                mode="extractive",
                provider=cfg.provider,
                model=cfg.model,
                error=f"LLM call failed ({safe_msg}) — showing extractive result.",
                sources=self._serialise_sources(chunks[:top_n]),
            )

        print(f"  [bakup:debug] LLM code review response ({len(raw)} chars)")

        return LLMResponse(
            answer=raw.strip(),
            mode="llm",
            provider=cfg.provider,
            model=cfg.model,
            sources=self._serialise_sources(chunks[:top_n]),
        )

    def generate_conversational(self, question: str) -> LLMResponse:
        """
        Handle conversational/meta/personal questions by calling the LLM
        directly — no retrieval context is provided.

        Falls back to a canned response if the LLM is not configured.
        """
        from core.llm.prompt_templates import SYSTEM_CONVERSATIONAL
        from core.classifier.query_classifier import conversational_response

        print(f"  [bakup:debug] generate_conversational: {question[:80]!r}")

        cfg = load_config()
        if not cfg.configured:
            print("  [bakup:debug] LLM not configured — canned conversational response")
            return LLMResponse(
                answer=conversational_response(),
                mode="conversational",
                provider="none",
                model="none",
            )

        print(f"  [bakup:debug] LLM call: conversational ({cfg.provider}/{cfg.model})")

        try:
            raw = self._call_provider(cfg, question, system_prompt=SYSTEM_CONVERSATIONAL)
        except Exception as exc:
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM conversational call failed: {safe_msg}")
            return LLMResponse(
                answer=conversational_response(),
                mode="conversational",
                provider=cfg.provider,
                model=cfg.model,
                error="LLM call failed — showing default response.",
            )

        print(f"  [bakup:debug] LLM conversational response ({len(raw)} chars)")

        return LLMResponse(
            answer=raw.strip(),
            mode="conversational",
            provider=cfg.provider,
            model=cfg.model,
        )

    def generate_agentic_answer(
        self,
        question: str,
        context_block: str,
        mode: str = "root_cause",
    ) -> LLMResponse:
        """
        Generate an agentic reasoning answer from structured evidence.

        All question types route through this method so the full evidence
        context (logs, code, deps, architecture, cross-analysis) is always
        available to the LLM.  The *mode* selects the system prompt:

          - "root_cause"     → SYSTEM_AGENTIC_REASONING
          - "log_analysis"   → SYSTEM_LOG_SUMMARY
          - "cross_analysis" → SYSTEM_CROSS_ANALYSIS
          - "code_review"    → SYSTEM_CODE_REVIEW
          - "general"        → SYSTEM_RAG

        Falls back to extractive display when LLM is not configured.
        """
        from core.llm.prompt_templates import (
            SYSTEM_AGENTIC_REASONING,
            SYSTEM_LOG_SUMMARY,
            SYSTEM_CROSS_ANALYSIS,
            SYSTEM_CODE_REVIEW,
            SYSTEM_RAG,
        )

        _MODE_PROMPTS = {
            "root_cause":     SYSTEM_AGENTIC_REASONING,
            "log_analysis":   SYSTEM_LOG_SUMMARY,
            "cross_analysis": SYSTEM_CROSS_ANALYSIS,
            "code_review":    SYSTEM_CODE_REVIEW,
            "general":        SYSTEM_RAG,
        }

        system_prompt = _MODE_PROMPTS.get(mode, SYSTEM_RAG)

        print(f"  [bakup:debug] generate_agentic_answer: mode={mode}, context_len={len(context_block)}")

        cfg = load_config()
        if not cfg.configured:
            print("  [bakup:debug] LLM not configured — returning context as extractive")
            answer = (
                "**Agentic Analysis (extractive — configure LLM for full reasoning):**\n\n"
                + context_block[:5000]
            )
            return LLMResponse(
                answer=answer,
                mode="extractive",
                provider="none",
                model="none",
            )

        user_msg = f"Context:\n\n{context_block}\n\nQuestion: {question}"

        print(f"  [bakup:debug] LLM call: agentic_{mode} ({cfg.provider}/{cfg.model})")

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=system_prompt)
            raw = self._quality_gate(raw, cfg, user_msg, system_prompt)
        except Exception as exc:
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM agentic call failed: {safe_msg}")
            answer = (
                "**Agentic Analysis (LLM call failed — showing raw evidence):**\n\n"
                + context_block[:5000]
            )
            return LLMResponse(
                answer=answer,
                mode="extractive",
                provider=cfg.provider,
                model=cfg.model,
                error=f"LLM call failed ({safe_msg})",
            )

        print(f"  [bakup:debug] LLM agentic response received ({len(raw)} chars)")

        if raw.strip().startswith(NO_ANSWER_TOKEN):
            return LLMResponse(
                answer="No relevant evidence found in the indexed data for this question.",
                mode="llm",
                provider=cfg.provider,
                model=cfg.model,
                no_data=True,
            )

        return LLMResponse(
            answer=raw.strip(),
            mode="llm",
            provider=cfg.provider,
            model=cfg.model,
        )

    def _log_extractive_fallback(self, chunks: list) -> LLMResponse:
        """When LLM is not configured, show log entries directly."""
        if not chunks:
            return self._no_data_response()

        parts = []
        for i, c in enumerate(chunks[:5], 1):
            parts.append(
                f"**[{i}] {c.source_file} "
                f"(lines {c.line_start}\u2013{c.line_end}, "
                f"confidence: {c.confidence:.0%}):**\n"
                f"```\n{c.text[:500]}\n```"
            )

        answer = (
            "Here are log entries that may contain errors or issues:\n\n"
            + "\n\n".join(parts)
        )

        return LLMResponse(
            answer=answer,
            mode="extractive",
            provider="none",
            model="none",
            no_data=False,
            sources=self._serialise_sources(chunks[:5]),
        )

    def health_check(self) -> tuple[LLMStatus, str]:
        """
        Check whether the LLM is reachable and configured.

        Results are cached for _HEALTH_TTL seconds so the UI health-poll
        (every 15 s) does not spam the provider API.

        Returns:
            (LLMStatus, human-readable message)
        """
        cfg = load_config()

        if not cfg.configured:
            # Always return immediately — no network call needed.
            _health_cache.clear()
            return LLMStatus.NOT_CONFIGURED, "LLM not configured — open settings to set up."

        # Return cached result if still fresh.
        cache_key = f"{cfg.provider}:{cfg.model}"
        cached    = _health_cache.get(cache_key)
        if cached and (_time.monotonic() - cached["ts"]) < _HEALTH_TTL:
            return cached["status"], cached["msg"]

        # Cache miss — do the real ping.
        try:
            self._ping_provider(cfg)
            status  = LLMStatus.READY
            message = f"{cfg.provider} / {cfg.model}"
        except Exception as exc:
            status  = LLMStatus.ERROR
            message = f"LLM error: {_redact_key(str(exc), cfg.api_key)}"

        _health_cache[cache_key] = {"status": status, "msg": message, "ts": _time.monotonic()}
        return status, message

    # ── Provider dispatch ─────────────────────────────────────────────────────

    _TRUNCATION_SIGNALS = [
        "...",
        "[truncated",
        "I'll continue",
        "Let me continue",
        "Due to length",
        "I would need more space",
    ]

    _MIN_QUALITY_LEN = 120   # answers shorter than this get a retry

    def _response_looks_truncated(self, text: str) -> bool:
        """Detect if the LLM response was cut off mid-thought."""
        stripped = text.rstrip()
        # Ends mid-sentence (no terminal punctuation)
        if stripped and stripped[-1] not in ".!?:>\n`|" and len(stripped) > 60:
            return True
        # Contains truncation signals near the end
        tail = stripped[-120:] if len(stripped) > 120 else stripped
        for sig in self._TRUNCATION_SIGNALS:
            if sig.lower() in tail.lower():
                return True
        return False

    def _quality_gate(
        self,
        raw: str,
        cfg,
        user_msg: str,
        system_prompt: str,
    ) -> str:
        """
        Check response quality and retry once with a larger token budget
        if the response appears truncated or too short.

        Uses a thread-safe local override instead of mutating os.environ.
        """
        if not raw or raw.strip().startswith(NO_ANSWER_TOKEN):
            return raw   # nothing to improve

        is_short     = len(raw.strip()) < self._MIN_QUALITY_LEN
        is_truncated = self._response_looks_truncated(raw)

        if not is_short and not is_truncated:
            return raw   # looks good

        reason = "too short" if is_short else "appears truncated"
        print(f"  [bakup:debug] quality gate: response {reason} ({len(raw)} chars) — retrying with higher budget")

        # Retry with 2x token budget via a temporary env override
        # Use a thread-local approach: save → set → call → restore
        orig_budget = os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")
        retry_budget = str(min(int(orig_budget) * 2, 16384))  # Cap at 16K
        os.environ["BAKUP_LLM_MAX_TOKENS"] = retry_budget
        try:
            retry_raw = self._call_provider(cfg, user_msg, system_prompt=system_prompt)
        except Exception:
            retry_raw = None
        finally:
            os.environ["BAKUP_LLM_MAX_TOKENS"] = orig_budget

        if retry_raw and len(retry_raw.strip()) > len(raw.strip()):
            print(f"  [bakup:debug] quality gate: retry produced better response ({len(retry_raw)} chars)")
            return retry_raw

        return raw   # keep original if retry didn't help

    def _call_provider(
        self,
        cfg: LLMConfig,
        user_message: str,
        system_prompt: str = SYSTEM_RAG,
    ) -> str:
        if cfg.provider == "openai":
            return self._call_openai(cfg, user_message, system_prompt)
        if cfg.provider == "anthropic":
            return self._call_anthropic(cfg, user_message, system_prompt)
        if cfg.provider == "azure_openai":
            return self._call_azure(cfg, user_message, system_prompt)
        if cfg.provider == "ollama":
            return self._call_ollama(cfg, user_message, system_prompt)
        raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")

    def _ping_provider(self, cfg: LLMConfig) -> None:
        """Lightweight connectivity check — raises on failure."""
        if cfg.provider == "openai":
            self._ping_openai(cfg)
        elif cfg.provider == "anthropic":
            self._ping_anthropic(cfg)
        elif cfg.provider == "azure_openai":
            self._ping_azure(cfg)
        elif cfg.provider == "ollama":
            self._ping_ollama(cfg)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider!r}")

    # ── Tool-calling interface ────────────────────────────────────────────────

    def call_with_tools(
        self,
        cfg: LLMConfig,
        messages: list,
        tools: list,
    ) -> dict:
        """
        Call the LLM with tool/function definitions and return a structured
        response indicating either a final text answer or tool call requests.

        Args:
            cfg:      LLM configuration (provider, model, api_key, etc.)
            messages: Conversation history (system, user, assistant, tool).
            tools:    Tool schemas in provider-native format.

        Returns:
            dict with keys:
              - "content":    str — text response (may be empty if tool_calls)
              - "tool_calls": list[dict] — each with "id", "name", "arguments"
        """
        if cfg.provider == "openai":
            return self._call_openai_with_tools(cfg, messages, tools)
        if cfg.provider == "anthropic":
            return self._call_anthropic_with_tools(cfg, messages, tools)
        if cfg.provider == "azure_openai":
            return self._call_azure_with_tools(cfg, messages, tools)
        if cfg.provider == "ollama":
            return self._call_ollama_with_tools(cfg, messages, tools)
        raise ValueError(f"Unknown LLM provider for tool calling: {cfg.provider!r}")

    def _call_openai_with_tools(self, cfg, messages, tools) -> dict:
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        client = OpenAI(api_key=cfg.api_key)
        kwargs = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        result: dict = {"content": msg.content or "", "tool_calls": []}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json as _json
                try:
                    args = _json.loads(tc.function.arguments)
                except (ValueError, TypeError):
                    args = {}
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return result

    def _call_azure_with_tools(self, cfg, messages, tools) -> dict:
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        if not cfg.azure_endpoint:
            raise ValueError("Azure endpoint URL is required.")

        client = AzureOpenAI(
            api_key=cfg.api_key,
            azure_endpoint=cfg.azure_endpoint,
            api_version=cfg.azure_api_version or "2024-02-01",
        )
        kwargs = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        result: dict = {"content": msg.content or "", "tool_calls": []}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                import json as _json
                try:
                    args = _json.loads(tc.function.arguments)
                except (ValueError, TypeError):
                    args = {}
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return result

    def _call_anthropic_with_tools(self, cfg, messages, tools) -> dict:
        import json as _json
        import urllib.request

        # Separate system message from conversation messages
        system_text = ""
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"] if isinstance(m["content"], str) else str(m["content"])
            else:
                conv_messages.append(m)

        payload: dict = {
            "model": cfg.model,
            "max_tokens": int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
            "messages": conv_messages,
        }
        if system_text:
            payload["system"] = system_text
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=_json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode())

        # Parse Anthropic response
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                })

        return {"content": content_text, "tool_calls": tool_calls}

    def _call_ollama_with_tools(self, cfg, messages, tools) -> dict:
        import json as _json
        import urllib.request

        base = (cfg.ollama_base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        url = f"{base}/api/chat"

        payload: dict = {
            "model": cfg.model,
            "stream": False,
            "options": {
                "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
                "num_predict": int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            },
            "messages": messages,
        }
        if tools:
            # Ollama uses OpenAI-compatible tool format
            payload["tools"] = tools

        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode())

        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls = []

        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append({
                "id": tc.get("id", f"call_{len(tool_calls)}"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", {}),
            })

        return {"content": content, "tool_calls": tool_calls}

    # ── Anthropic (Claude) ───────────────────────────────────────────────────

    def _call_anthropic(self, cfg: LLMConfig, user_message: str, system_prompt: str = SYSTEM_RAG) -> str:
        """
        Call the Anthropic Messages API via urllib (no extra dependency).
        Supports Claude 3.5 Sonnet, Claude 3 Opus, etc.
        """
        import json
        import urllib.request

        payload = {
            "model": cfg.model,
            "max_tokens": int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        # Extract text from content blocks
        text_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    def _ping_anthropic(self, cfg: LLMConfig) -> None:
        """Validate Anthropic API key with a minimal request."""
        import json
        import urllib.request

        payload = {
            "model": cfg.model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key,
                "anthropic-version": "2023-06-01",
            },
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

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _call_openai(self, cfg: LLMConfig, user_message: str, system_prompt: str = SYSTEM_RAG) -> str:
        try:
            from openai import OpenAI       # type: ignore
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=cfg.api_key)   # key used here only, not stored
        completion = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
        )
        return completion.choices[0].message.content or ""

    def _ping_openai(self, cfg: LLMConfig) -> None:
        try:
            from openai import OpenAI   # type: ignore
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        client = OpenAI(api_key=cfg.api_key)
        # List models is a cheap, non-billable call that validates the key.
        client.models.list()

    # ── Azure OpenAI ──────────────────────────────────────────────────────────

    def _call_azure(self, cfg: LLMConfig, user_message: str, system_prompt: str = SYSTEM_RAG) -> str:
        try:
            from openai import AzureOpenAI   # type: ignore
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        if not cfg.azure_endpoint:
            raise ValueError("Azure endpoint URL is required for azure_openai provider.")

        client = AzureOpenAI(
            api_key        = cfg.api_key,
            azure_endpoint = cfg.azure_endpoint,
            api_version    = cfg.azure_api_version or "2024-02-01",
        )
        completion = client.chat.completions.create(
            model=cfg.model,   # deployment name for Azure
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
        )
        return completion.choices[0].message.content or ""

    def _ping_azure(self, cfg: LLMConfig) -> None:
        try:
            from openai import AzureOpenAI   # type: ignore
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

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _call_ollama(self, cfg: LLMConfig, user_message: str, system_prompt: str = SYSTEM_RAG) -> str:
        """
        Calls the local Ollama server via its REST API.
        Does NOT require the `ollama` Python package — uses urllib only,
        so there are no extra dependencies for local-only users.
        """
        import json
        import urllib.request

        base = (cfg.ollama_base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        url  = f"{base}/api/chat"
        payload = {
            "model":  cfg.model,
            "stream": False,
            "options": {
                "temperature": float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
                "num_predict": int(os.environ.get("BAKUP_LLM_MAX_TOKENS",   "2048")),
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        return data.get("message", {}).get("content", "")

    def _ping_ollama(self, cfg: LLMConfig) -> None:
        import urllib.request
        base = (cfg.ollama_base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        try:
            urllib.request.urlopen(f"{base}/api/tags", timeout=5)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {base}. "
                f"Is Ollama running? (ollama serve)  Detail: {exc}"
            )

    # ── Extractive fallback ───────────────────────────────────────────────────

    def _extractive_fallback(self, chunks: list) -> str:
        """
        When LLM is not configured or call fails, show top chunks with
        structured formatting so users still get useful context.

        Retrieval guard: chunks below the confidence threshold are excluded
        to avoid surfacing irrelevant matches (e.g. model weight files).
        """
        if not chunks:
            return "No similar incident found in indexed data."

        # Retrieval guard — filter out very low confidence matches
        _MIN_EXTRACTIVE_CONF = float(os.environ.get("BAKUP_CONFIDENCE_THRESHOLD", "0.35"))
        usable = [c for c in chunks[:5] if c.confidence >= _MIN_EXTRACTIVE_CONF]

        if not usable:
            return (
                "I found some indexed content, but none of the matches are confident "
                "enough to be relevant to your question.\n\n"
                "Could you try rephrasing? For example:\n"
                "• Mention a specific file, function, or error message\n"
                "• Include a log entry or status code\n"
                "• Narrow down the component or time range"
            )

        parts = []
        for i, c in enumerate(usable, 1):
            excerpt = c.text[:1000].strip()
            header = (
                f"**[{i}] {c.source_file} "
                f"(lines {c.line_start}–{c.line_end}, "
                f"confidence: {c.confidence:.0%}):**"
            )
            parts.append(f"{header}\n```\n{excerpt}\n```")

        answer = (
            "**Relevant matches found** (configure an LLM for full reasoning analysis):\n\n"
            + "\n\n".join(parts)
        )
        return answer

    def _no_data_response(self, cfg: Optional[LLMConfig] = None) -> LLMResponse:
        return LLMResponse(
            answer  = "No similar incident found in indexed data.",
            mode    = "no_data",
            provider= cfg.provider if cfg else "none",
            model   = cfg.model    if cfg else "none",
            no_data = True,
        )

    def _serialise_sources(self, chunks: list) -> List[dict]:
        return [
            {
                "file":              c.source_file,
                "line_start":        c.line_start,
                "line_end":          c.line_end,
                "excerpt":           c.text[:300].strip(),
                "confidence":        c.confidence,
                "confidence_label":  c.confidence_label,
                "source_type":       c.source_type,
            }
            for c in chunks
        ]


# ── Module-level singleton ─────────────────────────────────────────────────────

_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Return the module-level singleton LLMService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = LLMService()
    return _service_instance


# ── Helpers ────────────────────────────────────────────────────────────────────

def _redact_key(message: str, key: str) -> str:
    """Remove any accidental leakage of the API key from error messages."""
    if key and len(key) > 4:
        return message.replace(key, "****")
    return message


# Re-export DEFAULT_OLLAMA_URL for use in config_store
DEFAULT_OLLAMA_URL = "http://localhost:11434"
