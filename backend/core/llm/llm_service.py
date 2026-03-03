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

_MAX_CHUNK_CHARS = 800   # Chars per chunk sent to LLM — guards against ctx overflow


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
    ) -> LLMResponse:
        """
        Summarise errors/issues found in log chunks.

        Used when a log-style query returns low-confidence embedding matches
        but keyword matching finds actual log entries. Instead of returning
        "No similar incident found", we pass the entries to the LLM and let
        it summarise whether any errors exist.

        Falls back to an extractive display of log entries when the LLM is
        not configured.
        """
        from core.llm.prompt_templates import SYSTEM_LOG_SUMMARY

        print(f"  [bakup:debug] generate_log_summary called with {len(log_chunks)} chunks")

        cfg = load_config()
        if not cfg.configured:
            print("  [bakup:debug] LLM not configured — extractive log fallback")
            return self._log_extractive_fallback(log_chunks)

        context = build_context_block(log_chunks[:5], max_chars=_MAX_CHUNK_CHARS)
        user_msg = f"Context (log entries):\n\n{context}\n\nQuestion: {question}"

        print(f"  [bakup:debug] LLM call: generate_log_summary ({cfg.provider}/{cfg.model})")
        print(f"  [bakup:debug]   context length: {len(context)} chars, {len(log_chunks[:5])} chunks")

        try:
            raw = self._call_provider(cfg, user_msg, system_prompt=SYSTEM_LOG_SUMMARY)
        except Exception as exc:
            safe_msg = _redact_key(str(exc), cfg.api_key)
            print(f"  [bakup:debug] LLM error in log summary: {safe_msg}")
            return self._log_extractive_fallback(log_chunks)

        print(f"  [bakup:debug] LLM log-summary response received ({len(raw)} chars)")

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

    def _call_provider(
        self,
        cfg: LLMConfig,
        user_message: str,
        system_prompt: str = SYSTEM_RAG,
    ) -> str:
        if cfg.provider == "openai":
            return self._call_openai(cfg, user_message, system_prompt)
        if cfg.provider == "azure_openai":
            return self._call_azure(cfg, user_message, system_prompt)
        if cfg.provider == "ollama":
            return self._call_ollama(cfg, user_message, system_prompt)
        raise ValueError(f"Unknown LLM provider: {cfg.provider!r}")

    def _ping_provider(self, cfg: LLMConfig) -> None:
        """Lightweight connectivity check — raises on failure."""
        if cfg.provider == "openai":
            self._ping_openai(cfg)
        elif cfg.provider == "azure_openai":
            self._ping_azure(cfg)
        elif cfg.provider == "ollama":
            self._ping_ollama(cfg)
        else:
            raise ValueError(f"Unknown provider: {cfg.provider!r}")

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
            max_tokens=int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "512")),
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
            max_tokens=int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "512")),
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
                "num_predict": int(os.environ.get("BAKUP_LLM_MAX_TOKENS",   "512")),
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
        if not chunks:
            return "No similar incident found in indexed data."
        top = chunks[0]
        excerpt = top.text[:600].strip()
        answer = (
            f"Most relevant match: {top.source_file} "
            f"lines {top.line_start}–{top.line_end} "
            f"(confidence: {top.confidence:.2f}):\n\n{excerpt}"
        )
        if len(top.text) > 600:
            answer += "\n\n[truncated — see source for full context]"
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
