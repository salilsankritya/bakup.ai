"""
core/retrieval/rag.py
─────────────────────────────────────────────────────────────────────────────
Retrieval-Augmented Generation pipeline.

Query flow (updated):
    ┌──────────────┐
    │ User question│
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Classifier  │──► greeting  → polite one-liner  (no RAG, no LLM)
    │  (regex)     │──► off_topic → scope-guard reply  (no RAG, no LLM)
    │              │──► project   → continue ▼
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Embed query │
    │  Retrieve    │
    │  Rank        │
    └──────┬───────┘
           │
    ┌──────▼───────────────────────┐
    │  Threshold check             │
    │  ├─ no results at all        │──► "No similar incident found."
    │  ├─ all below threshold      │──► clarification question (LLM or canned)
    │  └─ relevant results exist   │──► generate answer (LLM or extractive)
    └──────────────────────────────┘

Operating modes:
  1. Extractive (default, always available):
     Returns the most relevant chunks directly as the answer.
     No LLM. No hallucination possible. Confidence from embedding similarity.

  2. LLM-augmented (opt-in, requires configured LLM provider):
     Passes retrieved context to the LLM with a strict citation-only system
     prompt. The model is instructed to respond with NO_ANSWER if the context
     is insufficient — that signal is checked and honoured.

  3. Clarification (new):
     When retrieval finds chunks but all are below the confidence threshold,
     the system asks the user to refine their question instead of guessing.

Neither mode will produce a fabricated answer. If the data is not there,
we say so. This is an intentional design constraint.
"""

from __future__ import annotations

import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.classifier.query_classifier import (
    QueryCategory,
    classify_query,
    greeting_response,
    off_topic_response,
)
from core.embeddings.embedder import embed_query
from core.retrieval.ranker import RankedResult, has_relevant_results, rank_results, top_relevant
from core.retrieval.vector_store import query_chunks, keyword_search, collection_count, severity_search
from core.analysis.confidence import calculate_confidence, is_broad_query
from core.analysis.trends import analyze_error_trends
from core.analysis.clusters import cluster_log_events
from core.analysis.file_aggregation import aggregate_by_file

# Signal the LLM must produce when it cannot answer from context
_NO_ANSWER_SIGNAL = "NO_ANSWER"

# How many top results to include in the LLM context window
_LLM_CONTEXT_RESULTS = 5

# Max chars per chunk included in LLM context (avoids overflowing context window)
_MAX_CHUNK_CHARS_IN_CONTEXT = 800


# ── Log-query keyword detection ───────────────────────────────────────────────

_LOG_QUERY_WORDS = {
    "error", "exception", "traceback", "failed", "failure", "crash",
    "warning", "warn", "fatal", "critical", "timeout", "panic",
    "null", "none", "undefined", "oom", "stacktrace", "stack trace",
    "segfault", "deadlock", "bug", "issue",
}

# Map query-level keywords → content-level keywords to search in indexed docs
_KEYWORD_MAP = {
    "error":     ["ERROR", "Error", "error"],
    "exception": ["Exception", "exception", "Traceback"],
    "traceback": ["Traceback", "traceback"],
    "failed":    ["Failed", "failed", "FAILED"],
    "failure":   ["Failure", "failure", "FAILURE"],
    "crash":     ["crash", "Crash", "CRASH"],
    "warning":   ["WARN", "WARNING", "Warning"],
    "warn":      ["WARN", "WARNING", "Warning"],
    "fatal":     ["FATAL", "Fatal"],
    "critical":  ["CRITICAL", "Critical"],
    "timeout":   ["timeout", "Timeout", "TIMEOUT", "timed out"],
    "null":      ["NullPointerException", "NoneType", "null"],
    "none":      ["NoneType"],
    "panic":     ["panic", "PANIC"],
    "oom":       ["OutOfMemoryError", "OOM", "out of memory"],
}


def _is_log_query(question: str) -> bool:
    """Check if the question is about log errors/issues."""
    q_lower = question.lower()
    # Direct keyword match
    if any(kw in q_lower for kw in _LOG_QUERY_WORDS):
        return True
    # Also detect questions about "logs" in general
    if re.search(r"\blog(s|file)?\b", q_lower):
        return True
    return False


def _extract_search_keywords(question: str) -> List[str]:
    """
    For log-style queries, return keywords to search in indexed content.
    Maps query-level words to content-level keywords that appear in actual logs.
    """
    q_lower = question.lower()
    found: List[str] = []

    for query_kw, content_kws in _KEYWORD_MAP.items():
        if query_kw in q_lower:
            found.extend(content_kws)

    # If the question is about "logs" generically, search for common log levels
    if re.search(r"\blog(s|file)?\b", q_lower) and not found:
        found.extend(["ERROR", "Exception", "WARN", "Traceback", "FATAL"])

    # Deduplicate while preserving order
    return list(dict.fromkeys(found))


# ── Response model ─────────────────────────────────────────────────────────────

@dataclass
class Source:
    file: str
    line_start: int
    line_end: int
    excerpt: str            # First 300 chars of the chunk
    confidence: float
    confidence_label: str   # "high" | "medium" | "low"
    source_type: str        # "code" | "log"


@dataclass
class RAGResponse:
    answer: str
    confidence: float       # Score of the top-ranked source
    no_data: bool           # True when no relevant match exists
    mode: str               # "extractive" | "llm"
    sources: List[Source]
    debug_trace: Optional[List[dict]] = None   # Step-by-step trace (when debug=True)


# ── Main entry point ───────────────────────────────────────────────────────────

def answer_question(
    question: str,
    namespace: str,
    top_k: int = 8,
    debug: bool = False,
) -> RAGResponse:
    """
    Answer a plain-English question about an indexed project.

    Flow:
        1. Classify the question (project / greeting / off_topic).
        2. For greetings and off-topic → return canned response immediately.
        3. For project questions → embed, retrieve, rank, answer.
        4. For log-style queries → enhance retrieval with keyword search.
        5. If retrieval returns low-confidence results:
           - Log queries → pass to LLM for error summarisation.
           - Other queries → ask for clarification.

    Args:
        question:  The user's question.
        namespace: Project namespace used when indexing.
        top_k:     Number of candidates to retrieve from ChromaDB.
        debug:     When True, attach step-by-step trace to the response.

    Returns:
        RAGResponse with answer, sources, confidence, no_data flag, and
        optionally debug_trace.
    """
    steps: List[dict] = []
    t0 = time.perf_counter()

    def _step(step: str, message: str, **data):
        entry = {"step": step, "message": message, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if data:
            entry["data"] = data
        steps.append(entry)
        print(f"  [bakup:pipeline] {step}: {message}")

    def _result(resp: RAGResponse) -> RAGResponse:
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        _step("done", f"Pipeline complete ({total_ms:.0f}ms)", total_ms=total_ms)
        if debug:
            resp.debug_trace = steps
        return resp

    if not question.strip():
        return RAGResponse(
            answer="Question cannot be empty.",
            confidence=0.0,
            no_data=True,
            mode="extractive",
            sources=[],
        )

    # ── Step 1: Classify ──────────────────────────────────────────────────────
    _step("classify", "Classifying question...")
    category = classify_query(question)
    _step("classify_done", f"Category: {category.value}")

    if category == QueryCategory.GREETING:
        _step("response", "Greeting detected — returning polite response")
        print(f"  [bakup:debug] classification=greeting | docs_retrieved=0 | llm_called=no")
        return _result(RAGResponse(
            answer=greeting_response(),
            confidence=1.0,
            no_data=False,
            mode="greeting",
            sources=[],
        ))

    if category == QueryCategory.CONVERSATIONAL:
        _step("response", "Conversational/meta — calling LLM directly (no retrieval)")
        from core.llm.llm_service import get_llm_service
        svc = get_llm_service()
        llm_resp = svc.generate_conversational(question)
        _step("llm_direct", f"LLM responded (mode={llm_resp.mode}, {len(llm_resp.answer)} chars)")
        print(f"  [bakup:debug] classification=conversational | docs_retrieved=0 | llm_called={'yes' if llm_resp.mode == 'conversational' else 'no'}")
        return _result(RAGResponse(
            answer=llm_resp.answer,
            confidence=1.0,
            no_data=False,
            mode="conversational",
            sources=[],
        ))

    if category == QueryCategory.OFF_TOPIC:
        _step("response", "Off-topic detected — returning scope guard")
        print(f"  [bakup:debug] classification=off_topic | docs_retrieved=0 | llm_called=no")
        return _result(RAGResponse(
            answer=off_topic_response(),
            confidence=0.0,
            no_data=True,
            mode="off_topic",
            sources=[],
        ))

    # ── Step 2: Embed & retrieve ──────────────────────────────────────────────
    _step("embed", "Embedding query for semantic search...")
    query_vec = embed_query(question)

    count = collection_count(namespace)
    _step("retrieve", f"Searching {count} indexed documents...")
    raw_chunks = query_chunks(query_vec, namespace=namespace, top_k=top_k)
    _step("retrieve_done", f"Semantic search returned {len(raw_chunks)} candidates")

    # ── Step 2b: Keyword-enhanced search for log queries ──────────────────────
    is_log_q = _is_log_query(question)
    kw_chunks = []
    sev_chunks = []

    if is_log_q:
        search_keywords = _extract_search_keywords(question)
        if search_keywords:
            _step("keyword_search", f"Log-style query — enhancing with keyword search: {search_keywords[:5]}")
            kw_chunks = keyword_search(namespace, search_keywords, top_k=top_k)
            _step("keyword_done", f"Keyword search found {len(kw_chunks)} additional matches")

        # Also pull error-severity chunks for cross-file distribution
        _step("severity_search", "Searching for error-severity chunks across all files...")
        sev_chunks = severity_search(namespace, severity="error", top_k=top_k * 2)
        _step("severity_done", f"Severity search found {len(sev_chunks)} error chunks")

    # ── Step 2c: Merge & deduplicate ──────────────────────────────────────────
    all_raw = raw_chunks + kw_chunks + sev_chunks
    seen = set()
    deduped = []
    for c in all_raw:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    if len(deduped) != len(all_raw):
        _step("dedup", f"Deduplicated: {len(all_raw)} → {len(deduped)} unique chunks")

    # ── Step 3: Rank by confidence ────────────────────────────────────────────
    _step("rank", "Ranking results by relevance...")
    ranked = rank_results(deduped)

    if ranked:
        _step("rank_done",
              f"Top: {ranked[0].confidence:.0%} ({ranked[0].confidence_label}) in {ranked[0].source_file}, "
              f"Bottom: {ranked[-1].confidence:.0%}",
              retrieved_count=len(ranked),
              top_confidence=ranked[0].confidence,
              top_file=ranked[0].source_file)
    else:
        _step("rank_done", "No results to rank")

    # ── Step 4: Threshold check ───────────────────────────────────────────────
    if not has_relevant_results(ranked):
        if ranked:
            # ── Log queries: analyse + summarise log entries instead of giving up
            if is_log_q:
                log_entries = [r for r in ranked if r.source_type == "log"]
                analysis_chunks = log_entries if log_entries else ranked

                # ── Run analysis pipeline ─────────────────────────────────
                _step("trend_analysis", f"Detecting error trends across {len(analysis_chunks)} chunks...")
                trend_report = analyze_error_trends(analysis_chunks)
                _step("trend_done",
                      f"Trends: {trend_report.hourly_counts.total_errors} errors, "
                      f"{len(trend_report.spikes)} spikes, "
                      f"{len(trend_report.repeating_failures)} repeating failures",
                      trends=trend_report.to_dict())

                _step("cluster_analysis", "Clustering log events by temporal proximity and keywords...")
                cluster_report = cluster_log_events(analysis_chunks)
                _step("cluster_done",
                      f"Found {cluster_report.cluster_count} incident cluster(s)",
                      clusters=cluster_report.to_dict())

                _step("confidence_scoring", "Computing multi-factor confidence score...")
                conf_result = calculate_confidence(analysis_chunks, question=question)
                _step("confidence_done",
                      f"Confidence: {conf_result.confidence_score:.0%} ({conf_result.confidence_level})",
                      confidence=conf_result.to_dict())

                # File-level error aggregation
                _step("file_aggregation", f"Aggregating errors across files from {len(analysis_chunks)} chunks...")
                agg_report = aggregate_by_file(analysis_chunks)
                _step("file_aggregation_done",
                      f"{agg_report.files_affected} file(s), {agg_report.total_errors} error(s), dominant: {agg_report.dominant_file or 'N/A'}")

                _step("log_summarize",
                      f"Low embedding confidence but {len(analysis_chunks)} log entries found — "
                      "passing enriched context to LLM for structured summary...")
                from core.llm.llm_service import get_llm_service
                svc = get_llm_service()
                llm_resp = svc.generate_log_summary(
                    question,
                    analysis_chunks[:_LLM_CONTEXT_RESULTS],
                    trend_summary=trend_report.summary_text(),
                    cluster_summary=cluster_report.summary_text(),
                    confidence_summary=conf_result.reasoning,
                    file_aggregation_summary=agg_report.summary_text,
                )
                return _result(RAGResponse(
                    answer=llm_resp.answer,
                    confidence=conf_result.confidence_score,
                    no_data=llm_resp.no_data,
                    mode=llm_resp.mode,
                    sources=_build_sources(analysis_chunks[:5]),
                ))

            # ── Non-log queries: ask for clarification ────────────────────────
            _step("clarify", "Low confidence — asking LLM for clarification...")
            from core.llm.llm_service import get_llm_service
            svc      = get_llm_service()
            llm_resp = svc.generate_clarification(
                question=question,
                near_miss_chunks=ranked[:5],
                best_confidence=ranked[0].confidence,
            )
            return _result(RAGResponse(
                answer     = llm_resp.answer,
                confidence = ranked[0].confidence,
                no_data    = False,   # We ARE returning useful content (clarification)
                mode       = "clarification",
                sources    = _build_sources(ranked[:3]),
            ))

        # Truly empty — no chunks at all
        _step("no_data", "No indexed content matched the query")
        return _result(RAGResponse(
            answer="No similar incident found.",
            confidence=0.0,
            no_data=True,
            mode="extractive",
            sources=[],
        ))

    # ── Step 5: Generate answer from relevant results ─────────────────────────
    relevant = top_relevant(ranked, n=_LLM_CONTEXT_RESULTS)

    # ── For log queries, run the analysis pipeline even when confidence is high
    if is_log_q:
        _step("trend_analysis", f"Detecting error trends across {len(relevant)} chunks...")
        trend_report = analyze_error_trends(relevant)
        _step("trend_done",
              f"Trends: {trend_report.hourly_counts.total_errors} errors, "
              f"{len(trend_report.spikes)} spikes, "
              f"{len(trend_report.repeating_failures)} repeating failures",
              trends=trend_report.to_dict())

        _step("cluster_analysis", "Clustering log events by temporal proximity and keywords...")
        cluster_report = cluster_log_events(relevant)
        _step("cluster_done",
              f"Found {cluster_report.cluster_count} incident cluster(s)",
              clusters=cluster_report.to_dict())

        _step("confidence_scoring", "Computing multi-factor confidence score...")
        conf_result = calculate_confidence(relevant, question=question)
        _step("confidence_done",
              f"Confidence: {conf_result.confidence_score:.0%} ({conf_result.confidence_level})",
              confidence=conf_result.to_dict())

        # File-level error aggregation
        _step("file_aggregation", f"Aggregating errors across files from {len(relevant)} chunks...")
        agg_report = aggregate_by_file(relevant)
        _step("file_aggregation_done",
              f"{agg_report.files_affected} file(s), {agg_report.total_errors} error(s), dominant: {agg_report.dominant_file or 'N/A'}")

        _step("generate", f"Generating structured log analysis from {len(relevant)} chunks...")
        from core.llm.llm_service import get_llm_service
        svc = get_llm_service()
        llm_resp = svc.generate_log_summary(
            question,
            relevant,
            trend_summary=trend_report.summary_text(),
            cluster_summary=cluster_report.summary_text(),
            confidence_summary=conf_result.reasoning,
            file_aggregation_summary=agg_report.summary_text,
        )
        _step("generate_done", f"Log analysis generated (mode={llm_resp.mode})")

        # Debug: file-level and severity breakdown
        from collections import Counter
        file_names = [getattr(r, 'file_name', '') or r.source_file for r in relevant]
        sev_names = [getattr(r, 'severity', 'unknown') or 'unknown' for r in relevant]
        print(f"  [bakup:debug] classification=project(log) | docs_retrieved={len(ranked)} | "
              f"files_contributing={dict(Counter(file_names))} | "
              f"severity_dist={dict(Counter(sev_names))} | "
              f"confidence={conf_result.confidence_score:.2f}({conf_result.confidence_level}) | "
              f"cross_file_factor={conf_result.factors.get('cross_file', 'N/A')} | "
              f"llm_called={'yes' if llm_resp.mode == 'llm' else 'no (extractive)'}")

        return _result(RAGResponse(
            answer=llm_resp.answer,
            confidence=conf_result.confidence_score,
            no_data=llm_resp.no_data,
            mode=llm_resp.mode,
            sources=_build_sources(relevant),
        ))

    _step("generate", f"Generating answer from {len(relevant)} relevant chunks...")

    from core.llm.llm_service import get_llm_service
    svc      = get_llm_service()
    llm_resp = svc.generate_response(relevant, question)
    _step("generate_done", f"Answer generated (mode={llm_resp.mode})")
    print(f"  [bakup:debug] classification=project | docs_retrieved={len(ranked)} | llm_called={'yes' if llm_resp.mode == 'llm' else 'no (extractive)'}")

    return _result(RAGResponse(
        answer     = llm_resp.answer,
        confidence = relevant[0].confidence,
        no_data    = llm_resp.no_data,
        mode       = llm_resp.mode,
        sources    = _build_sources(relevant),
    ))


# ── Extractive mode ────────────────────────────────────────────────────────────

def _extractive_answer(question: str, relevant: List[RankedResult]) -> RAGResponse:
    """
    Return the most relevant chunks directly as the answer.
    No generation. No hallucination possible.
    """
    top = relevant[0]

    # Build a plain-text summary of the best match
    excerpt = top.text[:600].strip()
    answer = (
        f"Most relevant match found in {top.source_file} "
        f"(lines {top.line_start}–{top.line_end}, "
        f"confidence: {top.confidence:.2f}):\n\n"
        f"{excerpt}"
    )
    if len(top.text) > 600:
        answer += "\n\n[truncated — see source for full context]"

    return RAGResponse(
        answer=answer,
        confidence=top.confidence,
        no_data=False,
        mode="extractive",
        sources=_build_sources(relevant),
    )


# ── LLM-augmented mode ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = None  # Legacy — prompts now live in core.llm.prompt_templates

def _build_llm_context(relevant: List[RankedResult]) -> str:
    """Construct the context block passed to the LLM."""
    from core.llm.prompt_templates import build_context_block
    return build_context_block(relevant, max_chars=_MAX_CHUNK_CHARS_IN_CONTEXT)


def _llm_answer(
    question: str,
    relevant: List[RankedResult],
    model_path: Path,
) -> RAGResponse:
    """
    Run question + context through the local LLM and return a cited answer.
    Falls back to extractive mode if the LLM produces NO_ANSWER or errors.
    """
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        return _extractive_answer(question, relevant)

    context_block = _build_llm_context(relevant)
    user_message = f"Context:\n\n{context_block}\n\nQuestion: {question}"

    llm_context_window = int(os.environ.get("BAKUP_LLM_CONTEXT_WINDOW", "4096"))
    max_tokens = int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "512"))
    temperature = float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1"))

    try:
        from core.llm.prompt_templates import SYSTEM_RAG
        llm = _get_llm(str(model_path), llm_context_window)
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_RAG},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[],
        )
        raw_answer: str = output["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"bakup: LLM inference error: {exc} — falling back to extractive")
        return _extractive_answer(question, relevant)

    # Honour NO_ANSWER signal
    if raw_answer.strip().startswith(_NO_ANSWER_SIGNAL):
        return RAGResponse(
            answer="No similar incident found.",
            confidence=relevant[0].confidence,
            no_data=True,
            mode="llm",
            sources=_build_sources(relevant),
        )

    return RAGResponse(
        answer=raw_answer,
        confidence=relevant[0].confidence,
        no_data=False,
        mode="llm",
        sources=_build_sources(relevant),
    )


# ── LLM singleton ──────────────────────────────────────────────────────────────

_llm_instance = None
_llm_model_path: Optional[str] = None


def _get_llm(model_path: str, context_window: int):
    """Load the llama.cpp model once per process."""
    global _llm_instance, _llm_model_path
    if _llm_instance is None or _llm_model_path != model_path:
        from llama_cpp import Llama  # type: ignore
        print(f"bakup: loading LLM from {model_path}")
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )
        _llm_model_path = model_path
        print("bakup: LLM ready")
    return _llm_instance


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_llm_model() -> Optional[Path]:
    """
    Return the path to the LLM model file if it exists, else None.
    Extractive mode is used when no model is present.
    """
    cache_dir = Path(os.environ.get("BAKUP_MODEL_CACHE_DIR", "model-weights"))
    model_file = os.environ.get("BAKUP_LLM_MODEL_FILE", "llama-3.2-3b-instruct.Q4_K_M.gguf")
    path = cache_dir / model_file
    return path if path.exists() else None


def _build_sources(results: List[RankedResult]) -> List[Source]:
    return [
        Source(
            file=r.source_file,
            line_start=r.line_start,
            line_end=r.line_end,
            excerpt=r.text[:300].strip(),
            confidence=r.confidence,
            confidence_label=r.confidence_label,
            source_type=r.source_type,
        )
        for r in results
    ]
