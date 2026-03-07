"""
core/retrieval/rag.py
─────────────────────────────────────────────────────────────────────────────
Agentic Retrieval-Augmented Generation pipeline.

Query flow (v3 — agentic multi-step reasoning):
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
    ┌──────▼────────────┐
    │  Session Memory   │──► detect follow-up → inject prior context
    └──────┬────────────┘
           │
    ┌──────▼────────────┐
    │  Planner          │──► classify question → create retrieval plan
    │  (question type   │     (steps: search_logs → extract_refs →
    │   + step plan)    │      retrieve_code → get_deps → cross_analysis)
    └──────┬────────────┘
           │
    ┌──────▼────────────┐
    │  Agent Executor   │──► run plan steps sequentially
    │  (multi-step)     │     each step feeds evidence to the next
    └──────┬────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Structured Evidence        │
    │  { logs, code, deps,        │
    │    architecture, cross_ref } │
    └──────┬──────────────────────┘
           │
    ┌──────▼───────────────────────┐
    │  LLM / Extractive answer     │
    │  (uses evidence bundle as    │
    │   structured context)        │
    └──────┬───────────────────────┘
           │
    ┌──────▼────────────┐
    │  Session Store    │──► save turn for follow-up support
    └───────────────────┘

Operating modes:
  1. Extractive (default, always available):
     Returns the most relevant chunks directly as the answer.

  2. LLM-augmented (opt-in, requires configured LLM provider):
     Passes structured evidence to the LLM with role-appropriate prompts.

  3. Clarification:
     Asks user to refine when evidence is insufficient.

  4. Agentic root-cause (new):
     Multi-step reasoning: logs → code refs → dependencies → architecture.

Neither mode will produce a fabricated answer.
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

# New intelligence modules
from core.retrieval.context_bundler import bundle_context, bundles_to_ranked_list, build_bundled_context_block
from core.ingestion.symbol_graph import query_symbol_graph
from core.analysis.architecture import get_architecture
from core.analysis.log_code_linker import link_logs_to_code, build_cross_analysis_context

# Agentic retrieval modules
from core.retrieval.planner import (
    QuestionType, RetrievalPlan, create_plan, classify_question as classify_question_type,
)
from core.retrieval.agent import (
    StructuredEvidence, execute_plan, build_evidence_context,
)
from core.retrieval.session import (
    get_session, add_turn, get_session_info,
)

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


def _is_architecture_query(question: str) -> bool:
    """Check if the question asks about project architecture or structure."""
    q_lower = question.lower()
    _arch_patterns = [
        r"\barchitecture\b", r"\bproject\s+structure\b", r"\boverview\b",
        r"\bhow\s+is.*\borganized\b", r"\bexplain\s+the\s+project\b",
        r"\bdirectory\s+structure\b", r"\bmodules?\b.*\bproject\b",
        r"\bproject\s+layout\b", r"\bentry\s+points?\b",
        r"\btell\s+me\s+about\s+(?:the\s+)?(?:project|codebase)\b",
        r"\bwhat\s+does\s+(?:the\s+)?(?:project|codebase)\s+do\b",
    ]
    return any(re.search(p, q_lower) for p in _arch_patterns)


def _is_structural_query(question: str) -> bool:
    """Check if the question can be answered from the symbol graph."""
    q_lower = question.lower()
    _struct_patterns = [
        r"which\s+files?\s+(?:use|import|depend on|reference)",
        r"what\s+depends?\s+on",
        r"(?:what|which)\s+methods?\s+(?:does|has)",
        r"(?:what|which)\s+(?:functions?|classes?|symbols?)\s+(?:are\s+)?in",
    ]
    return any(re.search(p, q_lower) for p in _struct_patterns)


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

    V3 agentic flow:
        1. Classify the question (project / greeting / off_topic).
        2. For greetings and off-topic → return canned response immediately.
        3. Check session memory for follow-up context.
        4. Plan: classify question type → create multi-step retrieval plan.
        5. Execute: agent runs plan steps sequentially, building evidence.
        6. Generate answer from structured evidence (LLM or extractive).
        7. Store turn in session memory for follow-up support.

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

    # ── Step 1: Classify (greeting / off-topic / conversational) ─────────────
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

    # ── Step 2: Session memory — check for follow-up context ─────────────────
    session = get_session(namespace)
    session_context = ""
    is_follow_up = session.is_follow_up(question)

    if is_follow_up and session.turn_count > 0:
        _step("session", f"Follow-up detected — injecting context from {session.turn_count} prior turn(s)")
        session_context = session.format_context()
        prior = session.get_prior_context_for_follow_up()
        _step("session_context", f"Prior type: {prior.get('prior_type', 'unknown')}, "
              f"prior files: {prior.get('prior_files', [])[:3]}")
    else:
        _step("session", f"New question (session has {session.turn_count} turns)")

    # ── Step 3: Plan — classify question type and create retrieval plan ───────
    _step("planner", "Creating retrieval plan...")
    question_type = classify_question_type(question)
    plan = create_plan(question, question_type)
    _step("plan_done",
          f"Type: {plan.question_type.value} | Steps: {len(plan.steps)} | "
          f"Fast-path: {plan.fast_path}",
          reasoning=plan.reasoning,
          steps=[{"type": s.step_type.value, "desc": s.description} for s in plan.steps])

    # ── Fast paths (architecture, structural — no multi-step needed) ──────────
    if plan.fast_path:
        if plan.question_type == QuestionType.ARCHITECTURE:
            _step("architecture", "Architecture query — checking cached summary")
            arch = get_architecture(namespace)
            if arch:
                _step("architecture_hit", f"Serving cached architecture ({arch.total_files} files, {len(arch.modules)} modules)")
                add_turn(namespace, question, arch.summary_text(),
                         question_type="architecture", evidence_summary="cached architecture summary")
                return _result(RAGResponse(
                    answer=arch.summary_text(),
                    confidence=1.0,
                    no_data=False,
                    mode="architecture",
                    sources=[],
                ))
            _step("architecture_miss", "No cached architecture — falling through to agentic pipeline")

        if plan.question_type == QuestionType.STRUCTURAL:
            _step("symbol_graph", "Structural query — querying symbol graph")
            graph_answer = query_symbol_graph(namespace, question)
            if graph_answer:
                _step("symbol_graph_hit", f"Symbol graph answered ({len(graph_answer)} chars)")
                add_turn(namespace, question, graph_answer,
                         question_type="structural", evidence_summary="symbol graph query")
                return _result(RAGResponse(
                    answer=graph_answer,
                    confidence=1.0,
                    no_data=False,
                    mode="symbol_graph",
                    sources=[],
                ))
            _step("symbol_graph_miss", "Symbol graph cannot answer — falling through")

    # ── Step 4: Agent — execute multi-step retrieval plan ─────────────────────
    _step("agent", f"Executing {len(plan.steps)}-step retrieval plan...")
    evidence = execute_plan(plan, question, namespace, top_k=top_k)

    # Log agent step results
    for sr in evidence.step_results:
        _step(f"agent_{sr.step_type}",
              f"{sr.description} ({sr.ms:.0f}ms, {sr.chunks_found} total chunks)",
              **sr.details)

    _step("agent_done",
          f"Evidence: {len(evidence.logs)} logs, {len(evidence.code)} code, "
          f"{len(evidence.references_found)} refs, {len(evidence.dependencies)} deps, "
          f"cross_analysis={'yes' if evidence.has_cross_analysis else 'no'} "
          f"({evidence.total_ms:.0f}ms)")

    # Log causal confidence if available
    if evidence.causal_confidence:
        cc = evidence.causal_confidence
        _step("causal_confidence",
              f"Score: {cc.get('score', 0)}/100 ({cc.get('level', 'unknown')})",
              dominant_error=cc.get("dominant_error", ""),
              factors=cc.get("factors", {}))

    if evidence.error_cluster_report:
        ecr = evidence.error_cluster_report
        _step("error_clusters",
              f"{ecr.get('cluster_count', 0)} cluster(s) detected, "
              f"dominant: {ecr.get('dominant_cluster', {}).get('error_signature', 'none') if ecr.get('dominant_cluster') else 'none'}")

    if evidence.trend_detection_report:
        tdr = evidence.trend_detection_report
        _step("trend_detection",
              f"Spikes: {tdr.get('spike_count', 0)}, "
              f"Regressions: {tdr.get('regression_count', 0)}, "
              f"New errors: {tdr.get('new_error_count', 0)}")

    # ── Step 5: Generate answer from structured evidence ──────────────────────

    # Combine all evidence chunks for threshold check
    all_chunks = evidence.logs + evidence.code
    if not all_chunks:
        if evidence.graph_answer:
            add_turn(namespace, question, evidence.graph_answer,
                     question_type=plan.question_type.value,
                     evidence_summary="symbol graph fallback")
            return _result(RAGResponse(
                answer=evidence.graph_answer,
                confidence=1.0,
                no_data=False,
                mode="symbol_graph",
                sources=[],
            ))
        _step("no_data", "No indexed content matched the query")
        return _result(RAGResponse(
            answer="No similar incident found.",
            confidence=0.0,
            no_data=True,
            mode="extractive",
            sources=[],
        ))

    top_confidence = max(c.confidence for c in all_chunks) if all_chunks else 0.0

    if not has_relevant_results(all_chunks):
        # Low confidence — try LLM analysis for logs, clarification for others
        if evidence.has_logs:
            _step("low_conf_log", "Low embedding confidence but log entries found — generating analysis")
            return _generate_agentic_answer(
                question, evidence, session_context, plan,
                namespace, _step, _result, steps, debug, t0,
            )

        _step("clarify", "Low confidence — asking LLM for clarification...")
        from core.llm.llm_service import get_llm_service
        svc = get_llm_service()
        llm_resp = svc.generate_clarification(
            question=question,
            near_miss_chunks=all_chunks[:5],
            best_confidence=top_confidence,
        )
        return _result(RAGResponse(
            answer=llm_resp.answer,
            confidence=top_confidence,
            no_data=False,
            mode="clarification",
            sources=_build_sources(all_chunks[:3]),
        ))

    # Sufficient evidence — generate answer
    return _generate_agentic_answer(
        question, evidence, session_context, plan,
        namespace, _step, _result, steps, debug, t0,
    )


def _generate_agentic_answer(
    question: str,
    evidence: StructuredEvidence,
    session_context: str,
    plan: RetrievalPlan,
    namespace: str,
    _step,
    _result,
    steps: list,
    debug: bool,
    t0: float,
) -> RAGResponse:
    """
    Generate an answer from structured evidence using the appropriate
    prompt mode (log analysis, cross-analysis, code, or general).
    """
    all_chunks = evidence.logs + evidence.code
    top_confidence = max(c.confidence for c in all_chunks) if all_chunks else 0.0

    # Build structured context for LLM
    context_block = build_evidence_context(evidence)

    # Inject session context if this is a follow-up
    if session_context:
        context_block = session_context + "\n\n" + context_block

    _step("generate", f"Generating answer from structured evidence "
          f"(type={plan.question_type.value}, "
          f"{evidence.total_chunks} chunks, "
          f"cross_analysis={'yes' if evidence.has_cross_analysis else 'no'})")

    from core.llm.llm_service import get_llm_service
    svc = get_llm_service()

    # Choose the right generation mode based on evidence type
    if plan.question_type == QuestionType.ROOT_CAUSE:
        # Root-cause mode — use cross-analysis prompt with full evidence
        _step("mode", "Root-cause reasoning mode — correlating logs + code + deps")
        llm_resp = svc.generate_agentic_answer(
            question=question,
            context_block=context_block,
            mode="root_cause",
        )

    elif evidence.has_cross_analysis:
        # Log-to-code links found — use cross-analysis prompt
        _step("mode", "Cross-analysis mode — log errors linked to source code")
        relevant_chunks = (evidence.logs + evidence.code)[:_LLM_CONTEXT_RESULTS]
        llm_resp = svc.generate_log_summary(
            question,
            relevant_chunks,
            trend_summary=evidence.trend_summary,
            cluster_summary=evidence.cluster_summary,
            confidence_summary=evidence.confidence_summary,
            file_aggregation_summary=evidence.file_aggregation_summary,
            cross_analysis_context=evidence.cross_analysis_context,
        )

    elif evidence.has_logs and plan.question_type == QuestionType.LOG_ANALYSIS:
        # Pure log analysis
        _step("mode", "Log analysis mode — structured incident report")
        relevant_chunks = evidence.logs[:_LLM_CONTEXT_RESULTS]
        llm_resp = svc.generate_log_summary(
            question,
            relevant_chunks,
            trend_summary=evidence.trend_summary,
            cluster_summary=evidence.cluster_summary,
            confidence_summary=evidence.confidence_summary,
            file_aggregation_summary=evidence.file_aggregation_summary,
        )

    else:
        # Code analysis or general — use standard RAG with bundled context
        _step("mode", "Code/general analysis mode")
        relevant_chunks = (evidence.code + evidence.logs)[:_LLM_CONTEXT_RESULTS]
        llm_resp = svc.generate_response(relevant_chunks, question)

    _step("generate_done", f"Answer generated (mode={llm_resp.mode})")

    # Debug logging
    from collections import Counter
    file_names = [getattr(r, 'file_name', '') or r.source_file for r in all_chunks[:10]]
    print(f"  [bakup:debug] classification=project({plan.question_type.value}) | "
          f"plan_steps={len(plan.steps)} | "
          f"logs={len(evidence.logs)} | code={len(evidence.code)} | "
          f"refs={len(evidence.references_found)} | deps={len(evidence.dependencies)} | "
          f"cross_analysis={'yes' if evidence.has_cross_analysis else 'no'} | "
          f"session_follow_up={'yes' if session_context else 'no'} | "
          f"files={dict(Counter(file_names))} | "
          f"llm_called={'yes' if llm_resp.mode != 'extractive' else 'no (extractive)'}")

    # Store in session memory
    source_files = list({r.source_file for r in all_chunks[:10]})
    evidence_summary = (
        f"{len(evidence.logs)} logs, {len(evidence.code)} code chunks, "
        f"{len(evidence.references_found)} refs, {len(evidence.dependencies)} deps"
    )
    add_turn(
        namespace, question, llm_resp.answer,
        source_files=source_files,
        question_type=plan.question_type.value,
        evidence_summary=evidence_summary,
    )

    return _result(RAGResponse(
        answer=llm_resp.answer,
        confidence=top_confidence,
        no_data=llm_resp.no_data,
        mode=llm_resp.mode,
        sources=_build_sources(all_chunks[:_LLM_CONTEXT_RESULTS]),
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
