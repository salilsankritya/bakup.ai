"""
core/retrieval/agent.py
─────────────────────────────────────────────────────────────────────────────
Agentic retrieval executor — runs multi-step retrieval plans.

Takes a RetrievalPlan from the planner and executes each step sequentially,
feeding evidence from earlier steps into later ones. Produces a structured
evidence bundle and reasoning trace for the LLM.

Architecture:
    Planner  →  create_plan(question)  →  RetrievalPlan
                          │
    Agent    →  execute_plan(plan, question, namespace)
                          │
                 ┌────────▼────────┐
                 │  Step 1: logs   │──► evidence.logs
                 │  Step 2: refs   │──► evidence.references
                 │  Step 3: code   │──► evidence.code
                 │  Step 4: deps   │──► evidence.dependencies
                 │  Step 5: arch   │──► evidence.architecture
                 │  Step 6: cross  │──► evidence.cross_analysis
                 └────────┬────────┘
                          │
                 StructuredEvidence (sent to LLM)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.retrieval.planner import (
    PlanStep,
    QuestionType,
    RetrievalPlan,
    StepType,
)
from core.retrieval.ranker import RankedResult, rank_results, has_relevant_results, top_relevant

# Type hint only — actual imports deferred to avoid circular deps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.analysis.error_clustering import ErrorClusterReport
    from core.analysis.trend_detector import TrendDetectionReport
    from core.analysis.causal_confidence import CausalConfidenceResult
    from core.analysis.evidence_ranker import StructuredReasoningInput

logger = logging.getLogger("bakup.agent")


# ── Structured evidence ───────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Output of a single retrieval step."""
    step_type: str
    description: str
    chunks_found: int
    ms: float
    details: Dict = field(default_factory=dict)


@dataclass
class StructuredEvidence:
    """
    All evidence gathered by the agent, organised by category.

    This is the bundle sent to the LLM for reasoning.
    """
    logs: List[RankedResult] = field(default_factory=list)
    code: List[RankedResult] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    architecture_summary: str = ""
    cross_analysis_context: str = ""
    graph_answer: str = ""
    references_found: List[dict] = field(default_factory=list)

    # Analysis results (from log pipeline)
    trend_summary: str = ""
    cluster_summary: str = ""
    confidence_summary: str = ""
    file_aggregation_summary: str = ""

    # Causal analysis results (v4 — error clustering + trend detection)
    error_cluster_report: Optional[dict] = None        # ErrorClusterReport.to_dict()
    trend_detection_report: Optional[dict] = None       # TrendDetectionReport.to_dict()
    causal_confidence: Optional[dict] = None             # CausalConfidenceResult.to_dict()
    structured_reasoning: Optional[str] = None           # StructuredReasoningInput.to_prompt_block()
    structured_reasoning_dict: Optional[dict] = None     # StructuredReasoningInput.to_dict()

    # Reasoning trace
    step_results: List[StepResult] = field(default_factory=list)
    total_ms: float = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.logs) + len(self.code)

    @property
    def has_logs(self) -> bool:
        return len(self.logs) > 0

    @property
    def has_code(self) -> bool:
        return len(self.code) > 0

    @property
    def has_cross_analysis(self) -> bool:
        return bool(self.cross_analysis_context)


# ── Keyword helpers (from rag.py, centralised here) ───────────────────────────

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


def _extract_search_keywords(question: str) -> List[str]:
    """Map query-level log words to content-level keywords."""
    import re
    q = question.lower()
    found: List[str] = []
    for query_kw, content_kws in _KEYWORD_MAP.items():
        if query_kw in q:
            found.extend(content_kws)
    if re.search(r"\blog(s|file)?\b", q) and not found:
        found.extend(["ERROR", "Exception", "WARN", "Traceback", "FATAL"])
    return list(dict.fromkeys(found))


# ── Step executors ─────────────────────────────────────────────────────────────

def _execute_search_logs(
    question: str,
    namespace: str,
    top_k: int,
    evidence: StructuredEvidence,
) -> None:
    """Search for log entries: semantic + keyword + severity."""
    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import (
        query_chunks, keyword_search, severity_search, collection_count,
    )
    from core.analysis.confidence import calculate_confidence
    from core.analysis.trends import analyze_error_trends
    from core.analysis.clusters import cluster_log_events
    from core.analysis.file_aggregation import aggregate_by_file

    # Semantic search
    query_vec = embed_query(question)
    count = collection_count(namespace)
    raw_chunks = query_chunks(query_vec, namespace=namespace, top_k=top_k)

    # Keyword search
    search_kws = _extract_search_keywords(question)
    kw_chunks = []
    if search_kws:
        kw_chunks = keyword_search(namespace, search_kws, top_k=top_k)

    # Severity search
    sev_chunks = severity_search(namespace, severity="error", top_k=top_k * 2)

    # Merge & dedup
    all_raw = raw_chunks + kw_chunks + sev_chunks
    seen: Set[Tuple] = set()
    deduped = []
    for c in all_raw:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    # Rank
    ranked = rank_results(deduped)

    # Separate log vs code chunks
    for r in ranked:
        if r.source_type == "log":
            evidence.logs.append(r)
        else:
            evidence.code.append(r)

    # Run analysis pipeline on log chunks
    analysis_chunks = evidence.logs if evidence.logs else ranked[:top_k]
    if analysis_chunks:
        try:
            trend_report = analyze_error_trends(analysis_chunks)
            evidence.trend_summary = trend_report.summary_text()
        except Exception:
            pass

        try:
            cluster_report = cluster_log_events(analysis_chunks)
            evidence.cluster_summary = cluster_report.summary_text()
        except Exception:
            pass

        try:
            conf_result = calculate_confidence(analysis_chunks, question=question)
            evidence.confidence_summary = conf_result.reasoning
        except Exception:
            pass

        try:
            agg_report = aggregate_by_file(analysis_chunks)
            evidence.file_aggregation_summary = agg_report.summary_text
        except Exception:
            pass

        # ── V4: Error pattern clustering + trend detection + causal confidence ──
        try:
            from core.analysis.error_clustering import cluster_error_patterns
            ecr = cluster_error_patterns(analysis_chunks)
            evidence.error_cluster_report = ecr.to_dict()

            # Cluster summary for backward compat — prefer the richer report
            if ecr.clusters:
                evidence.cluster_summary = ecr.summary_text()
        except Exception:
            ecr = None

        try:
            from core.analysis.trend_detector import detect_trends
            if ecr and ecr.clusters:
                tdr = detect_trends(ecr, analysis_chunks)
                evidence.trend_detection_report = tdr.to_dict()
                # Enrich trend summary with per-cluster trends
                if tdr.has_alerts:
                    evidence.trend_summary = (evidence.trend_summary or "") + "\n\n" + tdr.summary_text()
            else:
                tdr = None
        except Exception:
            tdr = None

        try:
            from core.analysis.causal_confidence import compute_causal_confidence
            if ecr:
                ccr = compute_causal_confidence(
                    cluster_report=ecr,
                    trend_report=tdr,
                    code_chunk_count=len(evidence.code),
                    reference_count=len(evidence.references_found),
                    dependency_count=len(evidence.dependencies),
                    cross_analysis_available=bool(evidence.cross_analysis_context),
                )
                evidence.causal_confidence = ccr.to_dict()
                evidence.confidence_summary = ccr.explanation
        except Exception:
            ccr = None


def _execute_search_code(
    question: str,
    namespace: str,
    top_k: int,
    evidence: StructuredEvidence,
) -> None:
    """Semantic search specifically for code chunks."""
    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import query_chunks

    query_vec = embed_query(question)
    raw_chunks = query_chunks(query_vec, namespace=namespace, top_k=top_k)
    ranked = rank_results(raw_chunks)

    for r in ranked:
        # Avoid duplicating chunks already found in log search
        key = (r.source_file, r.line_start, r.line_end)
        existing = {(c.source_file, c.line_start, c.line_end) for c in evidence.code}
        if key not in existing:
            evidence.code.append(r)


def _execute_extract_refs(evidence: StructuredEvidence) -> None:
    """Extract code references from log text."""
    from core.analysis.log_code_linker import extract_code_references

    for log_chunk in evidence.logs:
        refs = extract_code_references(log_chunk.text)
        for ref in refs:
            evidence.references_found.append({
                "file_path": ref.file_path,
                "line_number": ref.line_number,
                "function_name": ref.function_name,
                "class_name": ref.class_name,
                "confidence": ref.confidence,
                "from_log": log_chunk.source_file,
            })


def _execute_retrieve_code(
    namespace: str,
    top_k: int,
    evidence: StructuredEvidence,
) -> None:
    """
    Fetch code chunks for identifiers extracted from logs.
    Uses keyword search to find chunks by function/class/file name.
    """
    from core.retrieval.vector_store import keyword_search

    if not evidence.references_found:
        return

    # Collect unique identifiers to search for
    search_terms: List[str] = []
    seen: Set[str] = set()
    for ref in evidence.references_found:
        for term in [ref.get("function_name", ""), ref.get("class_name", "")]:
            if term and term not in seen:
                seen.add(term)
                search_terms.append(term)

    if not search_terms:
        return

    kw_chunks = keyword_search(namespace, search_terms[:10], top_k=top_k)
    ranked = rank_results(kw_chunks)

    existing = {(c.source_file, c.line_start, c.line_end) for c in evidence.code}
    for r in ranked:
        key = (r.source_file, r.line_start, r.line_end)
        if key not in existing:
            existing.add(key)
            evidence.code.append(r)


def _execute_get_deps(namespace: str, evidence: StructuredEvidence) -> None:
    """Look up dependencies of evidence code chunks via symbol graph."""
    from core.ingestion.symbol_graph import get_graph

    graph = get_graph(namespace)
    if graph.node_count == 0:
        return

    seen: Set[str] = set()
    for chunk in evidence.code[:10]:  # Top 10 code chunks
        # Check what files import the chunk's file
        if chunk.source_file:
            dependents = graph.dependents_of(chunk.source_file)
            for dep in dependents:
                if dep not in seen:
                    seen.add(dep)
                    evidence.dependencies.append(dep)

        # Check what the chunk's function depends on
        if chunk.function_name:
            importers = graph.files_that_import(chunk.function_name)
            for imp in importers:
                if imp not in seen:
                    seen.add(imp)
                    evidence.dependencies.append(imp)


def _execute_get_arch(namespace: str, evidence: StructuredEvidence) -> None:
    """Fetch architecture summary."""
    from core.analysis.architecture import get_architecture

    arch = get_architecture(namespace)
    if arch:
        evidence.architecture_summary = arch.summary_text()


def _execute_query_graph(
    question: str,
    namespace: str,
    evidence: StructuredEvidence,
) -> None:
    """Answer structural questions from symbol graph."""
    from core.ingestion.symbol_graph import query_symbol_graph

    answer = query_symbol_graph(namespace, question)
    if answer:
        evidence.graph_answer = answer


def _execute_cross_analysis(evidence: StructuredEvidence) -> None:
    """Link log errors to code via the log-code linker, then rank evidence."""
    from core.analysis.log_code_linker import link_logs_to_code, build_cross_analysis_context

    if not evidence.logs or not evidence.code:
        return

    links = link_logs_to_code(evidence.logs, evidence.code)
    linked_count = sum(1 for l in links if l.code_chunks)
    if links and linked_count > 0:
        evidence.cross_analysis_context = build_cross_analysis_context(links)

    # ── V4: Build structured reasoning input after cross-analysis ────────────
    # This requires that error_cluster_report and causal_confidence were set
    # during search_logs. If not, re-run them now with updated evidence.
    try:
        from core.analysis.error_clustering import cluster_error_patterns
        from core.analysis.trend_detector import detect_trends
        from core.analysis.causal_confidence import compute_causal_confidence
        from core.analysis.evidence_ranker import rank_evidence

        # If clustering wasn't done yet (e.g., non-log entry path), do it now
        if evidence.error_cluster_report is None and evidence.logs:
            ecr = cluster_error_patterns(evidence.logs)
            evidence.error_cluster_report = ecr.to_dict()
        else:
            ecr = None
            if evidence.error_cluster_report:
                # Reconstruct from dict for the ranker
                from core.analysis.error_clustering import ErrorClusterReport as ECR, ErrorCluster as EC
                ecr = ECR(
                    total_entries=evidence.error_cluster_report.get("total_entries", 0),
                    unclustered_count=evidence.error_cluster_report.get("unclustered_count", 0),
                )
                for cd in evidence.error_cluster_report.get("clusters", []):
                    ecr.clusters.append(EC(
                        cluster_id=cd.get("cluster_id", 0),
                        error_signature=cd.get("error_signature", ""),
                        exception_type=cd.get("exception_type", ""),
                        count=cd.get("count", 0),
                        first_seen=cd.get("first_seen"),
                        last_seen=cd.get("last_seen"),
                        related_files=cd.get("related_files", []),
                        related_functions=cd.get("related_functions", []),
                        severity=cd.get("severity", "unknown"),
                        sample_messages=cd.get("sample_messages", []),
                        stack_frames=cd.get("stack_frames", []),
                        occurrences_1h=cd.get("occurrences_1h", 0),
                        occurrences_24h=cd.get("occurrences_24h", 0),
                        trend_pct_change=cd.get("trend_pct_change", 0),
                        trend_label=cd.get("trend_label", ""),
                    ))

        if ecr and ecr.clusters:
            # Trend detection (may already exist)
            tdr = None
            if evidence.trend_detection_report:
                from core.analysis.trend_detector import TrendDetectionReport as TDR, ClusterTrend
                tdr = TDR(
                    spike_count=evidence.trend_detection_report.get("spike_count", 0),
                    regression_count=evidence.trend_detection_report.get("regression_count", 0),
                    new_error_count=evidence.trend_detection_report.get("new_error_count", 0),
                    analysis_time_utc=evidence.trend_detection_report.get("analysis_time_utc", ""),
                )
                for td in evidence.trend_detection_report.get("cluster_trends", []):
                    tdr.cluster_trends.append(ClusterTrend(
                        cluster_id=td.get("cluster_id", 0),
                        total=td.get("total", 0),
                        last_1h=td.get("last_1h", 0),
                        last_24h=td.get("last_24h", 0),
                        pct_change=td.get("pct_change", 0),
                        trend_label=td.get("trend_label", "stable"),
                        spike_detected=td.get("spike_detected", False),
                        is_new=td.get("is_new", False),
                        is_regression=td.get("is_regression", False),
                    ))

            # Recompute causal confidence with updated cross-analysis status
            ccr = compute_causal_confidence(
                cluster_report=ecr,
                trend_report=tdr,
                code_chunk_count=len(evidence.code),
                reference_count=len(evidence.references_found),
                dependency_count=len(evidence.dependencies),
                cross_analysis_available=bool(evidence.cross_analysis_context),
            )
            evidence.causal_confidence = ccr.to_dict()
            evidence.confidence_summary = ccr.explanation

            # Rank evidence
            sri = rank_evidence(
                cluster_report=ecr,
                trend_report=tdr,
                confidence_result=ccr,
                code_chunks=evidence.code,
                dependencies=evidence.dependencies,
            )
            evidence.structured_reasoning = sri.to_prompt_block()
            evidence.structured_reasoning_dict = sri.to_dict()

    except Exception as exc:
        logger.warning("V4 evidence ranking failed: %s", exc)


def _execute_bundle_context(evidence: StructuredEvidence) -> None:
    """Bundle code chunks with siblings and imports."""
    from core.retrieval.context_bundler import bundle_context, bundles_to_ranked_list

    if not evidence.code:
        return

    bundles = bundle_context(evidence.code, top_n=5, max_siblings=2)
    if bundles:
        bundled = bundles_to_ranked_list(bundles)
        evidence.code = bundled  # Replace with richer context


# ── Main executor ──────────────────────────────────────────────────────────────

def execute_plan(
    plan: RetrievalPlan,
    question: str,
    namespace: str,
    top_k: int = 8,
) -> StructuredEvidence:
    """
    Execute a retrieval plan step-by-step, building structured evidence.

    Each step type maps to an executor function. Steps run sequentially
    so later steps can use evidence from earlier ones.

    Returns:
        StructuredEvidence with all gathered chunks, analysis, and trace.
    """
    evidence = StructuredEvidence()
    t0 = time.perf_counter()

    logger.info(
        "Agent executing plan: %s (%d steps)",
        plan.question_type.value, len(plan.steps),
    )

    for step in plan.steps:
        t_step = time.perf_counter()

        try:
            if step.step_type == StepType.SEARCH_LOGS:
                _execute_search_logs(question, namespace, top_k, evidence)

            elif step.step_type == StepType.SEARCH_CODE:
                _execute_search_code(question, namespace, top_k, evidence)

            elif step.step_type == StepType.EXTRACT_REFS:
                _execute_extract_refs(evidence)

            elif step.step_type == StepType.RETRIEVE_CODE:
                _execute_retrieve_code(namespace, top_k, evidence)

            elif step.step_type == StepType.GET_DEPS:
                _execute_get_deps(namespace, evidence)

            elif step.step_type == StepType.GET_ARCH:
                _execute_get_arch(namespace, evidence)

            elif step.step_type == StepType.QUERY_GRAPH:
                _execute_query_graph(question, namespace, evidence)

            elif step.step_type == StepType.CROSS_ANALYSIS:
                _execute_cross_analysis(evidence)

            elif step.step_type == StepType.BUNDLE_CONTEXT:
                _execute_bundle_context(evidence)

            step_ms = round((time.perf_counter() - t_step) * 1000, 1)
            evidence.step_results.append(StepResult(
                step_type=step.step_type.value,
                description=step.description,
                chunks_found=evidence.total_chunks,
                ms=step_ms,
                details={
                    "logs": len(evidence.logs),
                    "code": len(evidence.code),
                    "refs": len(evidence.references_found),
                    "deps": len(evidence.dependencies),
                    "has_arch": bool(evidence.architecture_summary),
                    "has_cross": bool(evidence.cross_analysis_context),
                },
            ))

            logger.debug(
                "  Step %s: %d chunks, %.1fms",
                step.step_type.value, evidence.total_chunks, step_ms,
            )

        except Exception as exc:
            step_ms = round((time.perf_counter() - t_step) * 1000, 1)
            logger.warning("Step %s failed: %s", step.step_type.value, exc)
            evidence.step_results.append(StepResult(
                step_type=step.step_type.value,
                description=step.description,
                chunks_found=evidence.total_chunks,
                ms=step_ms,
                details={"error": str(exc)},
            ))

    evidence.total_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "Agent complete: %d logs, %d code, %d refs, %d deps, %.1fms",
        len(evidence.logs), len(evidence.code),
        len(evidence.references_found), len(evidence.dependencies),
        evidence.total_ms,
    )

    return evidence


# ── Context rendering ─────────────────────────────────────────────────────────

def build_evidence_context(
    evidence: StructuredEvidence,
    max_chars: int = 800,
) -> str:
    """
    Render structured evidence into a context block for the LLM prompt.

    Organises evidence by category so the LLM sees a coherent picture
    rather than randomly interleaved log and code fragments.
    """
    sections: List[str] = []

    # Log evidence
    if evidence.logs:
        log_parts = []
        for i, chunk in enumerate(evidence.logs[:8], 1):
            text = chunk.text[:max_chars]
            if len(chunk.text) > max_chars:
                text += "\n[...truncated]"
            log_parts.append(
                f"[L{i}] {chunk.source_file}  lines {chunk.line_start}–{chunk.line_end}"
                f"  confidence: {chunk.confidence:.2f}\n{text}"
            )
        sections.append("## Log Evidence\n\n" + "\n\n---\n\n".join(log_parts))

    # Code evidence
    if evidence.code:
        code_parts = []
        for i, chunk in enumerate(evidence.code[:8], 1):
            text = chunk.text[:max_chars]
            if len(chunk.text) > max_chars:
                text += "\n[...truncated]"
            label = f"[C{i}] {chunk.source_file}  lines {chunk.line_start}–{chunk.line_end}"
            label += f"  confidence: {chunk.confidence:.2f}"
            if chunk.function_name:
                label += f"  function: {chunk.function_name}"
            if chunk.class_name:
                label += f"  class: {chunk.class_name}"
            code_parts.append(label + "\n" + text)
        sections.append("## Code Evidence\n\n" + "\n\n---\n\n".join(code_parts))

    # Dependencies
    if evidence.dependencies:
        dep_list = "\n".join(f"- {d}" for d in evidence.dependencies[:15])
        sections.append(f"## Dependencies\n\n{dep_list}")

    # Architecture summary
    if evidence.architecture_summary:
        # Truncate to fit context window
        arch = evidence.architecture_summary[:1500]
        if len(evidence.architecture_summary) > 1500:
            arch += "\n[...truncated]"
        sections.append(f"## Architecture Context\n\n{arch}")

    # Cross-analysis
    if evidence.cross_analysis_context:
        sections.append(f"## Log-to-Code Cross Analysis\n\n{evidence.cross_analysis_context}")

    # Analysis summaries
    analysis_parts = []
    if evidence.trend_summary:
        analysis_parts.append(f"### Error Trends\n{evidence.trend_summary}")
    if evidence.cluster_summary:
        analysis_parts.append(f"### Incident Clusters\n{evidence.cluster_summary}")
    if evidence.file_aggregation_summary:
        analysis_parts.append(f"### Error Distribution by File\n{evidence.file_aggregation_summary}")
    if evidence.confidence_summary:
        analysis_parts.append(f"### Confidence Assessment\n{evidence.confidence_summary}")
    if analysis_parts:
        sections.append("## Automated Analysis\n\n" + "\n\n".join(analysis_parts))

    # References extracted from logs
    if evidence.references_found:
        ref_lines = []
        for ref in evidence.references_found[:10]:
            parts = []
            if ref.get("file_path"):
                parts.append(f"file: {ref['file_path']}")
            if ref.get("function_name"):
                parts.append(f"function: {ref['function_name']}")
            if ref.get("class_name"):
                parts.append(f"class: {ref['class_name']}")
            if ref.get("line_number"):
                parts.append(f"line: {ref['line_number']}")
            ref_lines.append("- " + ", ".join(parts))
        sections.append("## Extracted Code References\n\n" + "\n".join(ref_lines))

    # Structured reasoning input (v4 — causal confidence + cluster ranking)
    if evidence.structured_reasoning:
        sections.append("## Structured Root-Cause Analysis\n\n" + evidence.structured_reasoning)

    return "\n\n" + "═" * 60 + "\n\n".join(sections)
