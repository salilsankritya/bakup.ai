"""
api/routes/debug.py
─────────────────────────────────────────────────────────────────────────────
Diagnostic endpoints for debugging the retrieval and indexing pipeline.

These endpoints help identify:
  - Whether logs are properly indexed
  - How many documents exist per namespace
  - Source-type breakdown (code vs log)
  - Sample indexed entries
  - Full retrieval diagnostics for a query
  - Agentic retrieval plan and reasoning trace
  - Session memory state
  - Error pattern clusters and causal confidence scoring
  - Time-based trend detection and spike alerts
  - Brain controller state and tool call trace

Endpoints:
  GET  /debug/index/{namespace}        — index diagnostics
  POST /debug/retrieval                — retrieval diagnostics for a query
  GET  /debug/symbols/{namespace}      — symbol graph summary
  GET  /debug/architecture/{namespace} — architecture summary
  POST /debug/plan                     — agentic retrieval plan for a query
  GET  /debug/session/{namespace}      — session memory state
  POST /debug/session/{namespace}/clear — clear session memory
  POST /debug/clusters                 — error pattern clusters for a query
  POST /debug/causal-confidence        — full causal confidence analysis
  POST /debug/trends                   — time-based trend detection
  POST /debug/router                   — hybrid query router diagnostics
  GET  /debug/brain                    — brain controller diagnostic state
  GET  /debug/brain/{namespace}        — last brain result for a namespace
"""
from __future__ import annotations

import time
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/debug", tags=["debug"])


# ── Response models ────────────────────────────────────────────────────────────

class SampleEntry(BaseModel):
    source_file: str
    source_type: str
    line_start: int
    line_end: int
    text_preview: str


class IndexDiagnostics(BaseModel):
    namespace: str
    collection_name: str
    total_chunks: int
    source_types: dict
    samples: List[SampleEntry]


class RetrievalResult(BaseModel):
    source_file: str
    line_start: int
    line_end: int
    source_type: str
    confidence: float
    confidence_label: str
    text_preview: str


class RetrievalDiagnostics(BaseModel):
    question: str
    namespace: str
    classification: str
    embedding_ms: float
    total_indexed: int
    semantic_results: int
    keyword_results: int
    merged_results: int
    top_results: List[RetrievalResult]
    has_relevant: bool
    threshold: float
    llm_configured: bool
    llm_would_be_called: bool
    pipeline_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/index/{namespace}", response_model=IndexDiagnostics)
async def debug_index(namespace: str) -> IndexDiagnostics:
    """
    Diagnostic view of what's indexed under a namespace.

    Shows:
      - Total chunks stored
      - Breakdown by source type (code vs log)
      - Sample entries with text previews
    """
    from core.retrieval.vector_store import collection_stats

    stats = collection_stats(namespace)

    samples = [
        SampleEntry(
            source_file=s.get("source_file", "unknown"),
            source_type=s.get("source_type", "unknown"),
            line_start=s.get("line_start", 0),
            line_end=s.get("line_end", 0),
            text_preview=s.get("text_preview", ""),
        )
        for s in stats.get("samples", [])
    ]

    return IndexDiagnostics(
        namespace=stats.get("namespace", namespace),
        collection_name=stats.get("collection_name", ""),
        total_chunks=stats.get("total_chunks", 0),
        source_types=stats.get("source_types", {}),
        samples=samples,
    )


@router.post("/retrieval", response_model=RetrievalDiagnostics)
async def debug_retrieval(body: dict) -> RetrievalDiagnostics:
    """
    Run the full retrieval pipeline for a query and return diagnostics
    WITHOUT generating an answer.

    Shows exactly what the RAG pipeline sees:
      - Classification result
      - Semantic search results + scores
      - Keyword search results (if log query)
      - Merged & ranked results
      - Whether threshold is met
      - Whether LLM would be called
    """
    question = body.get("question", "")
    namespace = body.get("namespace", "")
    top_k = body.get("top_k", 8)

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="Namespace required")

    from core.classifier.query_classifier import classify_query
    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import (
        query_chunks,
        collection_count,
        keyword_search,
    )
    from core.retrieval.ranker import rank_results, has_relevant_results, _get_threshold
    from core.retrieval.rag import _is_log_query, _extract_search_keywords
    from core.llm.config_store import load_config

    t0 = time.perf_counter()

    # 1. Classify
    category = classify_query(question)

    # 2. Embed
    t1 = time.perf_counter()
    query_vec = embed_query(question)
    embed_ms = (time.perf_counter() - t1) * 1000

    # 3. Semantic retrieve
    total = collection_count(namespace)
    raw = query_chunks(query_vec, namespace=namespace, top_k=top_k)

    # 4. Keyword search for log queries
    kw_chunks = []
    if _is_log_query(question):
        keywords = _extract_search_keywords(question)
        if keywords:
            kw_chunks = keyword_search(namespace, keywords, top_k=top_k)

    # 5. Merge & deduplicate
    all_chunks = raw + kw_chunks
    seen = set()
    deduped = []
    for c in all_chunks:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    # 6. Rank
    ranked = rank_results(deduped)

    # 7. Build results
    cfg = load_config()
    has_rel = has_relevant_results(ranked)

    top_results = [
        RetrievalResult(
            source_file=r.source_file,
            line_start=r.line_start,
            line_end=r.line_end,
            source_type=r.source_type,
            confidence=r.confidence,
            confidence_label=r.confidence_label,
            text_preview=r.text[:300],
        )
        for r in ranked[:10]
    ]

    pipeline_ms = (time.perf_counter() - t0) * 1000

    return RetrievalDiagnostics(
        question=question,
        namespace=namespace,
        classification=category.value,
        embedding_ms=round(embed_ms, 1),
        total_indexed=total,
        semantic_results=len(raw),
        keyword_results=len(kw_chunks),
        merged_results=len(deduped),
        top_results=top_results,
        has_relevant=has_rel,
        threshold=_get_threshold(),
        llm_configured=cfg.configured,
        llm_would_be_called=has_rel and cfg.configured,
        pipeline_ms=round(pipeline_ms, 1),
    )


# ── Symbol graph diagnostics ──────────────────────────────────────────────────

@router.get("/symbols/{namespace}")
async def debug_symbols(namespace: str) -> dict:
    """
    Return symbol graph summary for a namespace.

    Shows:
      - Node count and edge count
      - Files indexed with defined symbols
      - Import relationships
    """
    from core.ingestion.symbol_graph import get_graph

    graph = get_graph(namespace)
    if graph.node_count == 0:
        return {
            "namespace": namespace,
            "status": "empty",
            "message": "No symbol graph built. Index a project first.",
        }

    # Summarise top files by symbol count
    files_summary = {}
    for file_path, symbol_names in sorted(
        graph._file_defines.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )[:20]:
        files_summary[file_path] = {
            "symbols": len(symbol_names),
            "names": sorted(list(symbol_names))[:10],
        }

    # Top imported modules
    import_counts = {}
    for mod, importers in graph._file_imports.items():
        import_counts[mod] = len(importers)
    top_imports = dict(sorted(import_counts.items(), key=lambda kv: kv[1], reverse=True)[:15])

    return {
        "namespace": namespace,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "files_indexed": len(graph._file_defines),
        "top_files": files_summary,
        "top_imports": top_imports,
    }


# ── Architecture summary diagnostics ──────────────────────────────────────────

@router.get("/architecture/{namespace}")
async def debug_architecture(namespace: str) -> dict:
    """
    Return cached architecture summary for a namespace.

    Shows:
      - Languages, entry points, modules
      - Core dependencies, config files
      - Rendered summary text
    """
    from core.analysis.architecture import get_architecture

    arch = get_architecture(namespace)
    if arch is None:
        return {
            "namespace": namespace,
            "status": "empty",
            "message": "No architecture summary. Index a project first.",
        }

    return {
        "namespace": namespace,
        "project_name": arch.project_name,
        "languages": dict(arch.languages),
        "total_files": arch.total_files,
        "total_functions": arch.total_functions,
        "total_classes": arch.total_classes,
        "entry_points": arch.entry_points,
        "modules": [
            {
                "name": m.name,
                "path": m.path,
                "file_count": m.file_count,
                "languages": list(m.languages),
                "functions": m.functions,
                "classes": m.classes,
                "description": m.description,
            }
            for m in arch.modules
        ],
        "config_files": arch.config_files,
        "core_dependencies": arch.core_dependencies,
        "summary_text": arch.summary_text(),
    }


# ── Agentic retrieval plan diagnostics ────────────────────────────────────────

@router.post("/plan")
async def debug_plan(body: dict) -> dict:
    """
    Show the agentic retrieval plan for a question WITHOUT executing it.

    Shows:
      - Question type classification
      - Planned retrieval steps
      - Whether it's a fast-path query
      - Planner reasoning
    """
    question = body.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    from core.retrieval.planner import classify_question, create_plan

    question_type = classify_question(question)
    plan = create_plan(question, question_type)

    return {
        "question": question,
        "question_type": plan.question_type.value,
        "fast_path": plan.fast_path,
        "reasoning": plan.reasoning,
        "steps": [
            {
                "step_type": s.step_type.value,
                "description": s.description,
                "depends_on": s.depends_on,
            }
            for s in plan.steps
        ],
        "step_count": len(plan.steps),
    }


# ── Session memory diagnostics ────────────────────────────────────────────────

@router.get("/session/{namespace}")
async def debug_session(namespace: str) -> dict:
    """
    Return session memory state for a namespace.

    Shows:
      - Turn count and recent turns
      - Recent files and question types
      - Whether a follow-up would be detected for a new question
    """
    from core.retrieval.session import get_session_info

    return get_session_info(namespace)


@router.post("/session/{namespace}/clear")
async def debug_clear_session(namespace: str) -> dict:
    """Clear session memory for a namespace."""
    from core.retrieval.session import clear_session

    clear_session(namespace)
    return {
        "namespace": namespace,
        "status": "cleared",
        "message": f"Session memory for '{namespace}' has been cleared.",
    }


# ── Error pattern cluster diagnostics ─────────────────────────────────────────

@router.post("/clusters")
async def debug_clusters(body: dict) -> dict:
    """
    Run error pattern clustering on retrieved log chunks for a query.

    Shows:
      - Detected error clusters with signatures, counts, severity
      - Related files and functions per cluster
      - Sample messages and common stack frames
      - Dominant cluster identification
    """
    question = body.get("question", "")
    namespace = body.get("namespace", "")
    top_k = body.get("top_k", 15)

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="Namespace required")

    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import query_chunks, keyword_search, severity_search
    from core.retrieval.ranker import rank_results
    from core.retrieval.rag import _extract_search_keywords
    from core.analysis.error_clustering import cluster_error_patterns

    t0 = time.perf_counter()

    # Retrieve log chunks
    query_vec = embed_query(question)
    raw = query_chunks(query_vec, namespace=namespace, top_k=top_k)

    # Keyword search
    kws = _extract_search_keywords(question)
    kw_chunks = keyword_search(namespace, kws, top_k=top_k) if kws else []

    # Severity search
    sev_chunks = severity_search(namespace, severity="error", top_k=top_k)

    # Merge & rank
    all_chunks = raw + kw_chunks + sev_chunks
    seen = set()
    deduped = []
    for c in all_chunks:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    ranked = rank_results(deduped)

    # Cluster
    report = cluster_error_patterns(ranked)
    pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "question": question,
        "namespace": namespace,
        "pipeline_ms": pipeline_ms,
        "total_chunks_retrieved": len(ranked),
        **report.to_dict(),
        "summary": report.summary_text(),
    }


# ── Causal confidence scoring diagnostics ─────────────────────────────────────

@router.post("/causal-confidence")
async def debug_causal_confidence(body: dict) -> dict:
    """
    Run the full causal confidence scoring pipeline for a query.

    Shows:
      - Error clusters detected
      - Per-cluster time trends (1h, 24h, % change)
      - Causal confidence score (0–100) with factor breakdown
      - Structured reasoning input summary
      - Spike/regression/new error alerts
    """
    question = body.get("question", "")
    namespace = body.get("namespace", "")
    top_k = body.get("top_k", 15)

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="Namespace required")

    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import query_chunks, keyword_search, severity_search
    from core.retrieval.ranker import rank_results
    from core.retrieval.rag import _extract_search_keywords
    from core.analysis.error_clustering import cluster_error_patterns
    from core.analysis.trend_detector import detect_trends
    from core.analysis.causal_confidence import compute_causal_confidence
    from core.analysis.evidence_ranker import rank_evidence

    t0 = time.perf_counter()

    # Retrieve log chunks
    query_vec = embed_query(question)
    raw = query_chunks(query_vec, namespace=namespace, top_k=top_k)
    kws = _extract_search_keywords(question)
    kw_chunks = keyword_search(namespace, kws, top_k=top_k) if kws else []
    sev_chunks = severity_search(namespace, severity="error", top_k=top_k)

    all_chunks = raw + kw_chunks + sev_chunks
    seen = set()
    deduped = []
    for c in all_chunks:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    ranked = rank_results(deduped)

    # Error clustering
    ecr = cluster_error_patterns(ranked)

    # Trend detection
    tdr = detect_trends(ecr, ranked)

    # Causal confidence
    code_chunks = [r for r in ranked if r.source_type == "code"]
    ccr = compute_causal_confidence(
        cluster_report=ecr,
        trend_report=tdr,
        code_chunk_count=len(code_chunks),
        reference_count=0,
        dependency_count=0,
        cross_analysis_available=False,
    )

    # Evidence ranking
    sri = rank_evidence(
        cluster_report=ecr,
        trend_report=tdr,
        confidence_result=ccr,
        code_chunks=code_chunks,
    )

    pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "question": question,
        "namespace": namespace,
        "pipeline_ms": pipeline_ms,
        "total_chunks_retrieved": len(ranked),
        "error_clusters": ecr.to_dict(),
        "trend_detection": tdr.to_dict(),
        "causal_confidence": ccr.to_dict(),
        "structured_reasoning": sri.to_dict(),
        "structured_reasoning_prompt": sri.to_prompt_block(),
    }


# ── Time-based trend detection diagnostics ────────────────────────────────────

@router.post("/trends")
async def debug_trends(body: dict) -> dict:
    """
    Run time-based trend detection on retrieved log chunks.

    Shows:
      - Per-cluster occurrence counts (1h, 24h windows)
      - Percentage change vs previous window
      - Spike, regression, and new error detection
      - Existing trend analysis from the trends module
    """
    question = body.get("question", "")
    namespace = body.get("namespace", "")
    top_k = body.get("top_k", 15)

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")
    if not namespace.strip():
        raise HTTPException(status_code=400, detail="Namespace required")

    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import query_chunks, keyword_search, severity_search
    from core.retrieval.ranker import rank_results
    from core.retrieval.rag import _extract_search_keywords
    from core.analysis.error_clustering import cluster_error_patterns
    from core.analysis.trend_detector import detect_trends
    from core.analysis.trends import analyze_error_trends

    t0 = time.perf_counter()

    query_vec = embed_query(question)
    raw = query_chunks(query_vec, namespace=namespace, top_k=top_k)
    kws = _extract_search_keywords(question)
    kw_chunks = keyword_search(namespace, kws, top_k=top_k) if kws else []
    sev_chunks = severity_search(namespace, severity="error", top_k=top_k)

    all_chunks = raw + kw_chunks + sev_chunks
    seen = set()
    deduped = []
    for c in all_chunks:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    ranked = rank_results(deduped)

    # Existing trend analysis
    legacy_trends = analyze_error_trends(ranked)

    # Error clustering + new trend detection
    ecr = cluster_error_patterns(ranked)
    tdr = detect_trends(ecr, ranked)

    pipeline_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "question": question,
        "namespace": namespace,
        "pipeline_ms": pipeline_ms,
        "total_chunks_retrieved": len(ranked),
        "legacy_trends": legacy_trends.to_dict(),
        "error_clusters": ecr.to_dict(),
        "cluster_trends": tdr.to_dict(),
        "summary": tdr.summary_text(),
        "alerts": {
            "spikes": tdr.spike_count,
            "regressions": tdr.regression_count,
            "new_errors": tdr.new_error_count,
        },
    }


# ── Hybrid query router diagnostics ──────────────────────────────────────────

@router.post("/router")
async def debug_router(body: dict) -> dict:
    """
    Run the hybrid query router for a query and return full diagnostics.

    Shows:
      - Routing intent (project_query / conversational / unrelated)
      - Confidence score
      - Classification source (llm or rules)
      - Latency
      - Routing decision (what pipeline would be executed)
    """
    question = body.get("question", "")
    namespace = body.get("namespace", "")

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question required")

    from core.router.router import route_query, CONFIDENCE_THRESHOLD

    routing = route_query(
        query=question,
        namespace=namespace,
    )

    # Determine what the pipeline would do
    if routing.intent == "project_query":
        routing_decision = "planner + agent retrieval pipeline"
    elif routing.intent == "conversational":
        routing_decision = "conversational response (LLM direct or canned)"
    else:
        routing_decision = "polite scope message (unrelated to project)"

    # Also run the legacy classifier for comparison
    from core.classifier.query_classifier import classify_query
    legacy_category = classify_query(question)

    return {
        "query": question,
        "namespace": namespace,
        "intent": routing.intent,
        "confidence": round(routing.confidence, 3),
        "source": routing.source,
        "latency_ms": routing.latency_ms,
        "routing_decision": routing_decision,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "above_threshold": routing.confidence >= CONFIDENCE_THRESHOLD,
        "legacy_classifier": legacy_category.value,
    }


# ── Brain controller diagnostics ─────────────────────────────────────────────

@router.get("/brain")
async def debug_brain() -> dict:
    """
    Return the brain controller's current state.

    Shows:
      - LLM provider and model
      - Whether tool calling is supported
      - Available tools and their schemas
      - Brain mode (active or fallback)
      - Configuration (max_tool_calls, app_mode)
    """
    from core.llm.config_store import load_config, get_config_public
    from core.brain.tools import TOOLS
    from core.brain.brain import _provider_supports_tools, MAX_TOOL_CALLS

    cfg = load_config()
    public_cfg = get_config_public()

    tools_supported = _provider_supports_tools(cfg.provider) if cfg.configured else False

    import os
    app_mode = os.environ.get("BAKUP_APP_MODE", "local")

    return {
        "llm_configured": cfg.configured,
        "provider": public_cfg.get("provider", "none"),
        "model": public_cfg.get("model", "none"),
        "api_key_set": public_cfg.get("api_key_set", False),
        "tools_supported": tools_supported,
        "brain_mode": "active" if (cfg.configured and tools_supported) else "fallback",
        "max_tool_calls": MAX_TOOL_CALLS,
        "app_mode": app_mode,
        "available_tools": [
            {
                "name": t.name,
                "description": t.description,
                "params": [
                    {"name": p.name, "type": p.type, "required": p.required}
                    for p in t.params
                ],
            }
            for t in TOOLS
        ],
    }


@router.get("/brain/{namespace}")
async def debug_brain_namespace(namespace: str) -> dict:
    """
    Return the most recent brain result for a namespace.

    Shows:
      - Answer mode (brain, fallback, etc.)
      - Tool calls made (names, arguments, timing)
      - Reasoning trace (step-by-step)
      - Provider and model used
      - Total processing time
    """
    from core.brain.brain import get_debug_result

    result = get_debug_result(namespace)
    if not result:
        return {
            "namespace": namespace,
            "has_result": False,
            "message": "No brain result cached for this namespace. Run a query first.",
        }

    return {
        "namespace": namespace,
        "has_result": True,
        "mode": result.mode,
        "provider": result.provider,
        "model": result.model,
        "confidence": result.confidence,
        "no_data": result.no_data,
        "answer_length": len(result.answer),
        "answer_preview": result.answer[:500],
        "tool_calls": result.tool_calls,
        "reasoning_trace": result.reasoning_trace,
        "sources": result.sources[:5],
        "total_ms": result.total_ms,
        "error": result.error,
    }
