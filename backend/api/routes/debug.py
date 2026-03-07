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

Endpoints:
  GET  /debug/index/{namespace}    — index diagnostics
  POST /debug/retrieval            — retrieval diagnostics for a query
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
