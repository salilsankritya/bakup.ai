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
