"""
api/routes/query.py
─────────────────────────────────────────────────────────────────────────────
POST /ask        — answer a question about an indexed project.
POST /ask/stream — same, but streams step-by-step status events via SSE.

Query classification runs BEFORE retrieval:
  - greeting  → immediate polite reply (no DB query)
  - off_topic → scope guard (no DB query)
  - project   → full RAG pipeline (embed → retrieve → rank → answer)

Every response includes:
  - answer          Plain-text answer, clarification, or scope message
  - confidence      Float [0, 1] — score of the best-matching source
  - no_data         True when no relevant content was found
  - mode            "extractive" | "llm" | "greeting" | "conversational" | "off_topic" | "clarification"
  - sources         List of cited chunks with file, line, and confidence
  - debug_trace     (optional) Step-by-step pipeline trace when debug=true
"""
from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    namespace: str          # Must match the namespace used during /index
    top_k: int = 8          # Number of candidates to retrieve (1–20)
    debug: bool = False     # When true, include step-by-step pipeline trace


class SourceModel(BaseModel):
    file: str
    line_start: int
    line_end: int
    excerpt: str
    confidence: float
    confidence_label: str   # "high" | "medium" | "low"
    source_type: str        # "code" | "log"


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    no_data: bool
    mode: str               # "extractive" | "llm" | "greeting" | "conversational" | "off_topic" | "clarification"
    sources: List[SourceModel]
    debug_trace: Optional[List[dict]] = None   # Step-by-step trace when debug=true


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_query_response(result) -> QueryResponse:
    """Convert a RAGResponse into the API response model."""
    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        no_data=result.no_data,
        mode=result.mode,
        sources=[
            SourceModel(
                file=s.file,
                line_start=s.line_start,
                line_end=s.line_end,
                excerpt=s.excerpt,
                confidence=s.confidence,
                confidence_label=s.confidence_label,
                source_type=s.source_type,
            )
            for s in result.sources
        ],
        debug_trace=result.debug_trace,
    )


def _response_to_dict(result) -> dict:
    """Serialise a RAGResponse into a plain dict for SSE streaming."""
    return {
        "answer":      result.answer,
        "confidence":  result.confidence,
        "no_data":     result.no_data,
        "mode":        result.mode,
        "sources": [
            {
                "file":             s.file,
                "line_start":       s.line_start,
                "line_end":         s.line_end,
                "excerpt":          s.excerpt,
                "confidence":       s.confidence,
                "confidence_label": s.confidence_label,
                "source_type":      s.source_type,
            }
            for s in result.sources
        ],
        "debug_trace": result.debug_trace,
    }


# ── Route: POST /ask ─────────────────────────────────────────────────────────

@router.post("/ask", response_model=QueryResponse, tags=["query"])
async def ask(body: QueryRequest) -> QueryResponse:
    """
    Classify the question, then embed → retrieve → rank → answer.

    Classification happens first:
      - greeting  → immediate polite reply (no DB access)
      - off_topic → scope guard reply (no DB access)
      - project   → full RAG pipeline

    Set debug=true to include a step-by-step pipeline trace in the response.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # ── Classification shortcut ───────────────────────────────────────────────
    from core.classifier.query_classifier import classify_query, QueryCategory

    category = classify_query(body.question)
    if category in (QueryCategory.GREETING, QueryCategory.CONVERSATIONAL, QueryCategory.OFF_TOPIC):
        from core.retrieval.rag import answer_question as _answer
        result = _answer(
            question=body.question,
            namespace=body.namespace or "_",
            top_k=1,
            debug=body.debug,
        )
        return _build_query_response(result)

    # ── Project questions require a valid namespace ───────────────────────────
    if not body.namespace.strip():
        raise HTTPException(status_code=400, detail="Namespace must not be empty. Use the namespace returned by /index.")

    top_k = max(1, min(body.top_k, 20))

    from core.retrieval.rag import answer_question
    from core.retrieval.vector_store import collection_count

    count = collection_count(body.namespace)
    if count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Namespace '{body.namespace}' has no indexed content. "
                "Run POST /index first."
            ),
        )

    try:
        result = answer_question(
            question=body.question,
            namespace=body.namespace,
            top_k=top_k,
            debug=body.debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return _build_query_response(result)


# ── Route: POST /ask/stream (SSE) ────────────────────────────────────────────

@router.post("/ask/stream", tags=["query"])
async def ask_stream(body: QueryRequest):
    """
    Same as POST /ask but streams real-time status events via Server-Sent Events.

    Each event is a JSON line:
      data: {"type":"step", "step":"classify", "message":"Classifying..."}
      data: {"type":"step", "step":"retrieve", "message":"Searching 847 docs..."}
      data: {"type":"result", "data": {<full QueryResponse>}}

    The UI can consume this to show live progress:
      "Classifying your question..."  →  "Searching 847 documents..."  →  "Generating answer..."
    """
    async def event_generator():
        loop = asyncio.get_event_loop()

        def sse_step(step: str, message: str):
            payload = json.dumps({"type": "step", "step": step, "message": message})
            return f"data: {payload}\n\n"

        def sse_result(data: dict):
            payload = json.dumps({"type": "result", "data": data})
            return f"data: {payload}\n\n"

        try:
            # ── Step 1: Classify ──────────────────────────────────────────────
            yield sse_step("classify", "Classifying your question...")

            from core.classifier.query_classifier import classify_query, QueryCategory
            category = await loop.run_in_executor(None, classify_query, body.question)

            if category == QueryCategory.GREETING:
                yield sse_step("classify_done", "Greeting detected")
                from core.retrieval.rag import answer_question
                result = await loop.run_in_executor(
                    None, partial(answer_question, body.question, "_", 1, True)
                )
                yield sse_result(_response_to_dict(result))
                return

            if category == QueryCategory.CONVERSATIONAL:
                yield sse_step("classify_done", "Conversational question — no retrieval needed")
                yield sse_step("llm_direct", "Responding directly...")
                from core.llm.llm_service import get_llm_service
                svc = get_llm_service()
                llm_resp = await loop.run_in_executor(
                    None, partial(svc.generate_conversational, body.question)
                )
                yield sse_step("llm_done", f"Response ready ({llm_resp.mode})")
                from core.retrieval.rag import RAGResponse
                result = RAGResponse(
                    answer=llm_resp.answer,
                    confidence=1.0,
                    no_data=False,
                    mode="conversational",
                    sources=[],
                )
                yield sse_step("complete", "Done!")
                yield sse_result(_response_to_dict(result))
                return

            if category == QueryCategory.OFF_TOPIC:
                yield sse_step("classify_done", "Outside project scope")
                from core.retrieval.rag import answer_question
                result = await loop.run_in_executor(
                    None, partial(answer_question, body.question, "_", 1, True)
                )
                yield sse_result(_response_to_dict(result))
                return

            yield sse_step("classify_done", "Project question detected")

            # ── Step 2: Validate namespace ────────────────────────────────────
            if not body.namespace or not body.namespace.strip():
                yield sse_result({"error": "Namespace required"})
                return

            from core.retrieval.vector_store import collection_count
            count = await loop.run_in_executor(None, collection_count, body.namespace)

            if count == 0:
                yield sse_result({
                    "error": f"Namespace '{body.namespace}' has no indexed content."
                })
                return

            yield sse_step("index_check", f"Found {count} indexed documents in project")

            # ── Step 3: Embed ─────────────────────────────────────────────────
            yield sse_step("embed", "Understanding your question...")
            from core.embeddings.embedder import embed_query
            query_vec = await loop.run_in_executor(None, embed_query, body.question)
            yield sse_step("embed_done", "Query embedded")

            # ── Step 4: Semantic search ───────────────────────────────────────
            top_k = max(1, min(body.top_k, 20))
            yield sse_step("retrieve", f"Searching {count} indexed documents...")
            from core.retrieval.vector_store import query_chunks
            raw = await loop.run_in_executor(
                None, partial(query_chunks, query_vec, namespace=body.namespace, top_k=top_k)
            )
            yield sse_step("retrieve_done", f"Found {len(raw)} semantic matches")

            # ── Step 5: Keyword search for log queries ────────────────────────
            from core.retrieval.rag import _is_log_query, _extract_search_keywords
            kw_chunks = []

            if _is_log_query(body.question):
                keywords = _extract_search_keywords(body.question)
                if keywords:
                    yield sse_step("keyword_search",
                                   f"Enhancing with keyword search: {', '.join(keywords[:5])}")
                    from core.retrieval.vector_store import keyword_search
                    kw_chunks = await loop.run_in_executor(
                        None, partial(keyword_search, body.namespace, keywords, top_k)
                    )
                    yield sse_step("keyword_done",
                                   f"Found {len(kw_chunks)} keyword matches")

            # ── Step 6: Merge & rank ──────────────────────────────────────────
            yield sse_step("rank", "Ranking results by relevance...")
            all_chunks = raw + kw_chunks
            seen = set()
            deduped = []
            for c in all_chunks:
                key = (c.source_file, c.line_start, c.line_end)
                if key not in seen:
                    seen.add(key)
                    deduped.append(c)

            from core.retrieval.ranker import rank_results, has_relevant_results, top_relevant
            ranked = rank_results(deduped)

            if ranked:
                yield sse_step("rank_done",
                               f"Best: {ranked[0].confidence:.0%} in {ranked[0].source_file}")
            else:
                yield sse_step("rank_done", "No matches found")

            # ── Step 7: Generate answer ───────────────────────────────────────
            from core.llm.llm_service import get_llm_service
            is_log_q = _is_log_query(body.question)

            if has_relevant_results(ranked):
                relevant = top_relevant(ranked, n=5)

                # ── Run analysis pipeline for log queries ─────────────────
                if is_log_q:
                    from core.analysis.trends import analyze_error_trends
                    from core.analysis.clusters import cluster_log_events
                    from core.analysis.confidence import calculate_confidence

                    yield sse_step("trend_analysis", "Detecting error trends...")
                    trend_report = await loop.run_in_executor(
                        None, analyze_error_trends, relevant)
                    yield sse_step("trend_done",
                        f"{trend_report.hourly_counts.total_errors} errors, "
                        f"{len(trend_report.spikes)} spikes, "
                        f"{len(trend_report.repeating_failures)} repeating failures")

                    yield sse_step("cluster_analysis", "Clustering log events...")
                    cluster_report = await loop.run_in_executor(
                        None, cluster_log_events, relevant)
                    yield sse_step("cluster_done",
                        f"Found {cluster_report.cluster_count} incident cluster(s)")

                    yield sse_step("confidence_scoring", "Computing multi-factor confidence...")
                    conf_result = await loop.run_in_executor(
                        None, partial(calculate_confidence, relevant, "auto", body.question))
                    yield sse_step("confidence_done",
                        f"Confidence: {conf_result.confidence_score:.0%} ({conf_result.confidence_level})")

                    yield sse_step("generate", f"Generating structured log analysis from {len(relevant)} sources...")
                    svc = get_llm_service()
                    llm_resp = await loop.run_in_executor(
                        None, partial(
                            svc.generate_log_summary, body.question, relevant,
                            trend_report.summary_text(),
                            cluster_report.summary_text(),
                            conf_result.reasoning,
                        )
                    )
                    yield sse_step("generate_done", f"Log analysis ready ({llm_resp.mode})")

                    from core.retrieval.rag import _build_sources, RAGResponse
                    result = RAGResponse(
                        answer=llm_resp.answer,
                        confidence=conf_result.confidence_score,
                        no_data=llm_resp.no_data,
                        mode=llm_resp.mode,
                        sources=_build_sources(relevant),
                    )
                else:
                    yield sse_step("generate", f"Generating answer from {len(relevant)} sources...")
                    svc = get_llm_service()
                    llm_resp = await loop.run_in_executor(
                        None, partial(svc.generate_response, relevant, body.question)
                    )
                    yield sse_step("generate_done", f"Answer ready ({llm_resp.mode})")

                    from core.retrieval.rag import _build_sources, RAGResponse
                    result = RAGResponse(
                        answer=llm_resp.answer,
                        confidence=relevant[0].confidence,
                        no_data=llm_resp.no_data,
                        mode=llm_resp.mode,
                        sources=_build_sources(relevant),
                    )

            elif ranked and is_log_q:
                # ── Low confidence log query: analysis + summarize ────────
                from core.analysis.trends import analyze_error_trends
                from core.analysis.clusters import cluster_log_events
                from core.analysis.confidence import calculate_confidence

                analysis_chunks = ranked[:5]

                yield sse_step("trend_analysis", "Detecting error trends...")
                trend_report = await loop.run_in_executor(
                    None, analyze_error_trends, analysis_chunks)
                yield sse_step("trend_done",
                    f"{trend_report.hourly_counts.total_errors} errors, "
                    f"{len(trend_report.spikes)} spikes")

                yield sse_step("cluster_analysis", "Clustering log events...")
                cluster_report = await loop.run_in_executor(
                    None, cluster_log_events, analysis_chunks)
                yield sse_step("cluster_done",
                    f"Found {cluster_report.cluster_count} cluster(s)")

                yield sse_step("confidence_scoring", "Computing multi-factor confidence...")
                conf_result = await loop.run_in_executor(
                    None, partial(calculate_confidence, analysis_chunks, "auto", body.question))
                yield sse_step("confidence_done",
                    f"Confidence: {conf_result.confidence_score:.0%} ({conf_result.confidence_level})")

                yield sse_step("log_summarize", "Summarising log entries for errors...")
                svc = get_llm_service()
                llm_resp = await loop.run_in_executor(
                    None, partial(
                        svc.generate_log_summary, body.question, analysis_chunks,
                        trend_report.summary_text(),
                        cluster_report.summary_text(),
                        conf_result.reasoning,
                    )
                )
                yield sse_step("log_summarize_done", "Log summary ready")

                from core.retrieval.rag import _build_sources, RAGResponse
                result = RAGResponse(
                    answer=llm_resp.answer,
                    confidence=conf_result.confidence_score,
                    no_data=llm_resp.no_data,
                    mode=llm_resp.mode,
                    sources=_build_sources(analysis_chunks),
                )

            elif ranked:
                yield sse_step("clarify", "Preparing clarification question...")
                svc = get_llm_service()
                llm_resp = await loop.run_in_executor(
                    None, partial(
                        svc.generate_clarification,
                        body.question, ranked[:5], ranked[0].confidence
                    )
                )
                yield sse_step("clarify_done", "Clarification ready")

                from core.retrieval.rag import _build_sources, RAGResponse
                result = RAGResponse(
                    answer=llm_resp.answer,
                    confidence=ranked[0].confidence,
                    no_data=False,
                    mode="clarification",
                    sources=_build_sources(ranked[:3]),
                )
            else:
                from core.retrieval.rag import RAGResponse
                result = RAGResponse(
                    answer="No similar incident found.",
                    confidence=0.0,
                    no_data=True,
                    mode="extractive",
                    sources=[],
                )

            yield sse_step("complete", "Done!")
            yield sse_result(_response_to_dict(result))

        except Exception as exc:
            yield sse_result({"error": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
