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
import logging
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger("bakup.query")


# ── Input validation ──────────────────────────────────────────────────────────

import re

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")


def _validate_namespace(ns: str) -> None:
    """Reject namespaces with unexpected characters (injection guard)."""
    if not ns or not _NAMESPACE_RE.match(ns):
        raise HTTPException(
            status_code=400,
            detail="Invalid namespace format. Expected 1-100 alphanumeric/dash/underscore characters.",
        )

router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., max_length=10000)
    namespace: str = Field(..., min_length=1, max_length=100)  # Must match the namespace used during /index
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


def _brain_to_query_response(brain_result) -> QueryResponse:
    """Convert a BrainResponse into the API QueryResponse model."""
    # Normalise mode — strip "fallback:" prefix for API compatibility
    mode = brain_result.mode
    if mode.startswith("fallback:"):
        mode = mode[len("fallback:"):]

    sources = []
    for s in brain_result.sources:
        if isinstance(s, dict):
            sources.append(SourceModel(
                file=s.get("file", ""),
                line_start=s.get("line_start", 0),
                line_end=s.get("line_end", 0),
                excerpt=s.get("excerpt", ""),
                confidence=s.get("confidence", 0.0),
                confidence_label=s.get("confidence_label", "low"),
                source_type=s.get("source_type", "code"),
            ))
        else:
            sources.append(SourceModel(
                file=s.file,
                line_start=s.line_start,
                line_end=s.line_end,
                excerpt=s.excerpt,
                confidence=s.confidence,
                confidence_label=s.confidence_label,
                source_type=s.source_type,
            ))

    return QueryResponse(
        answer=brain_result.answer,
        confidence=brain_result.confidence,
        no_data=brain_result.no_data,
        mode=mode,
        sources=sources,
        debug_trace=brain_result.reasoning_trace if brain_result.reasoning_trace else None,
    )


# ── Route: POST /ask ─────────────────────────────────────────────────────────

@router.post("/ask", response_model=QueryResponse, tags=["query"])
async def ask(body: QueryRequest) -> QueryResponse:
    """
    Classify the question, then route through the brain controller.

    The brain controller decides whether to use LLM-orchestrated tool
    calling (when an LLM is configured) or the deterministic RAG pipeline.

    Classification happens first:
      - greeting  → immediate polite reply (no DB access)
      - off_topic → scope guard reply (no DB access)
      - project   → brain controller (LLM tools or fallback RAG)

    Set debug=true to include a step-by-step pipeline trace in the response.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # ── Hybrid Router ─────────────────────────────────────────────────────────
    from core.router.router import route_query, get_scope_message

    routing = route_query(
        query=body.question,
        namespace=body.namespace or "",
    )

    # Conversational → short-circuit via brain (no retrieval needed)
    if routing.intent == "conversational":
        from core.classifier.query_classifier import classify_query, QueryCategory
        sub = classify_query(body.question)
        pre = sub.value if sub in (QueryCategory.GREETING, QueryCategory.CONVERSATIONAL) else "conversational"

        from core.brain.brain import process_query, store_debug_result
        brain_result = process_query(
            question=body.question,
            namespace="_",
            top_k=1,
            debug=body.debug,
            pre_classified=pre,
        )
        return _brain_to_query_response(brain_result)

    # Unrelated → scope guard (no retrieval needed)
    if routing.intent == "unrelated":
        from core.brain.brain import process_query
        brain_result = process_query(
            question=body.question,
            namespace="_",
            top_k=1,
            debug=body.debug,
            pre_classified="off_topic",
        )
        return _brain_to_query_response(brain_result)

    # ── Project questions require a valid namespace ───────────────────────────
    has_namespace = bool(body.namespace and body.namespace.strip())
    if not has_namespace:
        raise HTTPException(status_code=400, detail="Namespace must not be empty. Use the namespace returned by /index.")

    _validate_namespace(body.namespace)

    top_k = max(1, min(body.top_k, 20))

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

    # ── Get session context for follow-up support ─────────────────────────────
    from core.retrieval.session import get_session
    session = get_session(body.namespace)
    session_context = ""
    if session.is_follow_up(body.question) and session.turn_count > 0:
        session_context = session.format_context()

    try:
        from core.brain.brain import process_query, store_debug_result

        brain_result = process_query(
            question=body.question,
            namespace=body.namespace,
            top_k=top_k,
            debug=body.debug,
            pre_classified="project",
            session_context=session_context,
        )

        # Store for debug endpoint
        store_debug_result(body.namespace, brain_result)

        # Store turn in session memory for follow-up support
        from core.retrieval.session import add_turn
        source_files = list({s.get("file", "") for s in brain_result.sources[:10]})
        add_turn(
            body.namespace,
            body.question,
            brain_result.answer,
            source_files=source_files,
            question_type=brain_result.mode,
            evidence_summary=f"{len(brain_result.tool_calls)} tool calls" if brain_result.tool_calls else "fallback pipeline",
        )

    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Query processing failed. Check server logs.")

    return _brain_to_query_response(brain_result)


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
            # ── Step 1: Hybrid Router ─────────────────────────────────────────
            yield sse_step("classify", "Routing your question...")

            from core.router.router import route_query
            routing = await loop.run_in_executor(
                None, partial(route_query, body.question, body.namespace or "")
            )

            # Conversational → short-circuit
            if routing.intent == "conversational":
                # Distinguish greeting vs conversational via regex sub-classifier
                from core.classifier.query_classifier import classify_query, QueryCategory
                sub = await loop.run_in_executor(None, classify_query, body.question)

                if sub == QueryCategory.GREETING:
                    yield sse_step("classify_done", "Greeting detected")
                    from core.retrieval.rag import answer_question
                    result = await loop.run_in_executor(
                        None, partial(answer_question, body.question, "_", 1, True,
                                      pre_classified="greeting")
                    )
                    yield sse_result(_response_to_dict(result))
                    return

                yield sse_step("classify_done", f"Conversational question — no retrieval needed (via {routing.source})")
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

            # Unrelated → scope guard
            if routing.intent == "unrelated":
                yield sse_step("classify_done", f"Outside project scope (via {routing.source})")
                from core.retrieval.rag import answer_question
                result = await loop.run_in_executor(
                    None, partial(answer_question, body.question, "_", 1, True,
                                  pre_classified="off_topic")
                )
                yield sse_result(_response_to_dict(result))
                return

            # Project query → full RAG pipeline
            yield sse_step("classify_done", f"Project question detected (via {routing.source}, {routing.confidence:.0%})")

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
            sev_chunks = []

            if _is_log_query(body.question):
                keywords = _extract_search_keywords(body.question)
                if keywords:
                    yield sse_step("keyword_search",
                                   f"Enhancing with keyword search: {', '.join(keywords[:5])}")
                    from core.retrieval.vector_store import keyword_search, severity_search
                    kw_chunks = await loop.run_in_executor(
                        None, partial(keyword_search, body.namespace, keywords, top_k)
                    )
                    yield sse_step("keyword_done",
                                   f"Found {len(kw_chunks)} keyword matches")

                # Pull error-severity chunks for cross-file distribution
                from core.retrieval.vector_store import severity_search
                yield sse_step("severity_search", "Searching for error-severity chunks...")
                sev_chunks = await loop.run_in_executor(
                    None, partial(severity_search, body.namespace, "error", top_k * 2)
                )
                yield sse_step("severity_done", f"Found {len(sev_chunks)} error chunks")

            # ── Step 6: Merge & rank ──────────────────────────────────────────
            yield sse_step("rank", "Ranking results by relevance...")
            all_chunks = raw + kw_chunks + sev_chunks
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

                    # File aggregation
                    from core.analysis.file_aggregation import aggregate_by_file
                    yield sse_step("file_aggregation", "Aggregating errors across files...")
                    agg_report = await loop.run_in_executor(
                        None, aggregate_by_file, relevant)
                    yield sse_step("file_aggregation_done",
                        f"{agg_report.files_affected} file(s), {agg_report.total_errors} error(s)")

                    yield sse_step("generate", f"Generating structured log analysis from {len(relevant)} sources...")
                    svc = get_llm_service()
                    llm_resp = await loop.run_in_executor(
                        None, partial(
                            svc.generate_log_summary, body.question, relevant,
                            trend_report.summary_text(),
                            cluster_report.summary_text(),
                            conf_result.reasoning,
                            agg_report.summary_text,
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

                # File aggregation
                from core.analysis.file_aggregation import aggregate_by_file
                yield sse_step("file_aggregation", "Aggregating errors across files...")
                agg_report = await loop.run_in_executor(
                    None, aggregate_by_file, analysis_chunks)
                yield sse_step("file_aggregation_done",
                    f"{agg_report.files_affected} file(s), {agg_report.total_errors} error(s)")

                yield sse_step("log_summarize", "Summarising log entries for errors...")
                svc = get_llm_service()
                llm_resp = await loop.run_in_executor(
                    None, partial(
                        svc.generate_log_summary, body.question, analysis_chunks,
                        trend_report.summary_text(),
                        cluster_report.summary_text(),
                        conf_result.reasoning,
                        agg_report.summary_text,
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
                # Low embedding confidence but chunks exist.
                # When LLM is configured, generate an answer instead of
                # asking the user to rephrase.
                from core.llm.config_store import load_config as _load_llm_cfg
                llm_configured = _load_llm_cfg().configured

                if llm_configured:
                    yield sse_step("low_conf_llm",
                        f"Low confidence ({ranked[0].confidence:.0%}) — generating LLM analysis...")
                    svc = get_llm_service()

                    # Use code review prompt for broad analytical questions
                    from core.retrieval.planner import classify_question, QuestionType
                    q_type = classify_question(body.question)
                    if q_type == QuestionType.CODE_REVIEW:
                        llm_resp = await loop.run_in_executor(
                            None, partial(svc.generate_code_review, ranked[:8], body.question)
                        )
                    else:
                        llm_resp = await loop.run_in_executor(
                            None, partial(svc.generate_response, ranked[:5], body.question)
                        )
                    yield sse_step("generate_done", f"Answer ready ({llm_resp.mode})")

                    from core.retrieval.rag import _build_sources, RAGResponse
                    result = RAGResponse(
                        answer=llm_resp.answer,
                        confidence=ranked[0].confidence,
                        no_data=llm_resp.no_data,
                        mode=llm_resp.mode,
                        sources=_build_sources(ranked[:5]),
                    )
                else:
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
            import traceback
            logger.error("SSE stream error: %s", traceback.format_exc())
            yield sse_result({"error": "An internal error occurred. Check server logs."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
