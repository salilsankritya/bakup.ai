"""
core/retrieval/multi_query.py
─────────────────────────────────────────────────────────────────────────────
Multi-query retrieval — generates semantic variants of a user question and
runs retrieval for each variant to improve recall.

Strategy:
    1. Generate 2–4 rule-based query variants (keyword extraction, rephrasing).
    2. Embed each variant and run vector search.
    3. Merge results using reciprocal rank fusion (RRF).
    4. Deduplicate by (source_file, line_start, line_end).

This module is LLM-free — it uses deterministic keyword transforms so it
works regardless of whether an LLM provider is configured.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Tuple

from core.retrieval.models import RetrievedChunk

logger = logging.getLogger("bakup.retrieval.multi_query")

# ── RRF constant (standard value from literature) ─────────────────────────────
_RRF_K = 60


# ── Query variant generation ──────────────────────────────────────────────────

# Stop-words to strip when extracting keywords
_STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "so", "then", "than",
    "about", "any", "all", "each", "every", "some",
    "show", "tell", "find", "get", "give", "explain",
})

# Technical synonyms for expansion
_SYNONYMS: Dict[str, List[str]] = {
    "error": ["exception", "failure", "fault"],
    "exception": ["error", "traceback"],
    "crash": ["failure", "fatal", "abort"],
    "timeout": ["timed out", "deadline exceeded"],
    "bug": ["defect", "issue", "problem"],
    "performance": ["slow", "latency", "bottleneck"],
    "memory": ["OOM", "out of memory", "heap"],
    "null": ["NoneType", "NullPointerException", "undefined"],
    "import": ["dependency", "require", "module"],
    "function": ["method", "def", "handler"],
    "class": ["type", "object", "struct"],
    "database": ["db", "SQL", "query"],
    "api": ["endpoint", "route", "handler"],
    "config": ["configuration", "settings", "environment"],
    "deploy": ["deployment", "release", "build"],
    "test": ["spec", "assert", "unittest"],
}


def generate_query_variants(question: str) -> List[str]:
    """
    Generate 2–4 deterministic query variants from the original question.

    Returns:
        List of query strings including the original as the first element.
    """
    variants: List[str] = [question]  # Original always first

    # Variant 1: Keyword-only query (strip stop-words)
    keyword_query = _extract_keywords(question)
    if keyword_query and keyword_query != question.lower().strip():
        variants.append(keyword_query)

    # Variant 2: Synonym-expanded query
    synonym_query = _expand_synonyms(question)
    if synonym_query and synonym_query != question:
        variants.append(synonym_query)

    # Variant 3: Technical focus — extract identifiers and error patterns
    tech_query = _extract_technical_focus(question)
    if tech_query and tech_query not in variants:
        variants.append(tech_query)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for v in variants:
        normalised = v.strip().lower()
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(v)

    logger.debug(
        "Multi-query: %d variant(s) from question '%s': %s",
        len(unique), question[:80], [v[:60] for v in unique],
    )

    return unique[:4]  # Cap at 4 variants


def _extract_keywords(question: str) -> str:
    """Extract non-stop-word tokens from the question."""
    tokens = re.findall(r"[a-zA-Z0-9_./-]+", question)
    keywords = [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 1]
    return " ".join(keywords)


def _expand_synonyms(question: str) -> str:
    """Replace one keyword with its synonym to broaden recall."""
    q_lower = question.lower()
    for word, syns in _SYNONYMS.items():
        if re.search(rf"\b{re.escape(word)}\b", q_lower):
            # Pick the first synonym that isn't already in the question
            for syn in syns:
                if syn.lower() not in q_lower:
                    return re.sub(
                        rf"\b{re.escape(word)}\b",
                        syn,
                        question,
                        count=1,
                        flags=re.IGNORECASE,
                    )
    return ""


def _extract_technical_focus(question: str) -> str:
    """
    Pull out dotted identifiers, CamelCase names, file paths,
    and error-like patterns into a focused retrieval query.
    """
    patterns: List[str] = []

    # Dotted identifiers (e.g. core.retrieval.rag)
    patterns.extend(re.findall(r"[a-zA-Z_]\w+(?:\.\w+){1,}", question))

    # CamelCase / PascalCase names
    patterns.extend(re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", question))

    # snake_case identifiers (2+ segments)
    patterns.extend(re.findall(r"\b[a-z]+(?:_[a-z]+){1,}\b", question))

    # File paths
    patterns.extend(re.findall(r"[\w./\\-]+\.\w{1,5}", question))

    # Error-like patterns (e.g., "NullPointerException", "ECONNREFUSED")
    patterns.extend(re.findall(r"\b[A-Z][A-Z_]{2,}\b", question))

    if patterns:
        return " ".join(dict.fromkeys(patterns))  # dedup preserving order
    return ""


# ── Multi-query retrieval execution ───────────────────────────────────────────

def multi_query_retrieve(
    question: str,
    namespace: str,
    top_k: int = 8,
) -> List[RetrievedChunk]:
    """
    Run multi-query retrieval: generate variants, embed each, retrieve,
    and fuse results using reciprocal rank fusion.

    Args:
        question:   The user's original question.
        namespace:  Project namespace.
        top_k:      Number of results per variant.

    Returns:
        Deduplicated list of RetrievedChunk sorted by fused score (best first).
    """
    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import query_chunks

    variants = generate_query_variants(question)

    logger.info(
        "Multi-query retrieval: %d variant(s) for namespace '%s'",
        len(variants), namespace,
    )

    # Collect ranked lists per variant
    ranked_lists: List[List[RetrievedChunk]] = []
    for variant in variants:
        embedding = embed_query(variant)
        chunks = query_chunks(embedding, namespace=namespace, top_k=top_k)
        ranked_lists.append(chunks)
        logger.debug(
            "  Variant '%s': %d chunk(s) retrieved",
            variant[:60], len(chunks),
        )

    # Fuse using reciprocal rank fusion
    fused = _reciprocal_rank_fusion(ranked_lists)

    logger.info(
        "Multi-query fusion: %d unique chunk(s) from %d variant(s)",
        len(fused), len(variants),
    )

    return fused


def _reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievedChunk]],
) -> List[RetrievedChunk]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) for each list where the doc appears.
    Lower distance is kept when the same chunk appears in multiple lists.
    """
    # chunk_key → (rrf_score, best_chunk)
    scores: Dict[Tuple[str, int, int], Tuple[float, RetrievedChunk]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            key = (chunk.source_file, chunk.line_start, chunk.line_end)
            rrf_score = 1.0 / (_RRF_K + rank + 1)  # rank is 0-based

            if key in scores:
                existing_score, existing_chunk = scores[key]
                # Accumulate RRF score; keep the chunk with the lower distance
                best = chunk if chunk.distance < existing_chunk.distance else existing_chunk
                scores[key] = (existing_score + rrf_score, best)
            else:
                scores[key] = (rrf_score, chunk)

    # Sort by descending RRF score
    sorted_items = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in sorted_items]
