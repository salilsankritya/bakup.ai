"""
core/retrieval/dedup.py
─────────────────────────────────────────────────────────────────────────────
Evidence deduplication — removes duplicate and overlapping chunks from
retrieval results to maximise information density in the LLM context.

Three strategies:
    1. Exact dedup — identical (source_file, line_start, line_end) tuples.
    2. Overlap merging — chunks from the same file whose line ranges
       overlap are merged (keep the higher-confidence one).
    3. Limit enforcement — caps per category (logs, code, clusters)
       so the context window isn't dominated by one type.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

from core.retrieval.ranker import RankedResult
from core.retrieval.models import RetrievedChunk

logger = logging.getLogger("bakup.retrieval.dedup")


# ── Deduplication of RetrievedChunks (pre-ranking) ────────────────────────────

def deduplicate_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Remove exact-duplicate and overlapping chunks.
    Keeps the chunk with the lower distance (higher similarity) on conflict.
    """
    if not chunks:
        return []

    # Phase 1: Exact dedup by (source_file, line_start, line_end)
    seen: Dict[Tuple[str, int, int], RetrievedChunk] = {}
    for c in chunks:
        key = (c.source_file, c.line_start, c.line_end)
        if key not in seen or c.distance < seen[key].distance:
            seen[key] = c

    unique = list(seen.values())

    # Phase 2: Merge overlapping ranges within the same file
    unique = _merge_overlapping(unique)

    logger.debug(
        "Dedup: %d input → %d unique chunk(s) (removed %d)",
        len(chunks), len(unique), len(chunks) - len(unique),
    )
    return unique


def _merge_overlapping(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    For chunks from the same file with overlapping line ranges,
    keep only the one with the lower distance (better match).
    Two chunks overlap if one's start..end intersects the other's.
    """
    # Group by source file
    by_file: Dict[str, List[RetrievedChunk]] = {}
    for c in chunks:
        by_file.setdefault(c.source_file, []).append(c)

    result: List[RetrievedChunk] = []
    for file_chunks in by_file.values():
        # Sort by line_start
        file_chunks.sort(key=lambda c: (c.line_start, c.line_end))
        kept: List[RetrievedChunk] = []

        for c in file_chunks:
            merged = False
            for i, existing in enumerate(kept):
                if _ranges_overlap(existing.line_start, existing.line_end,
                                   c.line_start, c.line_end):
                    # Keep the one with lower distance (better match)
                    if c.distance < existing.distance:
                        kept[i] = c
                    merged = True
                    break
            if not merged:
                kept.append(c)

        result.extend(kept)

    return result


def _ranges_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    """Check if two line ranges overlap (inclusive)."""
    return s1 <= e2 and s2 <= e1


# ── Deduplication of RankedResults (post-ranking) ─────────────────────────────

def deduplicate_ranked(ranked: List[RankedResult]) -> List[RankedResult]:
    """
    Remove exact-duplicate and overlapping RankedResults.
    Keeps the one with the higher confidence on conflict.
    """
    if not ranked:
        return []

    # Phase 1: Exact dedup
    seen: Dict[Tuple[str, int, int], RankedResult] = {}
    for r in ranked:
        key = (r.source_file, r.line_start, r.line_end)
        if key not in seen or r.confidence > seen[key].confidence:
            seen[key] = r

    unique = list(seen.values())

    # Phase 2: Merge overlapping ranges
    unique = _merge_overlapping_ranked(unique)

    # Re-sort by confidence
    unique.sort(key=lambda r: r.confidence, reverse=True)

    logger.debug(
        "Dedup ranked: %d input → %d unique",
        len(ranked), len(unique),
    )
    return unique


def _merge_overlapping_ranked(ranked: List[RankedResult]) -> List[RankedResult]:
    """Merge overlapping ranked results within the same file."""
    by_file: Dict[str, List[RankedResult]] = {}
    for r in ranked:
        by_file.setdefault(r.source_file, []).append(r)

    result: List[RankedResult] = []
    for file_results in by_file.values():
        file_results.sort(key=lambda r: (r.line_start, r.line_end))
        kept: List[RankedResult] = []

        for r in file_results:
            merged = False
            for i, existing in enumerate(kept):
                if _ranges_overlap(existing.line_start, existing.line_end,
                                   r.line_start, r.line_end):
                    if r.confidence > existing.confidence:
                        kept[i] = r
                    merged = True
                    break
            if not merged:
                kept.append(r)

        result.extend(kept)

    return result


# ── Category-based limit enforcement ──────────────────────────────────────────

def enforce_evidence_limits(
    logs: List[RankedResult],
    code: List[RankedResult],
    *,
    max_logs: int = 5,
    max_code: int = 5,
) -> Tuple[List[RankedResult], List[RankedResult]]:
    """
    Enforce per-category evidence limits.

    Already-sorted lists are truncated to the specified maximums.
    This prevents any single evidence type from dominating the context.

    Returns:
        (logs, code) — truncated lists.
    """
    limited_logs = deduplicate_ranked(logs)[:max_logs]
    limited_code = deduplicate_ranked(code)[:max_code]

    if len(logs) > max_logs:
        logger.debug("Limit: logs %d → %d", len(logs), max_logs)
    if len(code) > max_code:
        logger.debug("Limit: code %d → %d", len(code), max_code)

    return limited_logs, limited_code
