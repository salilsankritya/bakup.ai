"""
core/retrieval/ranker.py
─────────────────────────────────────────────────────────────────────────────
Converts raw cosine distances into human-readable confidence scores and
applies the "no relevant data" threshold.

Confidence derivation:
    cosine distance (ChromaDB) = 1 - cosine_similarity
    confidence = max(0.0, 1.0 - distance)

    For L2-normalised embeddings the range is [0, 1]:
      1.0 → identical content
      0.5 → moderate relevance
      0.0 → unrelated

Threshold policy:
    If the best-match confidence is below CONFIDENCE_THRESHOLD, the entire
    result set is considered "no relevant data found". The caller must
    honour this signal and not fabricate an answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

from core.retrieval.models import RetrievedChunk

# Chunks below this confidence are considered not relevant.
# Configurable via environment variable.
_DEFAULT_THRESHOLD = 0.35


def _get_threshold() -> float:
    return float(os.environ.get("BAKUP_CONFIDENCE_THRESHOLD", str(_DEFAULT_THRESHOLD)))


@dataclass(frozen=True)
class RankedResult:
    """A retrieved chunk with a normalised confidence score attached."""
    text: str
    source_file: str
    line_start: int
    line_end: int
    source_type: str           # "code" | "log"
    confidence: float          # [0.0, 1.0] — higher is more relevant
    confidence_label: str      # "high" | "medium" | "low"


def distance_to_confidence(distance: float) -> float:
    """
    Convert a ChromaDB cosine distance to a [0, 1] confidence score.
    Clamps to valid range.
    """
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


def _label(confidence: float) -> str:
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def rank_results(chunks: List[RetrievedChunk]) -> List[RankedResult]:
    """
    Convert retrieved chunks to RankedResults, sorted by confidence descending.
    """
    ranked = [
        RankedResult(
            text=c.text,
            source_file=c.source_file,
            line_start=c.line_start,
            line_end=c.line_end,
            source_type=c.source_type,
            confidence=distance_to_confidence(c.distance),
            confidence_label=_label(distance_to_confidence(c.distance)),
        )
        for c in chunks
    ]
    ranked.sort(key=lambda r: r.confidence, reverse=True)
    return ranked


def has_relevant_results(ranked: List[RankedResult]) -> bool:
    """
    Returns True only if at least one result meets the confidence threshold.
    If False, the caller must return "No similar incident found."
    """
    if not ranked:
        return False
    threshold = _get_threshold()
    return ranked[0].confidence >= threshold


def top_relevant(ranked: List[RankedResult], n: int = 5) -> List[RankedResult]:
    """
    Return the top n results that meet the confidence threshold.
    Returns an empty list if no results are relevant.
    """
    threshold = _get_threshold()
    return [r for r in ranked if r.confidence >= threshold][:n]
