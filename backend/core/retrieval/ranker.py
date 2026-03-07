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
from typing import List, Optional, Tuple

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
    file_name: str = ""
    severity: str = "info"
    detected_timestamp: Optional[str] = None
    # ── Code-aware metadata ───────────────────────────────────────────
    language: str = ""
    function_name: str = ""
    class_name: str = ""
    chunk_kind: str = ""
    docstring: str = ""
    imports: str = ""


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
    Applies code-aware boosting:
      - functions/methods/classes get a +0.05 boost
      - chunks with docstrings get a +0.02 boost
    """
    ranked = []
    for c in chunks:
        conf = distance_to_confidence(c.distance)

        # Code-aware boosting
        kind = getattr(c, 'chunk_kind', '') or ''
        if kind in ('function', 'method', 'class'):
            conf = min(1.0, round(conf + 0.05, 4))
        if getattr(c, 'docstring', ''):
            conf = min(1.0, round(conf + 0.02, 4))

        ranked.append(RankedResult(
            text=c.text,
            source_file=c.source_file,
            line_start=c.line_start,
            line_end=c.line_end,
            source_type=c.source_type,
            confidence=conf,
            confidence_label=_label(conf),
            file_name=getattr(c, 'file_name', '') or '',
            severity=getattr(c, 'severity', 'info') or 'info',
            detected_timestamp=getattr(c, 'detected_timestamp', None),
            language=getattr(c, 'language', '') or '',
            function_name=getattr(c, 'function_name', '') or '',
            class_name=getattr(c, 'class_name', '') or '',
            chunk_kind=getattr(c, 'chunk_kind', '') or '',
            docstring=getattr(c, 'docstring', '') or '',
            imports=getattr(c, 'imports', '') or '',
        ))

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
