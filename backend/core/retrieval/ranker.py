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

Multi-signal boosting (v2):
    - Code-aware:    +0.05 for function/method/class, +0.02 for docstring
    - Error signal:  +0.08 for chunks containing error/exception/traceback keywords
    - Stack trace:   +0.06 for chunks containing stack trace patterns
    - Recency:       +0.03 for chunks with timestamps in the last 24h

Threshold policy:
    If the best-match confidence is below CONFIDENCE_THRESHOLD, the entire
    result set is considered "no relevant data found". The caller must
    honour this signal and not fabricate an answer.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from core.retrieval.models import RetrievedChunk

logger = logging.getLogger("bakup.retrieval.ranker")

# Chunks below this confidence are considered not relevant.
# Configurable via environment variable.
_DEFAULT_THRESHOLD = 0.35

# ── Error / stack-trace signal patterns ───────────────────────────────────────

_ERROR_KEYWORDS = re.compile(
    r"\b(ERROR|Exception|Traceback|FATAL|CRITICAL|FAILED|FAILURE|panic|segfault|"
    r"OutOfMemoryError|NullPointerException|StackOverflowError|"
    r"RuntimeError|ValueError|TypeError|KeyError|AttributeError|"
    r"IOError|OSError|ConnectionError|TimeoutError)\b",
    re.IGNORECASE,
)

_STACK_TRACE_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"^\s+at\s+\S+\(.*:\d+\)|"      # Java/JS stack frames
    r"^\s+File\s+\".*\",\s+line\s+\d+|"  # Python stack frames
    r"goroutine\s+\d+|"             # Go stack traces
    r"^\s+\d+:\s+0x[0-9a-f]+\s+)",  # C/Rust stack frames
    re.MULTILINE | re.IGNORECASE,
)


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

    Multi-signal boosting (v2):
      - Code-aware:    +0.05 for function/method/class, +0.02 for docstring
      - Error signal:  +0.08 for chunks containing error/exception keywords
      - Stack trace:   +0.06 for chunks containing stack trace patterns
      - Recency:       +0.03 for chunks with recent timestamps (≤ 24h)

    All boosts are capped at 1.0.
    """
    ranked = []
    for c in chunks:
        conf = distance_to_confidence(c.distance)
        boosts: List[Tuple[str, float]] = []  # (reason, amount) for debug

        # ── Code-aware boosting ───────────────────────────────────────────
        kind = getattr(c, 'chunk_kind', '') or ''
        if kind in ('function', 'method', 'class'):
            boosts.append(("code_structure", 0.05))
        if getattr(c, 'docstring', ''):
            boosts.append(("docstring", 0.02))

        # ── Error keyword boosting ────────────────────────────────────────
        text = c.text or ''
        if _ERROR_KEYWORDS.search(text):
            boosts.append(("error_keyword", 0.08))

        # ── Stack trace boosting ──────────────────────────────────────────
        if _STACK_TRACE_PATTERNS.search(text):
            boosts.append(("stack_trace", 0.06))

        # ── Recency boosting ─────────────────────────────────────────────
        ts = getattr(c, 'detected_timestamp', None)
        if ts:
            recency_boost = _recency_boost(ts)
            if recency_boost > 0:
                boosts.append(("recency", recency_boost))

        # Apply all boosts
        total_boost = sum(b for _, b in boosts)
        if total_boost > 0:
            conf = min(1.0, round(conf + total_boost, 4))

        if boosts:
            logger.debug(
                "Boost %s L%d–%d: base=%.4f +%s = %.4f",
                c.source_file, c.line_start, c.line_end,
                conf - total_boost, boosts, conf,
            )

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


def _recency_boost(timestamp_str: str) -> float:
    """
    Calculate a recency boost based on the timestamp.
    Returns:
        0.03 if within 24 hours
        0.01 if within 7 days
        0.0  otherwise
    """
    try:
        # Try common timestamp formats
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ):
            try:
                ts = datetime.strptime(timestamp_str, fmt)
                age = datetime.now() - ts
                if age <= timedelta(hours=24):
                    return 0.03
                if age <= timedelta(days=7):
                    return 0.01
                return 0.0
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0


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
