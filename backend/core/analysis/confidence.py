"""
core/analysis/confidence.py
─────────────────────────────────────────────────────────────────────────────
Multi-factor confidence scoring for bakup.ai.

Replaces naive similarity-only scoring with a composite model that
considers:
  1. Retrieval similarity (embedding distance)
  2. Number of relevant chunks found
  3. Severity match (error > warning > info)
  4. Recency weighting (newer log entries weigh more)
  5. Query intent type (broad summary vs specific match)

Returns a structured, deterministic, explainable confidence result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


# ── Severity levels ────────────────────────────────────────────────────────────

_SEVERITY_PATTERNS = {
    "fatal":    re.compile(r"\bFATAL\b",                           re.IGNORECASE),
    "critical": re.compile(r"\bCRITICAL\b",                       re.IGNORECASE),
    "error":    re.compile(r"\bERROR\b|Exception|Traceback",       re.IGNORECASE),
    "warning":  re.compile(r"\bWARN(?:ING)?\b",                    re.IGNORECASE),
    "info":     re.compile(r"\bINFO\b",                            re.IGNORECASE),
    "debug":    re.compile(r"\bDEBUG\b",                           re.IGNORECASE),
}

_SEVERITY_WEIGHT = {
    "fatal":    1.0,
    "critical": 0.95,
    "error":    0.85,
    "warning":  0.55,
    "info":     0.2,
    "debug":    0.1,
    "unknown":  0.15,
}


# ── Timestamp extractor ───────────────────────────────────────────────────────

_TS_PATTERNS = [
    # ISO 8601:  2026-03-01T09:14:22Z
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"),
    # Bracketed: [2026-03-01 09:14:22]
    re.compile(r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]"),
]


def extract_timestamp(text: str) -> Optional[datetime]:
    """Try to parse a datetime from the first line of a log entry."""
    first_line = text.split("\n")[0]
    for pat in _TS_PATTERNS:
        m = pat.search(first_line)
        if m:
            raw = m.group(1).replace("T", " ")
            try:
                return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
    return None


def detect_severity(text: str) -> str:
    """Detect the log severity from chunk text. Returns best match."""
    first_line = text.split("\n")[0]
    for level, pat in _SEVERITY_PATTERNS.items():
        if pat.search(first_line):
            return level
    return "unknown"


# ── Query intent detection ─────────────────────────────────────────────────────

_BROAD_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"any\s+error",
        r"summarize|summarise|summary",
        r"what\s+(went\s+wrong|happened|errors?\s+are)",
        r"overview|report",
        r"list\s+(all|the)\s+(error|issue|problem)",
        r"are\s+there\s+(error|issue|problem|warning)",
    ]
]


def is_broad_query(question: str) -> bool:
    """True if the question asks for a broad summary rather than a specific match."""
    return any(p.search(question) for p in _BROAD_PATTERNS)


# ── Result model ───────────────────────────────────────────────────────────────

@dataclass
class SeverityDistribution:
    fatal: int = 0
    critical: int = 0
    error: int = 0
    warning: int = 0
    info: int = 0
    debug: int = 0
    unknown: int = 0

    def total_high_severity(self) -> int:
        return self.fatal + self.critical + self.error

    def to_dict(self) -> dict:
        return {
            "fatal": self.fatal, "critical": self.critical, "error": self.error,
            "warning": self.warning, "info": self.info, "debug": self.debug,
            "unknown": self.unknown,
        }


@dataclass
class ConfidenceResult:
    """Structured, explainable confidence output."""
    confidence_level: str        # "high" | "medium" | "low"
    confidence_score: float      # 0.0 – 1.0
    reasoning: str               # Human-readable explanation
    severity_distribution: SeverityDistribution = field(default_factory=SeverityDistribution)
    factors: dict = field(default_factory=dict)  # Raw factor scores for debug

    def to_dict(self) -> dict:
        return {
            "confidence_level": self.confidence_level,
            "confidence_score": round(self.confidence_score, 4),
            "reasoning": self.reasoning,
            "severity_distribution": self.severity_distribution.to_dict(),
            "factors": self.factors,
        }


# ── Main scoring function ─────────────────────────────────────────────────────

def calculate_confidence(
    retrieved_chunks: list,
    query_type: str = "auto",
    question: str = "",
) -> ConfidenceResult:
    """
    Multi-factor confidence scoring.

    Args:
        retrieved_chunks: List of RankedResult (from ranker.py) with
                          .text, .confidence, .source_type fields.
        query_type:       "broad" | "specific" | "auto" (auto-detect from question).
        question:         Original user question (used for auto query_type detection).

    Returns:
        ConfidenceResult with deterministic, explainable score.
    """
    if not retrieved_chunks:
        return ConfidenceResult(
            confidence_level="low",
            confidence_score=0.0,
            reasoning="No chunks retrieved — nothing to score.",
        )

    # ── Auto-detect query type ────────────────────────────────────────────────
    if query_type == "auto" and question:
        query_type = "broad" if is_broad_query(question) else "specific"

    # ── Factor 1: Retrieval similarity (0–1) ──────────────────────────────────
    top_similarity = retrieved_chunks[0].confidence
    avg_similarity = sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks)
    similarity_factor = 0.6 * top_similarity + 0.4 * avg_similarity

    # ── Factor 2: Volume — number of relevant chunks found ────────────────────
    n = len(retrieved_chunks)
    # Diminishing returns: 1 chunk = 0.3, 3 = 0.65, 5+ = 0.85, 8+ = 1.0
    if n >= 8:
        volume_factor = 1.0
    elif n >= 5:
        volume_factor = 0.85
    elif n >= 3:
        volume_factor = 0.65
    elif n >= 1:
        volume_factor = 0.3
    else:
        volume_factor = 0.0

    # ── Factor 3: Severity match ──────────────────────────────────────────────
    sev_dist = SeverityDistribution()
    severity_scores: List[float] = []

    for chunk in retrieved_chunks:
        sev = detect_severity(chunk.text)
        setattr(sev_dist, sev, getattr(sev_dist, sev) + 1)
        severity_scores.append(_SEVERITY_WEIGHT.get(sev, 0.15))

    severity_factor = max(severity_scores) if severity_scores else 0.0

    # ── Factor 4: Recency weighting ───────────────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    recency_scores: List[float] = []

    for chunk in retrieved_chunks:
        ts = extract_timestamp(chunk.text)
        if ts:
            age_hours = (now_utc - ts).total_seconds() / 3600
            # < 1h → 1.0, < 24h → 0.8, < 7d → 0.5, older → 0.2
            if age_hours < 1:
                recency_scores.append(1.0)
            elif age_hours < 24:
                recency_scores.append(0.8)
            elif age_hours < 168:  # 7 days
                recency_scores.append(0.5)
            else:
                recency_scores.append(0.2)

    recency_factor = (sum(recency_scores) / len(recency_scores)) if recency_scores else 0.5

    # ── Factor 5: Pattern consistency ─────────────────────────────────────────
    # If multiple chunks share similar error types, confidence increases
    high_sev_count = sev_dist.total_high_severity()
    if high_sev_count >= 3:
        pattern_factor = 1.0
    elif high_sev_count >= 2:
        pattern_factor = 0.75
    elif high_sev_count >= 1:
        pattern_factor = 0.5
    else:
        pattern_factor = 0.2

    # ── Composite score ───────────────────────────────────────────────────────
    # Weights vary by query type
    if query_type == "broad":
        # Broad queries care more about volume and patterns than exact similarity
        weights = {
            "similarity": 0.20,
            "volume":     0.25,
            "severity":   0.25,
            "recency":    0.15,
            "pattern":    0.15,
        }
    else:
        # Specific queries weight similarity highest
        weights = {
            "similarity": 0.40,
            "volume":     0.15,
            "severity":   0.20,
            "recency":    0.10,
            "pattern":    0.15,
        }

    composite = (
        weights["similarity"] * similarity_factor
        + weights["volume"]   * volume_factor
        + weights["severity"] * severity_factor
        + weights["recency"]  * recency_factor
        + weights["pattern"]  * pattern_factor
    )
    composite = round(max(0.0, min(1.0, composite)), 4)

    # ── Level thresholds ──────────────────────────────────────────────────────
    if composite >= 0.65:
        level = "high"
    elif composite >= 0.40:
        level = "medium"
    else:
        level = "low"

    # ── Build reasoning ──────────────────────────────────────────────────────
    parts = []
    parts.append(f"{n} chunk(s) retrieved")

    if high_sev_count:
        parts.append(f"{high_sev_count} high-severity match(es) (error/critical/fatal)")
    if sev_dist.warning:
        parts.append(f"{sev_dist.warning} warning(s)")

    if recency_scores:
        recent_count = sum(1 for s in recency_scores if s >= 0.8)
        if recent_count:
            parts.append(f"{recent_count} recent entry/entries (<24h)")

    if pattern_factor >= 0.75:
        parts.append("consistent error pattern detected")

    parts.append(f"top similarity {top_similarity:.0%}, avg {avg_similarity:.0%}")
    parts.append(f"query type: {query_type}")

    reasoning = "; ".join(parts) + f". Composite score: {composite:.2f} → {level}."

    factors = {
        "similarity": round(similarity_factor, 4),
        "volume":     round(volume_factor, 4),
        "severity":   round(severity_factor, 4),
        "recency":    round(recency_factor, 4),
        "pattern":    round(pattern_factor, 4),
        "weights":    weights,
    }

    return ConfidenceResult(
        confidence_level=level,
        confidence_score=composite,
        reasoning=reasoning,
        severity_distribution=sev_dist,
        factors=factors,
    )
