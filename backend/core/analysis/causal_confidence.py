"""
core/analysis/causal_confidence.py
─────────────────────────────────────────────────────────────────────────────
Root-cause causal confidence scoring for bakup.ai.

Computes a 0–100 confidence score for cross-analysis root-cause claims
based on five weighted factors:

  1. Log Frequency Concentration  (0–1)
     How concentrated are errors in a dominant pattern vs scattered?

  2. Stack Trace Consistency      (0–1)
     Do errors share consistent stack traces pointing to the same code?

  3. Code Reference Match Strength (0–1)
     How strongly do log errors map to actual indexed code chunks?

  4. Dependency Graph Proximity    (0–1)
     Are the failing code files close in the dependency graph?

  5. Recency Spike Correlation     (0–1)
     Is there a recent spike that correlates with the dominant error?

Formula:
    confidence_score = Σ(factor_i × weight_i) × 100

Returns a CausalConfidenceResult consumed by the evidence ranker,
debug output, and LLM prompt builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.analysis.error_clustering import ErrorClusterReport
from core.analysis.trend_detector import TrendDetectionReport


# ── Weight configuration ───────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "frequency":    0.25,   # Log frequency concentration
    "stack_trace":  0.20,   # Stack trace consistency
    "code_match":   0.25,   # Code reference match strength
    "dependency":   0.10,   # Dependency graph proximity
    "trend":        0.20,   # Recency spike correlation
}


# ── Result model ───────────────────────────────────────────────────────────────

@dataclass
class CausalConfidenceResult:
    """
    Structured causal confidence score for root-cause analysis.

    score is 0–100 where:
      80–100 = High   — strong evidence chain, clear root cause
      50–79  = Medium — partial evidence, probable root cause
      0–49   = Low    — insufficient evidence, speculative
    """
    score: int                    # 0–100
    level: str                    # "high" | "medium" | "low"
    explanation: str              # Human-readable reasoning
    factors: Dict[str, float] = field(default_factory=dict)  # factor name → 0.0–1.0
    weights: Dict[str, float] = field(default_factory=dict)
    dominant_cluster_id: Optional[int] = None
    dominant_error: str = ""

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "level": self.level,
            "explanation": self.explanation,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "weights": self.weights,
            "dominant_cluster_id": self.dominant_cluster_id,
            "dominant_error": self.dominant_error,
        }


# ── Factor calculators ────────────────────────────────────────────────────────

def _frequency_concentration(cluster_report: ErrorClusterReport) -> float:
    """
    Factor 1: Log frequency concentration.

    High when a single error pattern dominates; low when errors are scattered.
    Computed as: dominant_count / total_clustered_count
    """
    if not cluster_report.clusters:
        return 0.0

    dom = cluster_report.dominant_cluster
    if dom is None:
        return 0.0

    total_clustered = sum(c.count for c in cluster_report.clusters)
    if total_clustered == 0:
        return 0.0

    concentration = dom.count / total_clustered

    # Boost if dominant cluster has high count
    if dom.count >= 5:
        concentration = min(1.0, concentration + 0.1)

    return min(1.0, concentration)


def _stack_trace_consistency(cluster_report: ErrorClusterReport) -> float:
    """
    Factor 2: Stack trace consistency.

    High when clusters share common stack frames, indicating the errors
    converge on the same code path.
    """
    if not cluster_report.clusters:
        return 0.0

    dom = cluster_report.dominant_cluster
    if dom is None:
        return 0.0

    # Count clusters with shared stack frames
    clusters_with_frames = sum(1 for c in cluster_report.clusters if c.stack_frames)
    clusters_with_functions = sum(1 for c in cluster_report.clusters if c.related_functions)

    total = cluster_report.cluster_count
    if total == 0:
        return 0.0

    # Score based on:
    # - % of clusters with stack frames
    # - Number of common frames in dominant cluster
    frame_ratio = clusters_with_frames / total
    func_ratio = clusters_with_functions / total

    frame_count_score = min(1.0, len(dom.stack_frames) / 3) if dom.stack_frames else 0.0

    return min(1.0, 0.4 * frame_ratio + 0.3 * func_ratio + 0.3 * frame_count_score)


def _code_reference_match(
    cluster_report: ErrorClusterReport,
    code_chunk_count: int,
    reference_count: int,
    cross_analysis_available: bool,
) -> float:
    """
    Factor 3: Code reference match strength.

    High when log errors have been successfully linked to actual source code.
    """
    score = 0.0

    # Cross-analysis completed successfully
    if cross_analysis_available:
        score += 0.4

    # References extracted from logs
    if reference_count > 0:
        ref_score = min(1.0, reference_count / 5)
        score += 0.3 * ref_score

    # Code chunks found matching references
    if code_chunk_count > 0:
        code_score = min(1.0, code_chunk_count / 5)
        score += 0.3 * code_score

    # Bonus: dominant cluster has related files that appear in code evidence
    dom = cluster_report.dominant_cluster
    if dom and dom.related_files:
        score = min(1.0, score + 0.1)

    return min(1.0, score)


def _dependency_proximity(
    dependency_count: int,
    code_chunk_count: int,
) -> float:
    """
    Factor 4: Dependency graph proximity.

    High when the failing code files are connected in the dependency graph
    (indicating a clear propagation path).
    """
    if dependency_count == 0:
        return 0.1  # baseline — no deps found

    if code_chunk_count == 0:
        return 0.2

    # Ratio of deps to code chunks: more deps = more connected
    ratio = min(1.0, dependency_count / max(1, code_chunk_count))

    # Scale: 0 deps → 0.1, some deps → 0.5, many deps → 1.0
    return min(1.0, 0.1 + 0.9 * ratio)


def _recency_spike_correlation(
    trend_report: Optional[TrendDetectionReport],
    cluster_report: ErrorClusterReport,
) -> float:
    """
    Factor 5: Recency spike correlation.

    High when the dominant error cluster shows a recent spike, indicating
    a live / active issue. Low when errors are old or declining.
    """
    if trend_report is None or not trend_report.cluster_trends:
        return 0.3  # neutral baseline

    dom = cluster_report.dominant_cluster
    if dom is None:
        return 0.3

    # Find trend for dominant cluster
    dom_trend = None
    for ct in trend_report.cluster_trends:
        if ct.cluster_id == dom.cluster_id:
            dom_trend = ct
            break

    if dom_trend is None:
        return 0.3

    # Score based on trend label
    label_scores = {
        "spike": 1.0,
        "new": 0.9,
        "regression": 0.8,
        "stable": 0.5,
        "declining": 0.3,
        "unknown": 0.3,
    }
    base = label_scores.get(dom_trend.trend_label, 0.3)

    # Boost for recent activity
    if dom_trend.last_1h > 0:
        base = min(1.0, base + 0.1)

    # Boost if spike is detected anywhere
    if trend_report.spike_count > 0:
        base = min(1.0, base + 0.1)

    return min(1.0, base)


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_causal_confidence(
    cluster_report: ErrorClusterReport,
    trend_report: Optional[TrendDetectionReport] = None,
    code_chunk_count: int = 0,
    reference_count: int = 0,
    dependency_count: int = 0,
    cross_analysis_available: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> CausalConfidenceResult:
    """
    Compute a causal confidence score (0–100) for root-cause analysis.

    Combines five weighted factors:
      1. frequency    — error pattern concentration
      2. stack_trace  — stack trace consistency
      3. code_match   — log-to-code reference strength
      4. dependency   — dependency graph proximity
      5. trend        — recency and spike correlation

    Args:
        cluster_report:          Output of cluster_error_patterns()
        trend_report:            Output of detect_trends() (optional)
        code_chunk_count:        Number of code chunks in evidence
        reference_count:         Number of code references extracted from logs
        dependency_count:        Number of dependency links found
        cross_analysis_available: Whether cross-analysis linked logs to code
        weights:                 Custom weight overrides (optional)

    Returns:
        CausalConfidenceResult with score, level, explanation, and factors.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Normalise weights to sum to 1.0
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    # Calculate individual factors
    f_freq = _frequency_concentration(cluster_report)
    f_stack = _stack_trace_consistency(cluster_report)
    f_code = _code_reference_match(
        cluster_report, code_chunk_count, reference_count, cross_analysis_available
    )
    f_dep = _dependency_proximity(dependency_count, code_chunk_count)
    f_trend = _recency_spike_correlation(trend_report, cluster_report)

    factors = {
        "frequency": f_freq,
        "stack_trace": f_stack,
        "code_match": f_code,
        "dependency": f_dep,
        "trend": f_trend,
    }

    # Weighted composite
    composite = (
        w.get("frequency", 0) * f_freq
        + w.get("stack_trace", 0) * f_stack
        + w.get("code_match", 0) * f_code
        + w.get("dependency", 0) * f_dep
        + w.get("trend", 0) * f_trend
    )

    # Scale to 0–100
    score = int(round(max(0, min(100, composite * 100))))

    # Level thresholds
    if score >= 80:
        level = "high"
    elif score >= 50:
        level = "medium"
    else:
        level = "low"

    # Dominant cluster info
    dom = cluster_report.dominant_cluster
    dom_id = dom.cluster_id if dom else None
    dom_error = dom.error_signature if dom else ""

    # Build explanation
    explanation = _build_explanation(factors, w, score, level, cluster_report, trend_report)

    return CausalConfidenceResult(
        score=score,
        level=level,
        explanation=explanation,
        factors=factors,
        weights=w,
        dominant_cluster_id=dom_id,
        dominant_error=dom_error,
    )


def _build_explanation(
    factors: Dict[str, float],
    weights: Dict[str, float],
    score: int,
    level: str,
    cluster_report: ErrorClusterReport,
    trend_report: Optional[TrendDetectionReport],
) -> str:
    """Build a human-readable explanation of the confidence score."""
    parts = [f"Causal Confidence: {score}/100 ({level})"]

    # Factor breakdown
    factor_labels = {
        "frequency": "Error concentration",
        "stack_trace": "Stack trace consistency",
        "code_match": "Code reference match",
        "dependency": "Dependency proximity",
        "trend": "Recency/spike correlation",
    }

    for key, label in factor_labels.items():
        val = factors.get(key, 0)
        wt = weights.get(key, 0)
        contribution = round(val * wt * 100, 1)
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        parts.append(f"  {bar} {label}: {val:.0%} (contributes {contribution:.0f} pts)")

    # Cluster summary
    dom = cluster_report.dominant_cluster
    if dom:
        parts.append(f"\nDominant failure: \"{dom.error_signature}\" "
                     f"({dom.count} occurrences, {dom.severity})")

    # Trend alerts
    if trend_report and trend_report.has_alerts:
        alerts = []
        if trend_report.spike_count:
            alerts.append(f"{trend_report.spike_count} spike(s)")
        if trend_report.regression_count:
            alerts.append(f"{trend_report.regression_count} regression(s)")
        if trend_report.new_error_count:
            alerts.append(f"{trend_report.new_error_count} new error(s)")
        parts.append(f"Active alerts: {', '.join(alerts)}")

    return "\n".join(parts)
