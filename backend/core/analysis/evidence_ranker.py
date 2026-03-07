"""
core/analysis/evidence_ranker.py
─────────────────────────────────────────────────────────────────────────────
Evidence ranking and filtering for bakup.ai root-cause analysis.

Before sending evidence to the LLM, this module:
  1. Ranks error clusters by severity and impact
  2. Ranks individual evidence chunks by relevance to the dominant cluster
  3. Limits to top N clusters and chunks to avoid context overflow
  4. Produces a StructuredReasoningInput for the LLM prompt

This sits between the agent executor and the LLM call, refining the
raw evidence into a focused, high-signal package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.analysis.error_clustering import ErrorCluster, ErrorClusterReport
from core.analysis.trend_detector import TrendDetectionReport
from core.analysis.causal_confidence import CausalConfidenceResult


# ── Configuration ──────────────────────────────────────────────────────────────

MAX_CLUSTERS = 5          # Top N clusters to include in LLM context
MAX_LOG_CHUNKS = 8        # Max log evidence chunks
MAX_CODE_CHUNKS = 8       # Max code evidence chunks


# ── Severity scoring ──────────────────────────────────────────────────────────

_SEVERITY_SCORES = {
    "fatal": 10,
    "critical": 8,
    "error": 6,
    "warning": 3,
    "info": 1,
    "debug": 0,
    "unknown": 1,
}

_TREND_SCORES = {
    "spike": 5,
    "new": 4,
    "regression": 4,
    "stable": 1,
    "declining": 0,
    "unknown": 1,
}


# ── Structured reasoning input ────────────────────────────────────────────────

@dataclass
class StructuredReasoningInput:
    """
    Refined, ranked evidence package for the LLM prompt.

    Contains:
      - dominant_cluster:   The primary failure pattern
      - frequency_stats:    Count summaries across clusters
      - time_trend:         Spike/regression/new detection results
      - related_code:       Code chunk summaries linked to errors
      - dependency_chain:   Files in the dependency path
      - confidence_score:   Causal confidence (0–100)

    This is serialised into the LLM prompt as structured context.
    """
    dominant_cluster: Optional[Dict] = None
    top_clusters: List[Dict] = field(default_factory=list)
    frequency_stats: Dict = field(default_factory=dict)
    time_trend: Dict = field(default_factory=dict)
    related_code: List[Dict] = field(default_factory=list)
    dependency_chain: List[str] = field(default_factory=list)
    confidence_score: int = 0
    confidence_level: str = "low"
    confidence_explanation: str = ""

    def to_prompt_block(self) -> str:
        """Render into a structured text block for the LLM prompt."""
        sections = []

        # Confidence header
        sections.append(
            f"## Causal Confidence Score: {self.confidence_score}/100 "
            f"({self.confidence_level})"
        )

        # Dominant cluster
        if self.dominant_cluster:
            dc = self.dominant_cluster
            sections.append(
                f"\n## Dominant Failure Pattern\n"
                f"- **Error**: {dc.get('error_signature', 'unknown')}\n"
                f"- **Exception type**: {dc.get('exception_type', 'N/A')}\n"
                f"- **Occurrences**: {dc.get('count', 0)}\n"
                f"- **Severity**: {dc.get('severity', 'unknown')}\n"
                f"- **First seen**: {dc.get('first_seen', 'N/A')}\n"
                f"- **Last seen**: {dc.get('last_seen', 'N/A')}\n"
                f"- **Related files**: {', '.join(dc.get('related_files', []))}\n"
                f"- **Related functions**: {', '.join(dc.get('related_functions', []))}\n"
                f"- **Trend**: {dc.get('trend_label', 'unknown')}"
            )

        # Frequency stats
        if self.frequency_stats:
            fs = self.frequency_stats
            sections.append(
                f"\n## Error Frequency Statistics\n"
                f"- Total clusters: {fs.get('cluster_count', 0)}\n"
                f"- Total error entries: {fs.get('total_entries', 0)}\n"
                f"- Dominant pattern share: {fs.get('dominant_share_pct', 0):.0f}%"
            )

        # Time trend
        if self.time_trend:
            tt = self.time_trend
            parts = ["\n## Time-Based Trends"]
            if tt.get("spike_count", 0):
                parts.append(f"- ⚠ {tt['spike_count']} spike(s) detected")
            if tt.get("regression_count", 0):
                parts.append(f"- 🔙 {tt['regression_count']} regression(s)")
            if tt.get("new_error_count", 0):
                parts.append(f"- 🆕 {tt['new_error_count']} newly introduced error(s)")

            for ct in tt.get("cluster_trends", [])[:MAX_CLUSTERS]:
                parts.append(
                    f"  • Cluster {ct['cluster_id']}: "
                    f"total={ct['total']}, 1h={ct['last_1h']}, 24h={ct['last_24h']}, "
                    f"change={ct['pct_change']:+.0f}%, trend={ct['trend_label']}"
                )
            sections.append("\n".join(parts))

        # Other clusters
        if len(self.top_clusters) > 1:
            parts = ["\n## Other Error Clusters"]
            for cl in self.top_clusters[1:]:
                parts.append(
                    f"- \"{cl.get('error_signature', 'unknown')}\" × {cl.get('count', 0)} "
                    f"(severity: {cl.get('severity', 'unknown')}, "
                    f"trend: {cl.get('trend_label', 'unknown')})"
                )
            sections.append("\n".join(parts))

        # Dependency chain
        if self.dependency_chain:
            deps = "\n".join(f"- {d}" for d in self.dependency_chain[:10])
            sections.append(f"\n## Dependency Chain\n{deps}")

        # Confidence explanation
        if self.confidence_explanation:
            sections.append(f"\n## Confidence Breakdown\n{self.confidence_explanation}")

        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {
            "dominant_cluster": self.dominant_cluster,
            "top_clusters": self.top_clusters,
            "frequency_stats": self.frequency_stats,
            "time_trend": self.time_trend,
            "related_code": self.related_code,
            "dependency_chain": self.dependency_chain,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
        }


# ── Cluster ranking ───────────────────────────────────────────────────────────

def _rank_cluster(cluster: ErrorCluster) -> float:
    """
    Compute a ranking score for a cluster.

    Higher = more important. Combines:
      - Severity weight
      - Occurrence count (log scale)
      - Spike magnitude
      - Reference clarity (has files/functions)
    """
    import math

    severity_score = _SEVERITY_SCORES.get(cluster.severity, 1)
    trend_score = _TREND_SCORES.get(cluster.trend_label, 1)

    count_score = math.log2(max(1, cluster.count))  # diminishing returns

    # Reference clarity: does this cluster point to specific code?
    ref_clarity = 0.0
    if cluster.related_files:
        ref_clarity += min(1.0, len(cluster.related_files) / 3)
    if cluster.related_functions:
        ref_clarity += min(1.0, len(cluster.related_functions) / 3)
    if cluster.stack_frames:
        ref_clarity += min(1.0, len(cluster.stack_frames) / 2)

    return (
        severity_score * 3.0
        + count_score * 2.0
        + trend_score * 2.5
        + ref_clarity * 1.5
    )


# ── Main ranking function ─────────────────────────────────────────────────────

def rank_evidence(
    cluster_report: ErrorClusterReport,
    trend_report: Optional[TrendDetectionReport],
    confidence_result: CausalConfidenceResult,
    code_chunks: list = None,
    dependencies: list = None,
) -> StructuredReasoningInput:
    """
    Rank and filter evidence into a StructuredReasoningInput.

    Steps:
      1. Rank clusters by severity, count, trend, and reference clarity
      2. Limit to top N clusters
      3. Build frequency stats (total, dominant share)
      4. Build structured reasoning input for the LLM

    Args:
        cluster_report: Output of cluster_error_patterns()
        trend_report:   Output of detect_trends()
        confidence_result: Output of compute_causal_confidence()
        code_chunks:  Evidence code chunks (for related_code summary)
        dependencies: Dependency chain entries

    Returns:
        StructuredReasoningInput ready for LLM prompt injection.
    """
    if code_chunks is None:
        code_chunks = []
    if dependencies is None:
        dependencies = []

    result = StructuredReasoningInput(
        confidence_score=confidence_result.score,
        confidence_level=confidence_result.level,
        confidence_explanation=confidence_result.explanation,
    )

    if not cluster_report.clusters:
        return result

    # Rank clusters
    ranked = sorted(cluster_report.clusters, key=_rank_cluster, reverse=True)
    top_clusters = ranked[:MAX_CLUSTERS]

    # Build cluster dicts
    cluster_dicts = [c.to_dict() for c in top_clusters]
    result.top_clusters = cluster_dicts
    result.dominant_cluster = cluster_dicts[0] if cluster_dicts else None

    # Frequency stats
    total_clustered = sum(c.count for c in cluster_report.clusters)
    dom = cluster_report.dominant_cluster
    dom_share = (dom.count / total_clustered * 100) if dom and total_clustered > 0 else 0

    result.frequency_stats = {
        "cluster_count": cluster_report.cluster_count,
        "total_entries": cluster_report.total_entries,
        "total_clustered": total_clustered,
        "unclustered": cluster_report.unclustered_count,
        "dominant_share_pct": round(dom_share, 1),
    }

    # Time trend
    if trend_report:
        result.time_trend = trend_report.to_dict()

    # Related code (summaries of code chunks)
    for chunk in code_chunks[:MAX_CODE_CHUNKS]:
        result.related_code.append({
            "file": getattr(chunk, "source_file", ""),
            "function": getattr(chunk, "function_name", ""),
            "class": getattr(chunk, "class_name", ""),
            "lines": f"{getattr(chunk, 'line_start', 0)}–{getattr(chunk, 'line_end', 0)}",
            "confidence": getattr(chunk, "confidence", 0),
        })

    # Dependency chain
    result.dependency_chain = list(dependencies)[:15]

    return result
