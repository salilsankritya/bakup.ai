"""
core/analysis/trend_detector.py
─────────────────────────────────────────────────────────────────────────────
Per-cluster time-window trend detection for bakup.ai.

For each ErrorCluster, calculates:
  - Total occurrences
  - Occurrences in last 1 hour
  - Occurrences in last 24 hours
  - Percentage change vs previous window

Detects:
  - Spikes       (rate ≥ 2× average in recent window)
  - Regressions  (error re-appeared after a quiet period)
  - Newly introduced (first seen within last 24h)
  - Declining    (count dropping relative to previous window)
  - Stable       (no significant change)

Works alongside error_clustering.py — call detect_trends() AFTER
cluster_error_patterns() to enrich clusters with time-based metadata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core.analysis.confidence import extract_timestamp
from core.analysis.error_clustering import ErrorCluster, ErrorClusterReport


# ── Configuration ──────────────────────────────────────────────────────────────

WINDOW_1H = timedelta(hours=1)
WINDOW_24H = timedelta(hours=24)
SPIKE_MULTIPLIER = 2.0       # Rate must exceed this × average to be a spike
QUIET_GAP_HOURS = 6          # Hours of silence before a reappearance = regression


# ── Trend result model ────────────────────────────────────────────────────────

@dataclass
class ClusterTrend:
    """Time-based trend analysis for a single error cluster."""
    cluster_id: int
    total: int
    last_1h: int = 0
    last_24h: int = 0
    pct_change: float = 0.0      # % change: current window vs previous window
    trend_label: str = "stable"  # "spike" | "regression" | "new" | "stable" | "declining"
    spike_detected: bool = False
    is_new: bool = False
    is_regression: bool = False

    def describe(self) -> str:
        parts = [f"total={self.total}, 1h={self.last_1h}, 24h={self.last_24h}"]
        if self.pct_change != 0:
            parts.append(f"change={self.pct_change:+.0f}%")
        parts.append(f"trend={self.trend_label}")
        if self.spike_detected:
            parts.append("⚠ SPIKE")
        if self.is_new:
            parts.append("🆕 NEW")
        if self.is_regression:
            parts.append("🔙 REGRESSION")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "total": self.total,
            "last_1h": self.last_1h,
            "last_24h": self.last_24h,
            "pct_change": round(self.pct_change, 2),
            "trend_label": self.trend_label,
            "spike_detected": self.spike_detected,
            "is_new": self.is_new,
            "is_regression": self.is_regression,
        }


@dataclass
class TrendDetectionReport:
    """Aggregated trend detection across all clusters."""
    cluster_trends: List[ClusterTrend] = field(default_factory=list)
    spike_count: int = 0
    regression_count: int = 0
    new_error_count: int = 0
    analysis_time_utc: str = ""

    @property
    def has_alerts(self) -> bool:
        return self.spike_count > 0 or self.regression_count > 0 or self.new_error_count > 0

    def summary_text(self) -> str:
        """Human-readable summary for LLM prompt."""
        if not self.cluster_trends:
            return "No time-based trends detected."

        lines = ["Time-Based Trend Analysis:"]

        # Alert summary
        alerts = []
        if self.spike_count:
            alerts.append(f"{self.spike_count} spike(s)")
        if self.regression_count:
            alerts.append(f"{self.regression_count} regression(s)")
        if self.new_error_count:
            alerts.append(f"{self.new_error_count} new error(s)")
        if alerts:
            lines.append(f"  ⚠ Alerts: {', '.join(alerts)}")

        # Per-cluster trends
        for ct in sorted(self.cluster_trends, key=lambda t: -t.total):
            lines.append(f"  • Cluster {ct.cluster_id}: {ct.describe()}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "cluster_trends": [t.to_dict() for t in self.cluster_trends],
            "spike_count": self.spike_count,
            "regression_count": self.regression_count,
            "new_error_count": self.new_error_count,
            "analysis_time_utc": self.analysis_time_utc,
            "has_alerts": self.has_alerts,
        }


# ── Per-entry timestamp extraction ────────────────────────────────────────────

def _collect_timestamps(chunks: list) -> Dict[str, List[datetime]]:
    """
    Build a map: error_signature → list of timestamps.
    
    Requires the chunks to have been pre-processed by cluster_error_patterns.
    We re-extract timestamps from raw text matched to cluster signatures.
    """
    from core.analysis.error_clustering import _compute_signature

    sig_timestamps: Dict[str, List[datetime]] = defaultdict(list)

    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        ts = extract_timestamp(text)
        if ts is None:
            continue
        sig = _compute_signature(text)
        # Strip prefix for matching
        clean_sig = sig.split(":", 1)[-1] if ":" in sig else sig
        sig_timestamps[clean_sig].append(ts)

    return sig_timestamps


# ── Main detection function ──────────────────────────────────────────────────

def detect_trends(
    cluster_report: ErrorClusterReport,
    chunks: list,
    reference_time: Optional[datetime] = None,
) -> TrendDetectionReport:
    """
    Detect time-based trends for each error cluster.

    For each cluster:
      1. Count occurrences in last 1h and 24h windows
      2. Compare current window to previous window for % change
      3. Classify as spike / regression / new / stable / declining

    Args:
        cluster_report: Output of cluster_error_patterns()
        chunks:         Original log chunks (for timestamp extraction)
        reference_time: Reference "now" time. Defaults to UTC now.

    Returns:
        TrendDetectionReport with per-cluster trend data.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    report = TrendDetectionReport(
        analysis_time_utc=reference_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    if not cluster_report.clusters:
        return report

    # Build timestamp map from raw chunks
    sig_timestamps = _collect_timestamps(chunks)

    for cluster in cluster_report.clusters:
        sig = cluster.error_signature
        timestamps = sig_timestamps.get(sig, [])

        # If no timestamps found by signature, try to match by exception type
        if not timestamps and cluster.exception_type:
            for s, ts_list in sig_timestamps.items():
                if cluster.exception_type.lower() in s.lower():
                    timestamps.extend(ts_list)
                    break

        trend = _analyse_cluster_trend(cluster, timestamps, reference_time)
        report.cluster_trends.append(trend)

        # Update the cluster object in-place with trend data
        cluster.occurrences_1h = trend.last_1h
        cluster.occurrences_24h = trend.last_24h
        cluster.trend_pct_change = trend.pct_change
        cluster.trend_label = trend.trend_label

    # Aggregate counts
    report.spike_count = sum(1 for t in report.cluster_trends if t.spike_detected)
    report.regression_count = sum(1 for t in report.cluster_trends if t.is_regression)
    report.new_error_count = sum(1 for t in report.cluster_trends if t.is_new)

    return report


def _analyse_cluster_trend(
    cluster: ErrorCluster,
    timestamps: List[datetime],
    reference_time: datetime,
) -> ClusterTrend:
    """Analyse trend for a single cluster."""
    trend = ClusterTrend(
        cluster_id=cluster.cluster_id,
        total=cluster.count,
    )

    if not timestamps:
        # No timestamps available — use count only
        trend.trend_label = "unknown"
        return trend

    timestamps.sort()

    # Window counts
    cutoff_1h = reference_time - WINDOW_1H
    cutoff_24h = reference_time - WINDOW_24H

    trend.last_1h = sum(1 for ts in timestamps if ts >= cutoff_1h)
    trend.last_24h = sum(1 for ts in timestamps if ts >= cutoff_24h)

    # Previous 24h window: -48h to -24h
    cutoff_prev_24h = reference_time - timedelta(hours=48)
    prev_window_count = sum(1 for ts in timestamps
                            if cutoff_prev_24h <= ts < cutoff_24h)

    # Percentage change
    if prev_window_count > 0:
        trend.pct_change = ((trend.last_24h - prev_window_count) / prev_window_count) * 100
    elif trend.last_24h > 0:
        trend.pct_change = 100.0  # Infinite increase → cap at 100%

    # Detect spike: last 1h ≥ SPIKE_MULTIPLIER × average hourly rate
    total_hours = max(1, (timestamps[-1] - timestamps[0]).total_seconds() / 3600)
    avg_hourly = cluster.count / total_hours if total_hours > 0 else 0

    if avg_hourly > 0 and trend.last_1h >= SPIKE_MULTIPLIER * avg_hourly and trend.last_1h >= 2:
        trend.spike_detected = True
        trend.trend_label = "spike"

    # Detect new error: first seen within last 24h
    elif timestamps[0] >= cutoff_24h:
        trend.is_new = True
        trend.trend_label = "new"

    # Detect regression: gap > QUIET_GAP_HOURS then reappearance
    elif len(timestamps) >= 2:
        max_gap = max(
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
            for i in range(len(timestamps) - 1)
        )
        if max_gap >= QUIET_GAP_HOURS and timestamps[-1] >= cutoff_24h:
            trend.is_regression = True
            trend.trend_label = "regression"
        elif trend.pct_change <= -30:
            trend.trend_label = "declining"
        elif trend.pct_change >= 50:
            trend.trend_label = "spike"
            trend.spike_detected = True
        else:
            trend.trend_label = "stable"
    else:
        trend.trend_label = "stable"

    return trend
