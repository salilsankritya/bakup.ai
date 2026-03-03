"""
core/analysis/clusters.py
─────────────────────────────────────────────────────────────────────────────
Timeline clustering for bakup.ai.

Groups retrieved log chunks into incident clusters based on:
  1. Temporal proximity  (entries within CLUSTER_GAP_MINUTES of each other)
  2. Shared keywords     (common error tokens / exception names)
  3. Shared stack traces  (overlapping traceback frames)

Each cluster represents a cohesive incident window that the LLM can reason
about as a unit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from core.analysis.confidence import detect_severity, extract_timestamp

# ── Configuration ──────────────────────────────────────────────────────────────

CLUSTER_GAP_MINUTES = 5        # Max gap between events in the same time cluster
MIN_KEYWORD_OVERLAP = 2         # Minimum shared tokens to merge on keywords
MIN_CLUSTER_SIZE = 1            # Keep clusters with at least this many entries


# ── Keyword extraction ─────────────────────────────────────────────────────────

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "not", "with", "from", "by", "as",
    "it", "that", "this", "be", "has", "have", "had", "do", "does", "did",
    "log", "info", "debug", "error", "warn", "warning", "fatal", "critical",
    "null", "none", "true", "false",
}

_TOKEN_PAT = re.compile(r"[A-Za-z_][\w.]{2,}")

_STACK_FRAME_PAT = re.compile(
    r"(?:at\s+[\w$.]+\(|File\s+\"[^\"]+\",\s+line\s+\d+|"
    r"in\s+[\w.]+|[\w.]+\.(?:py|java|js|ts|rb|go):\d+)",
    re.IGNORECASE,
)


def _extract_keywords(text: str) -> Set[str]:
    """Extract meaningful keyword tokens from a log chunk."""
    tokens = set()
    for m in _TOKEN_PAT.findall(text):
        tok = m.lower()
        if tok not in _STOPWORDS and len(tok) >= 3:
            tokens.add(tok)
    return tokens


def _extract_stack_frames(text: str) -> Set[str]:
    """Extract normalised stack-trace frame strings."""
    frames = set()
    for m in _STACK_FRAME_PAT.finditer(text):
        frames.add(m.group(0).strip().lower())
    return frames


# ── Result models ──────────────────────────────────────────────────────────────

@dataclass
class LogCluster:
    """A group of related log entries forming one incident cluster."""
    cluster_id: int
    label: str                   # Human-readable cluster name
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    entries: List[dict] = field(default_factory=list)   # [{text, severity, timestamp}]
    dominant_error: Optional[str] = None
    severity_counts: Dict[str, int] = field(default_factory=dict)
    shared_keywords: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.entries)

    def describe(self) -> str:
        time_str = ""
        if self.start_time and self.end_time:
            if self.start_time == self.end_time:
                time_str = f" at {self.start_time}"
            else:
                time_str = f" from {self.start_time} to {self.end_time}"
        elif self.start_time:
            time_str = f" at {self.start_time}"

        dom = f', dominant: "{self.dominant_error}"' if self.dominant_error else ""
        return f"[{self.label}] {self.count} entries{time_str}{dom}"

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "entry_count": self.count,
            "dominant_error": self.dominant_error,
            "severity_counts": dict(self.severity_counts),
            "shared_keywords": self.shared_keywords[:10],
        }


@dataclass
class ClusterReport:
    """Aggregated clustering result."""
    clusters: List[LogCluster] = field(default_factory=list)
    unclustered_count: int = 0
    total_entries: int = 0

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    def summary_text(self) -> str:
        """Human-readable summary for LLM context."""
        if not self.clusters:
            return f"No incident clusters detected across {self.total_entries} log entries."

        lines = [
            f"Detected {self.cluster_count} incident cluster(s) "
            f"from {self.total_entries} log entries:"
        ]
        for cl in self.clusters:
            lines.append(f"  • {cl.describe()}")

        if self.unclustered_count:
            lines.append(f"  ({self.unclustered_count} entries did not cluster)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "cluster_count": self.cluster_count,
            "clusters": [c.to_dict() for c in self.clusters],
            "unclustered_count": self.unclustered_count,
            "total_entries": self.total_entries,
        }


# ── Clustering algorithm ──────────────────────────────────────────────────────

def cluster_log_events(chunks: list) -> ClusterReport:
    """
    Cluster retrieved log chunks into incident groups.

    Uses a two-pass approach:
      Pass 1 — Temporal clustering (entries within CLUSTER_GAP_MINUTES)
      Pass 2 — Merge temporally-separate clusters that share keywords/stacks

    Args:
        chunks: List of RankedResult (or any object with .text attribute).

    Returns:
        ClusterReport.
    """
    report = ClusterReport(total_entries=len(chunks))

    if not chunks:
        return report

    # ── Extract metadata for each chunk ──────────────────────────────────────
    entries = []
    for chunk in chunks:
        ts = extract_timestamp(chunk.text)
        sev = detect_severity(chunk.text)
        keywords = _extract_keywords(chunk.text)
        stack_frames = _extract_stack_frames(chunk.text)
        entries.append({
            "text": chunk.text,
            "timestamp": ts,
            "severity": sev,
            "keywords": keywords,
            "stack_frames": stack_frames,
        })

    # ── Pass 1: Temporal clustering ──────────────────────────────────────────
    # Separate entries with and without timestamps
    timed = [(i, e) for i, e in enumerate(entries) if e["timestamp"]]
    untimed = [(i, e) for i, e in enumerate(entries) if not e["timestamp"]]

    # Sort by timestamp
    timed.sort(key=lambda x: x[1]["timestamp"])

    raw_clusters: List[List[int]] = []  # list of entry indices

    if timed:
        current_cluster = [timed[0][0]]
        prev_ts = timed[0][1]["timestamp"]

        for idx, entry in timed[1:]:
            gap = (entry["timestamp"] - prev_ts).total_seconds() / 60
            if gap <= CLUSTER_GAP_MINUTES:
                current_cluster.append(idx)
            else:
                raw_clusters.append(current_cluster)
                current_cluster = [idx]
            prev_ts = entry["timestamp"]

        raw_clusters.append(current_cluster)

    # ── Pass 2: Keyword/stack merge ──────────────────────────────────────────
    # If two temporal clusters share significant keywords or stack frames,
    # merge them.
    merged = True
    while merged:
        merged = False
        new_clusters = []
        skip = set()

        for i in range(len(raw_clusters)):
            if i in skip:
                continue

            current = raw_clusters[i]
            current_kw = set()
            current_sf = set()
            for idx in current:
                current_kw |= entries[idx]["keywords"]
                current_sf |= entries[idx]["stack_frames"]

            for j in range(i + 1, len(raw_clusters)):
                if j in skip:
                    continue

                other_kw = set()
                other_sf = set()
                for idx in raw_clusters[j]:
                    other_kw |= entries[idx]["keywords"]
                    other_sf |= entries[idx]["stack_frames"]

                kw_overlap = len(current_kw & other_kw)
                sf_overlap = len(current_sf & other_sf)

                if kw_overlap >= MIN_KEYWORD_OVERLAP or sf_overlap >= 1:
                    current = current + raw_clusters[j]
                    current_kw |= other_kw
                    current_sf |= other_sf
                    skip.add(j)
                    merged = True

            new_clusters.append(current)

        raw_clusters = new_clusters

    # ── Assign untimed entries to nearest matching cluster ────────────────────
    assigned_untimed = set()
    for ui, ue in untimed:
        best_match = -1
        best_overlap = 0
        for ci, cluster_indices in enumerate(raw_clusters):
            cluster_kw = set()
            for idx in cluster_indices:
                cluster_kw |= entries[idx]["keywords"]
            overlap = len(ue["keywords"] & cluster_kw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ci
        if best_match >= 0 and best_overlap >= MIN_KEYWORD_OVERLAP:
            raw_clusters[best_match].append(ui)
            assigned_untimed.add(ui)

    unclustered = [ui for ui, _ in untimed if ui not in assigned_untimed]

    # ── Build LogCluster objects ─────────────────────────────────────────────
    cluster_id = 0
    for cluster_indices in raw_clusters:
        if len(cluster_indices) < MIN_CLUSTER_SIZE:
            unclustered.extend(cluster_indices)
            continue

        cluster_id += 1
        cl_entries = [entries[i] for i in cluster_indices]

        # Time window
        valid_ts = sorted(e["timestamp"] for e in cl_entries if e["timestamp"])
        start_str = valid_ts[0].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None
        end_str = valid_ts[-1].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None

        # Severity distribution
        sev_counts: Counter = Counter()
        for e in cl_entries:
            sev_counts[e["severity"]] += 1

        # Dominant error
        from core.analysis.trends import _extract_error_signature
        error_sigs: Counter = Counter()
        for e in cl_entries:
            sig = _extract_error_signature(e["text"])
            if sig:
                error_sigs[sig] += 1
        dominant = error_sigs.most_common(1)[0][0] if error_sigs else None

        # Shared keywords across entries
        if len(cl_entries) >= 2:
            all_kw: Counter = Counter()
            for e in cl_entries:
                for k in e["keywords"]:
                    all_kw[k] += 1
            shared = [k for k, c in all_kw.most_common(10) if c >= 2]
        else:
            shared = []

        # Label
        if dominant:
            label = dominant[:60]
        elif sev_counts.get("error", 0) > 0:
            label = f"Error cluster at {start_str or 'unknown time'}"
        else:
            top_sev = sev_counts.most_common(1)[0][0] if sev_counts else "log"
            label = f"{top_sev.title()} cluster at {start_str or 'unknown time'}"

        lc = LogCluster(
            cluster_id=cluster_id,
            label=label,
            start_time=start_str,
            end_time=end_str,
            entries=[
                {"text": e["text"][:200], "severity": e["severity"],
                 "timestamp": e["timestamp"].isoformat() if e["timestamp"] else None}
                for e in cl_entries
            ],
            dominant_error=dominant,
            severity_counts=dict(sev_counts),
            shared_keywords=shared,
        )
        report.clusters.append(lc)

    report.unclustered_count = len(unclustered)
    return report
