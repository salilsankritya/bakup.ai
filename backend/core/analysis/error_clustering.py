"""
core/analysis/error_clustering.py
─────────────────────────────────────────────────────────────────────────────
Error pattern clustering for bakup.ai root-cause analysis.

Groups similar errors by:
  1. Error message similarity (normalised Levenshtein-like fingerprinting)
  2. Stack trace similarity  (shared frame sequences)
  3. File + line reference   (same file:line across entries)
  4. Exception type          (same exception class)

Produces ErrorCluster objects consumed by the causal confidence scorer,
evidence ranker, and LLM prompt.

Different from clusters.py (which groups by temporal proximity).
This module groups by *error pattern identity* regardless of time.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from core.analysis.confidence import detect_severity, extract_timestamp


# ── Normalisation helpers ──────────────────────────────────────────────────────

_EXCEPTION_PAT = re.compile(
    r"((?:[\w.]+)?(?:Error|Exception|Fault|Failure|Panic)(?::\s*.{0,120})?)",
    re.IGNORECASE,
)

_STACK_FRAME_PAT = re.compile(
    r"(?:at\s+[\w$.]+\(|File\s+\"[^\"]+\",\s+line\s+\d+|"
    r"in\s+[\w.]+|[\w.]+\.(?:py|java|js|ts|rb|go):\d+)",
    re.IGNORECASE,
)

_FILE_LINE_PAT = re.compile(
    r"([\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php))\s*[:\s]+(?:line\s+)?(\d+)",
    re.IGNORECASE,
)

_FUNC_PAT = re.compile(
    r"(?:in|function|method|def)\s+['\"]?(\w{2,})['\"]?",
    re.IGNORECASE,
)

_CLASS_METHOD_PAT = re.compile(r"\b([A-Z][a-zA-Z0-9]+)\.([\w]+)\b")

# Strip variable data: hex addresses, UUIDs, timestamps, numbers
_NORMALISE_PATS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), "<uuid>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\dZ]*"), "<ts>"),
    (re.compile(r"\b\d{4,}\b"), "<num>"),
]

_LEVEL_ONLY = {"error", "warning", "warn", "info", "debug", "fatal", "critical"}


def _normalise_error_text(text: str) -> str:
    """Strip variable content to produce a stable error fingerprint."""
    result = text.split("\n")[0][:200]  # first line only
    for pat, repl in _NORMALISE_PATS:
        result = pat.sub(repl, result)
    return result.strip().lower()


def _extract_exception_type(text: str) -> str:
    """Extract the exception class name from text."""
    for m in _EXCEPTION_PAT.finditer(text):
        sig = m.group(1).strip()
        # Remove the message part after ':'
        base = sig.split(":")[0].strip()
        if base.lower() not in _LEVEL_ONLY:
            return base
    return ""


def _extract_stack_frames(text: str) -> List[str]:
    """Extract normalised stack-trace frame strings."""
    return [m.group(0).strip().lower() for m in _STACK_FRAME_PAT.finditer(text)]


def _extract_file_lines(text: str) -> List[Tuple[str, int]]:
    """Extract (file, line) references from text."""
    results = []
    for m in _FILE_LINE_PAT.finditer(text):
        results.append((m.group(1), int(m.group(2))))
    return results


def _extract_functions(text: str) -> Set[str]:
    """Extract function/method names from text."""
    funcs: Set[str] = set()
    for m in _FUNC_PAT.finditer(text):
        funcs.add(m.group(1))
    for m in _CLASS_METHOD_PAT.finditer(text):
        if m.group(1) not in ("System", "Console", "Object", "Array", "String",
                               "Integer", "Boolean", "Math", "Date"):
            funcs.add(m.group(2))
    return funcs


def _compute_signature(text: str) -> str:
    """
    Compute a stable error signature for grouping.

    Priority:
      1. Exception type (e.g. "NullPointerException")
      2. Normalised first-line error message
      3. File:line reference
    """
    exc = _extract_exception_type(text)
    if exc:
        return f"exc:{exc}"

    normalised = _normalise_error_text(text)

    # Remove log-level prefixes for better grouping
    normalised = re.sub(r"^(error|warn(ing)?|fatal|critical)\s*[:\s]*", "", normalised)
    normalised = re.sub(r"^\[?\s*<ts>\s*\]?\s*", "", normalised)

    if normalised:
        return f"msg:{normalised[:100]}"

    file_lines = _extract_file_lines(text)
    if file_lines:
        return f"loc:{file_lines[0][0]}:{file_lines[0][1]}"

    return f"raw:{text[:50].strip().lower()}"


def _signature_similarity(sig1: str, sig2: str) -> float:
    """
    Compute similarity between two error signatures.
    Returns 0.0–1.0 where 1.0 is identical.
    """
    if sig1 == sig2:
        return 1.0

    # Same prefix type
    t1, _, v1 = sig1.partition(":")
    t2, _, v2 = sig2.partition(":")

    if t1 != t2:
        return 0.0

    if t1 == "exc":
        # Exception types: check if one is a subclass name of the other
        if v1 in v2 or v2 in v1:
            return 0.8
        return 0.0

    if t1 == "msg":
        # Normalised message: use token overlap
        tokens1 = set(v1.split())
        tokens2 = set(v2.split())
        if not tokens1 or not tokens2:
            return 0.0
        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)
        return overlap / total if total > 0 else 0.0

    return 0.0


# ── ErrorCluster data model ───────────────────────────────────────────────────

@dataclass
class ErrorCluster:
    """
    A group of similar errors clustered by pattern identity.

    Groups errors by exception type, message similarity, stack trace
    similarity, and file+line reference.
    """
    cluster_id: int
    error_signature: str             # Normalised fingerprint
    exception_type: str = ""         # e.g. "NullPointerException"
    count: int = 0
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    related_files: List[str] = field(default_factory=list)
    related_functions: List[str] = field(default_factory=list)
    severity: str = "unknown"        # Dominant severity within cluster
    sample_messages: List[str] = field(default_factory=list)  # Top 3 unique messages
    stack_frames: List[str] = field(default_factory=list)     # Common frames

    # Time-window stats (populated by trend_detector)
    occurrences_1h: int = 0
    occurrences_24h: int = 0
    trend_pct_change: float = 0.0    # % change vs previous window
    trend_label: str = ""            # "spike" | "regression" | "new" | "stable" | "declining"

    def describe(self) -> str:
        """Human-readable summary of this cluster."""
        parts = [f'"{self.error_signature}" × {self.count}']
        if self.exception_type:
            parts.append(f"type: {self.exception_type}")
        if self.first_seen and self.last_seen:
            if self.first_seen == self.last_seen:
                parts.append(f"at {self.first_seen}")
            else:
                parts.append(f"from {self.first_seen} to {self.last_seen}")
        if self.related_files:
            parts.append(f"files: {', '.join(self.related_files[:3])}")
        if self.trend_label:
            parts.append(f"trend: {self.trend_label}")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "error_signature": self.error_signature,
            "exception_type": self.exception_type,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "related_files": self.related_files,
            "related_functions": self.related_functions,
            "severity": self.severity,
            "sample_messages": self.sample_messages,
            "stack_frames": self.stack_frames[:5],
            "occurrences_1h": self.occurrences_1h,
            "occurrences_24h": self.occurrences_24h,
            "trend_pct_change": round(self.trend_pct_change, 2),
            "trend_label": self.trend_label,
        }


@dataclass
class ErrorClusterReport:
    """Aggregated error pattern clustering result."""
    clusters: List[ErrorCluster] = field(default_factory=list)
    total_entries: int = 0
    unclustered_count: int = 0

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def dominant_cluster(self) -> Optional[ErrorCluster]:
        """The cluster with the highest count."""
        if not self.clusters:
            return None
        return max(self.clusters, key=lambda c: c.count)

    def summary_text(self) -> str:
        """Human-readable summary for LLM context."""
        if not self.clusters:
            return f"No error pattern clusters detected across {self.total_entries} entries."

        lines = [
            f"Detected {self.cluster_count} error pattern cluster(s) "
            f"from {self.total_entries} log entries:"
        ]
        for cl in sorted(self.clusters, key=lambda c: -c.count):
            lines.append(f"  • {cl.describe()}")
        if self.unclustered_count:
            lines.append(f"  ({self.unclustered_count} entries did not match any cluster)")

        # Dominant cluster highlight
        dom = self.dominant_cluster
        if dom and dom.count >= 2:
            lines.append(f"\nDominant failure pattern: \"{dom.error_signature}\" "
                         f"({dom.count} occurrences, severity: {dom.severity})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        dom = self.dominant_cluster
        return {
            "cluster_count": self.cluster_count,
            "total_entries": self.total_entries,
            "unclustered_count": self.unclustered_count,
            "clusters": [c.to_dict() for c in self.clusters],
            "dominant_cluster": dom.to_dict() if dom else None,
        }


# ── Main clustering function ──────────────────────────────────────────────────

_MERGE_THRESHOLD = 0.5  # Minimum signature similarity to merge clusters


def cluster_error_patterns(chunks: list) -> ErrorClusterReport:
    """
    Cluster log chunks by error pattern identity.

    Groups similar errors by:
      1. Exception type (exact match)
      2. Normalised error message (token overlap ≥ MERGE_THRESHOLD)
      3. File + line reference (exact match)
      4. Stack trace overlap (≥ 2 shared frames)

    Args:
        chunks: List of RankedResult or any object with .text and optionally
                .source_file, .severity attributes.

    Returns:
        ErrorClusterReport with clustered error patterns.
    """
    report = ErrorClusterReport(total_entries=len(chunks))

    if not chunks:
        return report

    # ── Phase 1: Extract metadata per entry ──────────────────────────────────
    entries = []
    for chunk in chunks:
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        sev = detect_severity(text)

        # Only cluster error-level entries (error/fatal/critical/warning)
        if sev in ("info", "debug", "unknown"):
            continue

        ts = extract_timestamp(text)
        sig = _compute_signature(text)
        exc_type = _extract_exception_type(text)
        frames = _extract_stack_frames(text)
        file_lines = _extract_file_lines(text)
        funcs = _extract_functions(text)
        source_file = getattr(chunk, "source_file", "") or ""
        file_name = getattr(chunk, "file_name", "") or source_file.split("/")[-1].split("\\")[-1]

        entries.append({
            "text": text,
            "severity": sev,
            "timestamp": ts,
            "signature": sig,
            "exception_type": exc_type,
            "stack_frames": frames,
            "file_lines": file_lines,
            "functions": funcs,
            "source_file": source_file,
            "file_name": file_name,
        })

    if not entries:
        report.unclustered_count = len(chunks)
        return report

    # ── Phase 2: Group by signature ──────────────────────────────────────────
    # First pass: exact signature match
    sig_groups: Dict[str, List[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        sig_groups[entry["signature"]].append(i)

    # Second pass: merge groups with similar signatures
    group_keys = list(sig_groups.keys())
    merged: Set[int] = set()
    final_groups: List[List[int]] = []

    for i, key_i in enumerate(group_keys):
        if i in merged:
            continue
        group = list(sig_groups[key_i])
        for j in range(i + 1, len(group_keys)):
            if j in merged:
                continue
            key_j = group_keys[j]
            sim = _signature_similarity(key_i, key_j)

            # Also check stack trace overlap
            frames_i: Set[str] = set()
            for idx in sig_groups[key_i]:
                frames_i.update(entries[idx]["stack_frames"])
            frames_j: Set[str] = set()
            for idx in sig_groups[key_j]:
                frames_j.update(entries[idx]["stack_frames"])
            frame_overlap = len(frames_i & frames_j) if frames_i and frames_j else 0

            if sim >= _MERGE_THRESHOLD or frame_overlap >= 2:
                group.extend(sig_groups[key_j])
                merged.add(j)

        final_groups.append(group)

    # ── Phase 3: Build ErrorCluster objects ──────────────────────────────────
    cluster_id = 0
    for group_indices in final_groups:
        if not group_indices:
            continue

        cluster_id += 1
        group_entries = [entries[i] for i in group_indices]

        # Signature: most common in group
        sig_counter = Counter(e["signature"] for e in group_entries)
        primary_sig = sig_counter.most_common(1)[0][0]

        # Exception type
        exc_counter = Counter(e["exception_type"] for e in group_entries if e["exception_type"])
        exc_type = exc_counter.most_common(1)[0][0] if exc_counter else ""

        # Time range
        valid_ts = sorted(e["timestamp"] for e in group_entries if e["timestamp"])
        first_seen = valid_ts[0].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None
        last_seen = valid_ts[-1].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None

        # Related files
        all_files: Set[str] = set()
        for e in group_entries:
            if e["file_name"]:
                all_files.add(e["file_name"])
            for fl in e["file_lines"]:
                all_files.add(fl[0].split("/")[-1].split("\\")[-1])

        # Related functions
        all_funcs: Set[str] = set()
        for e in group_entries:
            all_funcs.update(e["functions"])

        # Severity (most severe in group)
        sev_priority = {"fatal": 5, "critical": 4, "error": 3, "warning": 2, "info": 1}
        dominant_sev = max(group_entries, key=lambda e: sev_priority.get(e["severity"], 0))["severity"]

        # Sample messages (unique, top 3)
        unique_msgs: List[str] = []
        seen_msgs: Set[str] = set()
        for e in group_entries:
            first_line = e["text"].split("\n")[0][:150]
            normalised = _normalise_error_text(e["text"])
            if normalised not in seen_msgs:
                seen_msgs.add(normalised)
                unique_msgs.append(first_line)
            if len(unique_msgs) >= 3:
                break

        # Common stack frames
        all_frames: List[str] = []
        for e in group_entries:
            all_frames.extend(e["stack_frames"])
        frame_counter = Counter(all_frames)
        common_frames = [f for f, c in frame_counter.most_common(5) if c >= 2] if len(group_entries) >= 2 else []

        cluster = ErrorCluster(
            cluster_id=cluster_id,
            error_signature=primary_sig.split(":", 1)[-1] if ":" in primary_sig else primary_sig,
            exception_type=exc_type,
            count=len(group_entries),
            first_seen=first_seen,
            last_seen=last_seen,
            related_files=sorted(all_files)[:10],
            related_functions=sorted(all_funcs)[:10],
            severity=dominant_sev,
            sample_messages=unique_msgs,
            stack_frames=common_frames,
        )
        report.clusters.append(cluster)

    report.unclustered_count = len(chunks) - sum(c.count for c in report.clusters)
    return report
