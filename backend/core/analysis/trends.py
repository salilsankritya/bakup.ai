"""
core/analysis/trends.py
─────────────────────────────────────────────────────────────────────────────
Error-trend detection for bakup.ai.

Analyses retrieved log chunks to:
  1. Extract timestamps + severity per chunk
  2. Count errors per hour / per day
  3. Detect spike windows (rate exceeds 2× rolling average)
  4. Identify repeating failure signatures (same error message recurring)

Returns a structured TrendReport consumed by the LLM prompt and debug trace.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core.analysis.confidence import detect_severity, extract_timestamp


# ── Repeating-failure fingerprint ──────────────────────────────────────────────

_EXCEPTION_PAT = re.compile(
    r"((?:[\w.]+)?(?:Error|Exception|Fault|Failure)(?::\s*.{0,120})?)",
    re.IGNORECASE,
)
_HTTP_STATUS_PAT = re.compile(
    r"(?:status|code|HTTP)\s*[:=]?\s*(4\d{2}|5\d{2})", re.IGNORECASE
)
_TIMEOUT_PAT = re.compile(r"(?:timed?\s*out|timeout|connection\s+refused)", re.IGNORECASE)

# Log-level keywords that should NOT be treated as error signatures
_LEVEL_ONLY = {"error", "warning", "warn", "info", "debug", "fatal", "critical"}


def _extract_error_signature(text: str) -> Optional[str]:
    """Return a short normalised error fingerprint from chunk text."""
    # Try specific exception patterns first
    for m in _EXCEPTION_PAT.finditer(text):
        sig = m.group(1).strip()
        sig = re.sub(r"\s+", " ", sig)[:120]
        # Skip bare log-level words like "ERROR" / "Error"
        if sig.lower() in _LEVEL_ONLY:
            continue
        return sig

    m = _HTTP_STATUS_PAT.search(text)
    if m:
        return f"HTTP {m.group(1)}"

    if _TIMEOUT_PAT.search(text):
        return "Connection timeout"

    return None


# ── Result models ──────────────────────────────────────────────────────────────

@dataclass
class HourlyCounts:
    """Error counts bucketed by hour."""
    buckets: Dict[str, int] = field(default_factory=dict)  # "YYYY-MM-DD HH:00" → count
    total_errors: int = 0
    total_warnings: int = 0

    def to_dict(self) -> dict:
        return {
            "buckets": dict(self.buckets),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
        }


@dataclass
class SpikeWindow:
    """A detected spike in error rate."""
    start: str       # "YYYY-MM-DD HH:00"
    end: str         # exclusive
    count: int       # errors in this window
    avg_rate: float  # average rate in surrounding windows
    severity: str    # "spike" | "elevated"

    def describe(self) -> str:
        return (
            f"{self.start}–{self.end}: {self.count} errors "
            f"(avg {self.avg_rate:.1f}, {self.severity})"
        )

    def to_dict(self) -> dict:
        return {
            "start": self.start, "end": self.end, "count": self.count,
            "avg_rate": round(self.avg_rate, 2), "severity": self.severity,
        }


@dataclass
class RepeatingFailure:
    """A recurring error signature."""
    signature: str
    occurrences: int
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None

    def describe(self) -> str:
        time_info = ""
        if self.first_seen and self.last_seen and self.first_seen != self.last_seen:
            time_info = f" (first: {self.first_seen}, last: {self.last_seen})"
        elif self.first_seen:
            time_info = f" (at {self.first_seen})"
        return f'"{self.signature}" × {self.occurrences}{time_info}'

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class TrendReport:
    """Aggregated trend analysis result."""
    hourly_counts: HourlyCounts = field(default_factory=HourlyCounts)
    spikes: List[SpikeWindow] = field(default_factory=list)
    repeating_failures: List[RepeatingFailure] = field(default_factory=list)
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    chunk_count: int = 0

    @property
    def has_trends(self) -> bool:
        return bool(self.spikes or self.repeating_failures)

    def summary_text(self) -> str:
        """Produce a human-readable summary for inclusion in LLM context."""
        lines = []
        if self.time_range_start and self.time_range_end:
            lines.append(
                f"Time Range: {self.time_range_start} → {self.time_range_end}"
            )
        lines.append(
            f"Totals: {self.hourly_counts.total_errors} error(s), "
            f"{self.hourly_counts.total_warnings} warning(s) "
            f"across {self.chunk_count} log entries"
        )

        if self.spikes:
            lines.append("")
            lines.append("Error Spikes Detected:")
            for sp in self.spikes:
                lines.append(f"  • {sp.describe()}")

        if self.repeating_failures:
            lines.append("")
            lines.append("Repeating Failures:")
            for rf in self.repeating_failures:
                lines.append(f"  • {rf.describe()}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "hourly_counts": self.hourly_counts.to_dict(),
            "spikes": [s.to_dict() for s in self.spikes],
            "repeating_failures": [r.to_dict() for r in self.repeating_failures],
            "time_range_start": self.time_range_start,
            "time_range_end": self.time_range_end,
            "chunk_count": self.chunk_count,
            "has_trends": self.has_trends,
        }


# ── Main analysis function ─────────────────────────────────────────────────────

def analyze_error_trends(chunks: list) -> TrendReport:
    """
    Analyse retrieved chunks for error trends.

    Args:
        chunks: List of RankedResult (or any object with .text attribute).

    Returns:
        TrendReport with hourly counts, spike detection, repeating failures.
    """
    report = TrendReport(chunk_count=len(chunks))

    if not chunks:
        return report

    # ── Phase 1: extract per-chunk metadata ──────────────────────────────────
    entries: List[Tuple[Optional[datetime], str, str]] = []  # (ts, severity, text)

    for chunk in chunks:
        ts = extract_timestamp(chunk.text)
        sev = detect_severity(chunk.text)
        entries.append((ts, sev, chunk.text))

    # ── Phase 2: time range ──────────────────────────────────────────────────
    timestamps = sorted(ts for ts, _, _ in entries if ts is not None)
    if timestamps:
        report.time_range_start = timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
        report.time_range_end   = timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")

    # ── Phase 3: hourly bucket counts ────────────────────────────────────────
    hourly: Dict[str, int] = defaultdict(int)
    total_err = 0
    total_warn = 0

    for ts, sev, _ in entries:
        if sev in ("error", "critical", "fatal"):
            total_err += 1
            if ts:
                bucket = ts.strftime("%Y-%m-%d %H:00")
                hourly[bucket] += 1
        elif sev == "warning":
            total_warn += 1

    report.hourly_counts = HourlyCounts(
        buckets=dict(sorted(hourly.items())),
        total_errors=total_err,
        total_warnings=total_warn,
    )

    # ── Phase 4: spike detection ─────────────────────────────────────────────
    if len(hourly) >= 2:
        sorted_buckets = sorted(hourly.items())
        counts = [c for _, c in sorted_buckets]
        avg = sum(counts) / len(counts)

        for bucket_key, count in sorted_buckets:
            if avg > 0 and count >= 2 * avg and count >= 2:
                # Determine severity of spike
                ratio = count / avg if avg else count
                severity = "spike" if ratio >= 3 else "elevated"

                # End bucket = start + 1h
                try:
                    start_dt = datetime.strptime(bucket_key, "%Y-%m-%d %H:00")
                    end_dt = start_dt + timedelta(hours=1)
                    end_key = end_dt.strftime("%Y-%m-%d %H:00")
                except ValueError:
                    end_key = bucket_key

                report.spikes.append(SpikeWindow(
                    start=bucket_key,
                    end=end_key,
                    count=count,
                    avg_rate=round(avg, 2),
                    severity=severity,
                ))

    # ── Phase 5: repeating failure signatures ────────────────────────────────
    sig_occurrences: Dict[str, List[Optional[datetime]]] = defaultdict(list)

    for ts, sev, text in entries:
        if sev in ("error", "critical", "fatal", "warning"):
            sig = _extract_error_signature(text)
            if sig:
                sig_occurrences[sig].append(ts)

    for sig, ts_list in sorted(sig_occurrences.items(), key=lambda x: -len(x[1])):
        if len(ts_list) < 2:
            continue
        valid_ts = sorted(t for t in ts_list if t is not None)
        report.repeating_failures.append(RepeatingFailure(
            signature=sig,
            occurrences=len(ts_list),
            first_seen=valid_ts[0].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None,
            last_seen=valid_ts[-1].strftime("%Y-%m-%d %H:%M:%S") if valid_ts else None,
        ))

    return report
