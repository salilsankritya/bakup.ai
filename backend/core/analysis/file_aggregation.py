"""
core/analysis/file_aggregation.py
─────────────────────────────────────────────────────────────────────────────
Cross-file error distribution analysis.

Aggregates retrieved chunks by source file to produce:
  - Error frequency per file (ranked by count)
  - Severity distribution per file
  - Dominant files (top contributors to issues)
  - Plain-text summary suitable for embedding into LLM prompts

This module does NOT call the LLM — it produces deterministic aggregation
reports used as context for the LLM summary step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.retrieval.models import RetrievedChunk


@dataclass
class FileErrorStats:
    """Error statistics for a single file."""
    file_name: str
    source_file: str           # relative path
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    total_chunks: int = 0

    @property
    def severity_score(self) -> float:
        """Weighted score: errors × 3 + warnings × 1."""
        return self.error_count * 3 + self.warning_count


@dataclass
class FileAggregationReport:
    """Cross-file aggregation report."""
    files_affected: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    file_stats: List[FileErrorStats] = field(default_factory=list)
    dominant_file: Optional[str] = None     # file_name with most errors
    summary_text: str = ""

    @property
    def files_with_errors(self) -> int:
        return sum(1 for f in self.file_stats if f.error_count > 0)


def aggregate_by_file(chunks: List[RetrievedChunk]) -> FileAggregationReport:
    """
    Aggregate a list of retrieved chunks by their source file.

    Counts errors, warnings, and info entries per file.
    Ranks files by error count descending.
    Produces a plain-text summary block.
    """
    if not chunks:
        return FileAggregationReport()

    # Group by file_name (fall back to source_file basename)
    file_map: Dict[str, FileErrorStats] = {}

    for chunk in chunks:
        key = chunk.file_name or chunk.source_file.split("/")[-1].split("\\")[-1]
        if key not in file_map:
            file_map[key] = FileErrorStats(
                file_name=key,
                source_file=chunk.source_file,
            )
        stats = file_map[key]
        stats.total_chunks += 1

        sev = getattr(chunk, "severity", "info") or "info"
        if sev == "error":
            stats.error_count += 1
        elif sev == "warning":
            stats.warning_count += 1
        else:
            stats.info_count += 1

    # Sort by error count descending, then warning count, then name
    ranked = sorted(
        file_map.values(),
        key=lambda s: (-s.error_count, -s.warning_count, s.file_name),
    )

    total_errors = sum(s.error_count for s in ranked)
    total_warnings = sum(s.warning_count for s in ranked)
    dominant = ranked[0].file_name if ranked and ranked[0].error_count > 0 else None

    # Build summary text
    lines: List[str] = []
    lines.append("Error Distribution Across Files:")
    for s in ranked:
        parts = []
        if s.error_count:
            parts.append(f"{s.error_count} error{'s' if s.error_count > 1 else ''}")
        if s.warning_count:
            parts.append(f"{s.warning_count} warning{'s' if s.warning_count > 1 else ''}")
        if not parts:
            parts.append("info only")
        lines.append(f"  {s.file_name} — {', '.join(parts)}")

    if len(ranked) > 1 and total_errors > 0:
        lines.append(f"\nDominant error source: {dominant} ({ranked[0].error_count}/{total_errors} errors)")

    summary = "\n".join(lines)

    return FileAggregationReport(
        files_affected=len(ranked),
        total_errors=total_errors,
        total_warnings=total_warnings,
        file_stats=ranked,
        dominant_file=dominant,
        summary_text=summary,
    )
