"""
core/ingestion/log_parser.py
─────────────────────────────────────────────────────────────────────────────
Parses local log files into indexable chunks.

Supports:
  - Common structured log formats (timestamps at start of line)
  - Plain multi-line logs (falls back to line-window chunking)
  - Large files: reads in streaming fashion, does not load into memory at once

Each logical log entry (one timestamp-bounded event) becomes one chunk.
Entries longer than MAX_LOG_ENTRY_LINES are truncated at that boundary.

Metadata enrichment (v2):
  - severity:           "error" | "warning" | "info"
  - detected_timestamp: first timestamp found in the chunk
  - file_name:          basename of the log file
  - last_modified:      ISO 8601 mtime of the file on disk
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from core.ingestion.chunker import Chunk, CHUNK_LINES, MIN_CHUNK_CHARS

# Patterns that indicate the start of a new log entry (line begins with timestamp)
_TIMESTAMP_PATTERNS: List[re.Pattern] = [
    # ISO 8601:        2024-01-15T14:23:01
    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
    # Common log:      [2024-01-15 14:23:01]
    re.compile(r"^\[?\d{4}-\d{2}-\d{2}[\s_]\d{2}:\d{2}"),
    # Epoch millis:    1705329781234
    re.compile(r"^\d{13,}\s"),
    # Level prefix:    ERROR | WARN | INFO | DEBUG | FATAL
    re.compile(r"^(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL)\b", re.IGNORECASE),
]

MAX_LOG_ENTRY_LINES: int = 80  # Guard against runaway stack traces

# ── Severity detection ─────────────────────────────────────────────────────────

_ERROR_INDICATORS = re.compile(
    r"\bERROR\b|\bException\b|\bTraceback\b|\bFailed\b|\bCritical\b|\bFatal\b|\bFATAL\b|\bCRITICAL\b",
    re.IGNORECASE,
)
_WARNING_INDICATORS = re.compile(
    r"\bWARN(?:ING)?\b",
    re.IGNORECASE,
)

# ── Timestamp extraction ──────────────────────────────────────────────────────

_TS_EXTRACT = [
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"),
    re.compile(r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]"),
]


def _detect_severity(text: str) -> str:
    """Detect severity from chunk text. error > warning > info."""
    if _ERROR_INDICATORS.search(text):
        return "error"
    if _WARNING_INDICATORS.search(text):
        return "warning"
    return "info"


def _extract_first_timestamp(text: str) -> Optional[str]:
    """Extract the first parseable timestamp from chunk text as ISO string."""
    first_line = text.split("\n")[0]
    for pat in _TS_EXTRACT:
        m = pat.search(first_line)
        if m:
            raw = m.group(1).replace("T", " ")
            try:
                dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue
    return None


def _get_file_mtime_iso(filepath: Path) -> Optional[str]:
    """Return the file's last-modified time as ISO 8601 string."""
    try:
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _starts_new_entry(line: str) -> bool:
    return any(p.match(line) for p in _TIMESTAMP_PATTERNS)


def _make_chunk(
    lines: List[str],
    source_file: str,
    start_line: int,
    *,
    file_name: str = "",
    last_modified: Optional[str] = None,
) -> Chunk | None:
    text = "\n".join(lines).strip()
    if len(text) < MIN_CHUNK_CHARS:
        return None
    return Chunk(
        text=text,
        source_file=source_file,
        line_start=start_line,
        line_end=start_line + len(lines) - 1,
        source_type="log",
        file_name=file_name,
        last_modified=last_modified,
        detected_timestamp=_extract_first_timestamp(text),
        severity=_detect_severity(text),
    )


def parse_log_file(filepath: Path, project_root: Path) -> List[Chunk]:
    """
    Parse a log file into per-entry chunks with severity and metadata.
    Falls back to line-window chunking if no timestamp structure is detected.
    Never loads more than one entry into memory at a time.
    """
    try:
        relative = str(filepath.relative_to(project_root))
    except ValueError:
        relative = filepath.name

    fname = filepath.name
    mtime = _get_file_mtime_iso(filepath)

    chunks: List[Chunk] = []
    current_entry: List[str] = []
    current_start: int = 1
    structured_entries_seen: int = 0

    try:
        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.rstrip("\n")

                if _starts_new_entry(line):
                    # Flush previous entry
                    if current_entry:
                        chunk = _make_chunk(
                            current_entry, relative, current_start,
                            file_name=fname, last_modified=mtime,
                        )
                        if chunk:
                            chunks.append(chunk)
                    current_entry = [line]
                    current_start = lineno
                    structured_entries_seen += 1
                else:
                    current_entry.append(line)
                    # Guard against unbounded entries (e.g. huge stack traces)
                    if len(current_entry) >= MAX_LOG_ENTRY_LINES:
                        chunk = _make_chunk(
                            current_entry, relative, current_start,
                            file_name=fname, last_modified=mtime,
                        )
                        if chunk:
                            chunks.append(chunk)
                        current_entry = []
                        current_start = lineno + 1

            # Flush final entry
            if current_entry:
                chunk = _make_chunk(
                    current_entry, relative, current_start,
                    file_name=fname, last_modified=mtime,
                )
                if chunk:
                    chunks.append(chunk)

    except (PermissionError, OSError):
        return []

    # If no structured entries found, fall back to line-window chunking
    if structured_entries_seen == 0 and chunks:
        return chunks  # Already chunked by MAX_LOG_ENTRY_LINES windows
    if structured_entries_seen == 0 and not chunks:
        return _fallback_chunk(filepath, relative)

    return chunks


def _fallback_chunk(filepath: Path, relative: str) -> List[Chunk]:
    """Line-window fallback for unstructured logs with metadata enrichment."""
    from core.ingestion.chunker import chunk_file

    fname = filepath.name
    mtime = _get_file_mtime_iso(filepath)

    try:
        root = filepath.parent
        raw_chunks = chunk_file(filepath, root, source_type="log")
        # Enrich with metadata (Chunk is frozen, so rebuild)
        enriched: List[Chunk] = []
        for c in raw_chunks:
            enriched.append(Chunk(
                text=c.text,
                source_file=c.source_file,
                line_start=c.line_start,
                line_end=c.line_end,
                source_type=c.source_type,
                file_name=fname,
                last_modified=mtime,
                detected_timestamp=_extract_first_timestamp(c.text),
                severity=_detect_severity(c.text),
            ))
        return enriched
    except Exception:
        return []
