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
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List

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


def _starts_new_entry(line: str) -> bool:
    return any(p.match(line) for p in _TIMESTAMP_PATTERNS)


def _make_chunk(lines: List[str], source_file: str, start_line: int) -> Chunk | None:
    text = "\n".join(lines).strip()
    if len(text) < MIN_CHUNK_CHARS:
        return None
    return Chunk(
        text=text,
        source_file=source_file,
        line_start=start_line,
        line_end=start_line + len(lines) - 1,
        source_type="log",
    )


def parse_log_file(filepath: Path, project_root: Path) -> List[Chunk]:
    """
    Parse a log file into per-entry chunks.
    Falls back to line-window chunking if no timestamp structure is detected.
    Never loads more than one entry into memory at a time.
    """
    try:
        relative = str(filepath.relative_to(project_root))
    except ValueError:
        relative = filepath.name

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
                        chunk = _make_chunk(current_entry, relative, current_start)
                        if chunk:
                            chunks.append(chunk)
                    current_entry = [line]
                    current_start = lineno
                    structured_entries_seen += 1
                else:
                    current_entry.append(line)
                    # Guard against unbounded entries (e.g. huge stack traces)
                    if len(current_entry) >= MAX_LOG_ENTRY_LINES:
                        chunk = _make_chunk(current_entry, relative, current_start)
                        if chunk:
                            chunks.append(chunk)
                        current_entry = []
                        current_start = lineno + 1

            # Flush final entry
            if current_entry:
                chunk = _make_chunk(current_entry, relative, current_start)
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
    """Line-window fallback for unstructured logs."""
    from core.ingestion.chunker import chunk_file
    # Re-use generic chunker
    try:
        root = filepath.parent
        return chunk_file(filepath, root, source_type="log")
    except Exception:
        return []
