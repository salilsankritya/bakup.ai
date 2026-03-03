"""
core/ingestion/chunker.py
─────────────────────────────────────────────────────────────────────────────
Splits source text into overlapping chunks suitable for embedding.

Strategy:
  - Split on lines.
  - Slide a window of CHUNK_LINES lines with OVERLAP_LINES lines of overlap.
  - Each chunk records its source file path, start line, and end line.
  - Chunks shorter than MIN_CHUNK_CHARS are discarded (e.g. blank files).

This is deliberately line-based rather than token-based so we can attach
exact line references to every answer without a tokenizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

CHUNK_LINES: int = 40     # Lines per chunk
OVERLAP_LINES: int = 8    # Lines shared between adjacent chunks
MIN_CHUNK_CHARS: int = 60 # Discard chunks shorter than this


@dataclass(frozen=True)
class Chunk:
    """A single indexable piece of a source file or log."""
    text: str
    source_file: str    # Relative path from project root
    line_start: int     # 1-based
    line_end: int       # 1-based inclusive
    source_type: str    # "code" | "log"


def chunk_text(
    text: str,
    source_file: str,
    source_type: str = "code",
) -> List[Chunk]:
    """
    Split text into overlapping line-window chunks.
    Returns an empty list for empty or binary-looking content.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: List[Chunk] = []
    step = max(1, CHUNK_LINES - OVERLAP_LINES)
    total = len(lines)

    i = 0
    while i < total:
        end = min(i + CHUNK_LINES, total)
        window = lines[i:end]
        chunk_text_str = "\n".join(window)

        if len(chunk_text_str.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(
                text=chunk_text_str,
                source_file=source_file,
                line_start=i + 1,
                line_end=end,
                source_type=source_type,
            ))

        i += step

    return chunks


def chunk_file(filepath: Path, project_root: Path, source_type: str = "code") -> List[Chunk]:
    """
    Read a single file and return its chunks.
    Returns [] if the file cannot be decoded as text.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, PermissionError):
        try:
            text = filepath.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return []

    relative = str(filepath.relative_to(project_root))
    return chunk_text(text, source_file=relative, source_type=source_type)
