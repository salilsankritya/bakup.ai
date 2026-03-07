"""
core/ingestion/code_chunker.py
─────────────────────────────────────────────────────────────────────────────
Converts language-parsed CodeUnits into indexable Chunks with rich metadata.

Strategy:
  - Each logical code unit (function, class, method, config block) becomes
    one Chunk including its imports, docstring, and file context.
  - Large units (>MAX_UNIT_LINES) are split with overlap so no context is lost.
  - Small adjacent units are merged if they fit within MIN_MERGE_LINES.
  - Every chunk carries structured metadata: language, function_name,
    class_name, imports, docstring, decorators.

Falls back to the original line-window chunker for unsupported languages
or files that produce no parsed units.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from core.ingestion.chunker import Chunk, chunk_text, MIN_CHUNK_CHARS
from core.ingestion.code_parser import (
    CodeUnit,
    detect_language,
    parse_file,
)

logger = logging.getLogger("bakup.code_chunker")

# ── Tuning constants ──────────────────────────────────────────────────────────

MAX_UNIT_LINES: int = 80       # Split units larger than this
SPLIT_OVERLAP: int = 8         # Overlap lines when splitting large units
MIN_MERGE_LINES: int = 10      # Merge units smaller than this with neighbours
CONTEXT_HEADER_LINES: int = 3  # Number of import/context lines to prepend


def _build_context_header(
    source_file: str,
    language: str,
    imports: List[str],
    class_name: str = "",
) -> str:
    """
    Build a short context header prepended to each chunk so the embedding
    model and LLM can locate the code within the project.
    """
    parts = [f"# File: {source_file}"]
    if language != "text":
        parts.append(f"# Language: {language}")
    if class_name:
        parts.append(f"# Class: {class_name}")
    # Include up to CONTEXT_HEADER_LINES imports for context
    if imports:
        for imp in imports[:CONTEXT_HEADER_LINES]:
            parts.append(imp)
    return "\n".join(parts)


def _split_large_unit(unit: CodeUnit, source_file: str) -> List[Chunk]:
    """Split a CodeUnit that exceeds MAX_UNIT_LINES into overlapping chunks."""
    lines = unit.text.splitlines()
    total = len(lines)
    chunks: List[Chunk] = []
    step = max(1, MAX_UNIT_LINES - SPLIT_OVERLAP)

    header = _build_context_header(source_file, unit.language, unit.imports, unit.class_name)

    i = 0
    part = 0
    while i < total:
        end = min(i + MAX_UNIT_LINES, total)
        window = lines[i:end]
        chunk_text_str = header + "\n" + "\n".join(window)

        if len(chunk_text_str.strip()) >= MIN_CHUNK_CHARS:
            part += 1
            chunks.append(Chunk(
                text=chunk_text_str,
                source_file=source_file,
                line_start=unit.start_line + i,
                line_end=unit.start_line + end - 1,
                source_type="code",
                file_name=Path(source_file).name,
                # Extended metadata
                language=unit.language,
                function_name=unit.name if unit.kind in ("function", "method") else "",
                class_name=unit.class_name,
                chunk_kind=unit.kind,
                docstring=unit.docstring if part == 1 else "",
                imports="\n".join(unit.imports[:5]),
            ))

        i += step

    return chunks


def code_units_to_chunks(
    units: List[CodeUnit],
    source_file: str,
) -> List[Chunk]:
    """
    Convert parsed CodeUnits into Chunks with context headers and metadata.
    """
    if not units:
        return []

    chunks: List[Chunk] = []

    for unit in units:
        line_count = unit.text.count("\n") + 1

        # Large unit → split
        if line_count > MAX_UNIT_LINES:
            chunks.extend(_split_large_unit(unit, source_file))
            continue

        # Build context-enriched text
        header = _build_context_header(
            source_file, unit.language, unit.imports, unit.class_name,
        )

        # For functions/methods, include the docstring prominently
        enriched_parts = [header]
        if unit.comments:
            enriched_parts.append(unit.comments)
        if unit.decorators:
            enriched_parts.extend(unit.decorators)
        enriched_parts.append(unit.text)

        enriched_text = "\n".join(enriched_parts)

        if len(enriched_text.strip()) < MIN_CHUNK_CHARS:
            continue

        chunks.append(Chunk(
            text=enriched_text,
            source_file=source_file,
            line_start=unit.start_line,
            line_end=unit.end_line,
            source_type="code",
            file_name=Path(source_file).name,
            # Extended metadata
            language=unit.language,
            function_name=unit.name if unit.kind in ("function", "method") else "",
            class_name=unit.class_name,
            chunk_kind=unit.kind,
            docstring=unit.docstring[:500] if unit.docstring else "",
            imports="\n".join(unit.imports[:5]),
        ))

    return chunks


def chunk_file_code_aware(
    filepath: Path,
    project_root: Path,
) -> List[Chunk]:
    """
    Read a source file, detect its language, parse structures, and return
    code-aware Chunks with metadata.

    Falls back to the original line-window chunker for:
      - Files that cannot be decoded
      - Languages not supported by the parser
      - Files where parsing produces no units
    """
    # Read file
    try:
        text = filepath.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, PermissionError):
        try:
            text = filepath.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return []

    if not text.strip():
        return []

    relative = str(filepath.relative_to(project_root))
    language = detect_language(filepath)

    # Parse with language-specific parser
    units = parse_file(text, language)

    if units:
        chunks = code_units_to_chunks(units, relative)
        if chunks:
            # Log parsing stats
            func_count = sum(1 for u in units if u.kind == "function")
            class_count = sum(1 for u in units if u.kind == "class")
            method_count = sum(1 for u in units if u.kind == "method")
            logger.debug(
                "Parsed %s (%s): %d unit(s) → %d chunk(s) "
                "[%d func, %d class, %d method]",
                relative, language, len(units), len(chunks),
                func_count, class_count, method_count,
            )
            return chunks

    # Fallback: original line-window chunker
    logger.debug("Fallback to line-window chunker for %s (%s)", relative, language)
    return chunk_text(text, source_file=relative, source_type="code")
