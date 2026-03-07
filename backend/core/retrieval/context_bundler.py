"""
core/retrieval/context_bundler.py
─────────────────────────────────────────────────────────────────────────────
Retrieval context bundling — prevents sending fragmented code to the LLM.

Instead of returning isolated chunks, this module:
  1. Takes the primary retrieval results
  2. Fetches surrounding chunks from the same file (siblings)
  3. Fetches import definitions referenced by the primary chunks
  4. Bundles everything into a coherent context window

This ensures the LLM sees complete functions, their imports, and related
helpers — not random 40-line fragments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from core.retrieval.models import RetrievedChunk
from core.retrieval.ranker import RankedResult

logger = logging.getLogger("bakup.retrieval")


@dataclass
class ContextBundle:
    """A bundled retrieval result with surrounding context."""
    primary: RankedResult
    siblings: List[RankedResult]       # Nearby chunks from the same file
    import_chunks: List[RankedResult]  # Chunks that define imported symbols


def _fetch_siblings_from_ranked(
    primary: RankedResult,
    all_ranked: List[RankedResult],
    max_siblings: int = 2,
) -> List[RankedResult]:
    """
    Find chunks from the same file that are adjacent by line number.
    Returns up to max_siblings chunks sorted by proximity.
    """
    siblings: List[RankedResult] = []
    for r in all_ranked:
        if r is primary:
            continue
        if r.source_file != primary.source_file:
            continue
        # Check adjacency: within 100 lines
        gap = min(
            abs(r.line_start - primary.line_end),
            abs(primary.line_start - r.line_end),
        )
        if gap <= 100:
            siblings.append(r)

    # Sort by proximity
    siblings.sort(
        key=lambda s: min(
            abs(s.line_start - primary.line_end),
            abs(primary.line_start - s.line_end),
        )
    )
    return siblings[:max_siblings]


def _find_import_chunks(
    primary: RankedResult,
    all_ranked: List[RankedResult],
) -> List[RankedResult]:
    """
    If the primary chunk references imports, look for chunks that define
    those imported modules/functions in the same retrieval set.
    """
    if not primary.imports:
        return []

    # Extract imported module/symbol names from the imports string
    imported_names: Set[str] = set()
    for line in primary.imports.split("\n"):
        line = line.strip()
        if not line:
            continue
        # "from foo.bar import baz, qux" → extract baz, qux
        if "import" in line:
            parts = line.split("import")
            if len(parts) >= 2:
                symbols = parts[-1].strip()
                for sym in symbols.split(","):
                    sym = sym.strip().split(" as ")[0].strip()
                    if sym and sym != "*":
                        imported_names.add(sym)

    if not imported_names:
        return []

    # Search ranked results for chunks that define those symbols
    import_chunks: List[RankedResult] = []
    for r in all_ranked:
        if r is primary:
            continue
        # Match by function_name or class_name
        if r.function_name and r.function_name in imported_names:
            import_chunks.append(r)
        elif r.class_name and r.class_name in imported_names:
            import_chunks.append(r)

    return import_chunks[:3]


def bundle_context(
    ranked: List[RankedResult],
    top_n: int = 5,
    max_siblings: int = 2,
) -> List[ContextBundle]:
    """
    Bundle top-N ranked results with their surrounding context.

    Each bundle contains:
      - The primary chunk (highest confidence)
      - Sibling chunks from the same file (adjacent by line number)
      - Import definition chunks (functions/classes referenced by imports)

    Returns bundles ordered by primary confidence (descending).
    """
    if not ranked:
        return []

    bundles: List[ContextBundle] = []
    used_keys: Set[str] = set()

    for primary in ranked[:top_n]:
        key = f"{primary.source_file}:{primary.line_start}"
        if key in used_keys:
            continue
        used_keys.add(key)

        siblings = _fetch_siblings_from_ranked(primary, ranked, max_siblings)
        import_chunks = _find_import_chunks(primary, ranked)

        # Mark siblings/imports as used so they don't become primary in another bundle
        for s in siblings:
            used_keys.add(f"{s.source_file}:{s.line_start}")
        for ic in import_chunks:
            used_keys.add(f"{ic.source_file}:{ic.line_start}")

        bundles.append(ContextBundle(
            primary=primary,
            siblings=siblings,
            import_chunks=import_chunks,
        ))

    logger.debug(
        "Context bundling: %d primary → %d bundle(s), %d sibling(s), %d import ref(s)",
        len(ranked[:top_n]),
        len(bundles),
        sum(len(b.siblings) for b in bundles),
        sum(len(b.import_chunks) for b in bundles),
    )

    return bundles


def bundles_to_ranked_list(bundles: List[ContextBundle]) -> List[RankedResult]:
    """
    Flatten bundles back into a deduplicated ranked list for use
    in the existing LLM context builder.

    Order: primary first, then siblings, then imports — within each bundle.
    """
    seen: Set[str] = set()
    result: List[RankedResult] = []

    for bundle in bundles:
        for chunk in [bundle.primary] + bundle.siblings + bundle.import_chunks:
            key = f"{chunk.source_file}:{chunk.line_start}:{chunk.line_end}"
            if key not in seen:
                seen.add(key)
                result.append(chunk)

    return result


def build_bundled_context_block(
    bundles: List[ContextBundle],
    max_chars: int = 800,
) -> str:
    """
    Render bundled context for the LLM prompt.

    Groups related chunks under file headers so the LLM sees coherent
    code blocks rather than random fragments.
    """
    parts: List[str] = []
    idx = 0

    for bundle in bundles:
        all_chunks = [bundle.primary] + bundle.siblings + bundle.import_chunks

        # Group by file
        file_groups: Dict[str, List[RankedResult]] = {}
        for c in all_chunks:
            file_groups.setdefault(c.source_file, []).append(c)

        for source_file, chunks in file_groups.items():
            # Sort by line number within each file
            chunks.sort(key=lambda c: c.line_start)

            for c in chunks:
                idx += 1
                text = c.text[:max_chars]
                if len(c.text) > max_chars:
                    text += "\n[...truncated]"

                label_parts = [f"[{idx}] {c.source_file}"]
                label_parts.append(f"lines {c.line_start}–{c.line_end}")
                label_parts.append(f"confidence: {c.confidence:.2f}")
                label_parts.append(f"type: {c.source_type}")
                if c.function_name:
                    label_parts.append(f"function: {c.function_name}")
                if c.class_name:
                    label_parts.append(f"class: {c.class_name}")

                parts.append(
                    "  ".join(label_parts) + "\n" + text
                )

    return "\n\n---\n\n".join(parts)
