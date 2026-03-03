"""
core/retrieval/models.py
─────────────────────────────────────────────────────────────────────────────
Lightweight dataclasses shared between vector_store.py and ranker.py.
No external dependencies — safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity query, with its raw distance score."""
    text: str
    source_file: str
    line_start: int
    line_end: int
    source_type: str    # "code" | "log"
    distance: float     # cosine distance: 0 = identical, 1 = orthogonal
    file_name: str = ""
    severity: str = "info"  # "error" | "warning" | "info"
    detected_timestamp: Optional[str] = None
