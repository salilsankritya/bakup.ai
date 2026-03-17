"""
core/recent_projects.py
─────────────────────────────────────────────────────────────────────────────
Lightweight store for recently indexed projects.

Persists a JSON file alongside the ChromaDB data directory so the list
survives server restarts.  No cloud sync, no accounts — purely local.

Public API:
    record_project(...)   — upsert after a successful /index call
    list_projects()       — return list of recent projects (newest first)
    remove_project(ns)    — delete one entry by namespace
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("bakup.recent_projects")

MAX_RECENT = 10

# Module-level path — set once by init()
_store_path: Path | None = None


def init(data_dir: Path) -> None:
    """
    Call once at startup to set the storage location.
    *data_dir* is typically the same directory as chroma_persist_dir.
    """
    global _store_path
    data_dir.mkdir(parents=True, exist_ok=True)
    _store_path = data_dir / "recent_projects.json"
    logger.info("Recent-projects store: %s", _store_path)


def _load() -> List[dict]:
    if _store_path is None or not _store_path.exists():
        return []
    try:
        data = json.loads(_store_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        logger.warning("Corrupt recent_projects.json — resetting")
    return []


def _save(entries: List[dict]) -> None:
    if _store_path is None:
        return
    try:
        _store_path.write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to write recent_projects.json")


def record_project(
    *,
    project_path: str,
    project_name: str,
    namespace: str,
    source_type: str = "local",      # "local" | "github" | "upload"
    chunks_stored: int = 0,
    branch: Optional[str] = None,
) -> None:
    """
    Add or update a project entry after a successful index operation.
    Keeps at most MAX_RECENT entries; oldest are evicted.
    If *namespace* already exists the entry is updated (upsert).
    """
    entries = _load()

    now = datetime.now(timezone.utc).isoformat()

    # Build the new/updated record
    record: dict[str, Any] = {
        "project_path": project_path,
        "project_name": project_name,
        "namespace": namespace,
        "source_type": source_type,
        "chunks_stored": chunks_stored,
        "last_indexed": now,
    }
    if branch:
        record["branch"] = branch

    # Remove any existing entry with the same namespace (upsert)
    entries = [e for e in entries if e.get("namespace") != namespace]

    # Prepend (most recent first)
    entries.insert(0, record)

    # Trim to limit
    entries = entries[:MAX_RECENT]

    _save(entries)
    logger.info("Recorded recent project: %s (%s)", project_name, source_type)


def list_projects() -> List[dict]:
    """
    Return recent projects with availability info.

    For local paths, checks whether the directory still exists
    and sets ``available: true/false``.  GitHub entries are always
    marked available.
    """
    entries = _load()
    result = []
    for entry in entries:
        entry = dict(entry)  # shallow copy
        src = entry.get("source_type", "local")
        if src == "local":
            entry["available"] = os.path.isdir(entry.get("project_path", ""))
        else:
            # GitHub / upload — always re-indexable
            entry["available"] = True
        result.append(entry)
    return result


def remove_project(namespace: str) -> bool:
    """Remove a project by namespace. Returns True if found and removed."""
    entries = _load()
    filtered = [e for e in entries if e.get("namespace") != namespace]
    if len(filtered) == len(entries):
        return False
    _save(filtered)
    return True
