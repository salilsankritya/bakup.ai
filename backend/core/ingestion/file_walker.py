"""
core/ingestion/file_walker.py
─────────────────────────────────────────────────────────────────────────────
Read-only recursive scanner for a local project directory.

Security:
  - Opens files with read-only intent only (no os.O_WRONLY, no write calls).
  - Resolves symlinks and rejects any path that escapes the declared root.
  - Skips binary files, generated artefacts, and VCS internals.
  - Never follows symlinks that point outside the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

from core.ingestion.chunker import Chunk, chunk_file
from core.ingestion.log_parser import parse_log_file

# Extensions that are treated as log files and parsed with the log parser
LOG_EXTENSIONS: frozenset = frozenset({".log"})

# Directories to skip entirely — generated, vendored, or irrelevant
SKIP_DIRS: frozenset = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "env", ".env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target",
    ".idea", ".vscode",
})

# File extensions considered text/code
TEXT_EXTENSIONS: frozenset = frozenset({
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".ex", ".exs", ".clj", ".hs", ".lua", ".sh",
    ".bash", ".zsh", ".fish", ".ps1",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env.example",
    # Docs
    ".md", ".rst", ".txt",
    # Query / schema
    ".sql", ".graphql", ".proto",
    # Web
    ".html", ".css", ".scss",
    # Logs
    ".log",
})

# Hard limit: skip files larger than this (bytes)
MAX_FILE_BYTES: int = 512 * 1024  # 512 KB


def _is_safe_path(path: Path, root: Path) -> bool:
    """Return True if path is inside root after resolving symlinks."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def walk_project(project_root: Path) -> Iterator[Chunk]:
    """
    Yield Chunk objects for every readable text file under project_root.
    Read-only: only Path.read_text() is ever called.
    """
    root = project_root.resolve()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)

        # Prune skip dirs in-place so os.walk does not descend into them
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            filepath = current / filename

            # Safety: must stay inside root
            if not _is_safe_path(filepath, root):
                continue

            # Extension filter
            if filepath.suffix.lower() not in TEXT_EXTENSIONS:
                continue

            # Size filter
            try:
                if filepath.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            # Route log files through the log parser for per-entry chunking
            if _is_log_file(filepath):
                yield from parse_log_file(filepath, root)
            else:
                yield from chunk_file(filepath, root, source_type="code")


def _is_log_file(filepath: Path) -> bool:
    """True if the file should be parsed with the log parser."""
    return filepath.suffix.lower() in LOG_EXTENSIONS


def list_indexed_files(project_root: Path) -> List[str]:
    """
    Return a sorted list of relative file paths that would be indexed.
    Useful for dry-runs and diagnostic logging.
    """
    seen: set = set()
    root = project_root.resolve()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for filename in filenames:
            filepath = current / filename
            if not _is_safe_path(filepath, root):
                continue
            if filepath.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                if filepath.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            seen.add(str(filepath.relative_to(root)))

    return sorted(seen)
