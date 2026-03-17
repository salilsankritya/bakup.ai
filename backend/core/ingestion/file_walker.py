"""
core/ingestion/file_walker.py
─────────────────────────────────────────────────────────────────────────────
Read-only recursive scanner for a local project directory.

Security:
  - Opens files with read-only intent only (no os.O_WRONLY, no write calls).
  - Resolves symlinks and rejects any path that escapes the declared root.
  - Skips binary files, generated artefacts, and VCS internals.
  - Never follows symlinks that point outside the project root.

Code-aware ingestion:
  - Routes code files through the language-aware code chunker.
  - Routes log files through the log parser.
  - Produces structured chunks with function/class/method metadata.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Iterator, List

from core.ingestion.chunker import Chunk, chunk_file
from core.ingestion.code_chunker import chunk_file_code_aware
from core.ingestion.log_parser import parse_log_file

logger = logging.getLogger("bakup.ingestion")

# Extensions that are treated as log files and parsed with the log parser
LOG_EXTENSIONS: frozenset = frozenset({".log", ".out"})

# Directories to skip entirely — generated, vendored, or irrelevant
SKIP_DIRS: frozenset = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "env", ".env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".next", ".nuxt", "target",
    ".idea", ".vscode",
    "model-weights",       # embedding model binaries — never index
    "vectordb",            # ChromaDB persistence — never index
    "coverage", ".coverage",  # test coverage artefacts
    ".tox", ".nox",           # test automation
    ".cache",                 # generic caches (pip, etc.)
    "__pypackages__",         # PEP 582 local packages
    ".eggs",                  # setuptools build artefacts
    # Additional noise directories
    "out", ".output",         # common build output folders
    ".gradle", ".maven",      # Java build caches
    ".cargo",                 # Rust build cache
    "vendor",                 # vendored dependencies (Go, PHP, Ruby)
    "bower_components",       # legacy JS dependency manager
    ".terraform",             # Terraform state
    ".serverless",            # Serverless framework artefacts
    ".webpack",               # Webpack cache
    ".parcel-cache",          # Parcel bundler cache
    "site-packages",          # Python installed packages
    ".sass-cache",            # Sass compile cache
    "tmp", ".tmp",            # temporary directories
})

# File extensions to always skip (binary / model weights / generated artefacts)
SKIP_EXTENSIONS: frozenset = frozenset({
    # ML model weights
    ".bin", ".model", ".vocab", ".onnx", ".pt", ".pth",
    ".safetensors", ".gguf", ".ggml", ".h5", ".hdf5",
    ".tflite", ".pb",
    # Serialised data
    ".pkl", ".pickle", ".npy", ".npz", ".parquet", ".feather",
    # Native binaries
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj", ".a", ".lib",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".xz", ".rar",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
    ".bmp", ".tiff", ".psd",
    # Audio / video
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".flv", ".mkv",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Source maps & minified bundles
    ".map",
    # Lock files (package managers)
    ".lock",
    # Database files
    ".db", ".sqlite", ".sqlite3",
    # Coverage / profiling artefacts
    ".coverage", ".prof", ".trace",
    # Compiled Python
    ".pyc", ".pyo",
    # Java / Kotlin bytecode
    ".class", ".jar", ".war", ".ear",
    # .NET
    ".nupkg",
})

# File names to always skip (noise files that clutter retrieval)
SKIP_FILENAMES: frozenset = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", "Gemfile.lock",
    "composer.lock", "Cargo.lock", "go.sum",
    ".DS_Store", "Thumbs.db", "desktop.ini",
})

# File suffixes to skip (e.g. minified bundles)
# Checked against the full filename, not just the extension.
SKIP_SUFFIXES: tuple = (
    ".min.js", ".min.css", ".bundle.js", ".chunk.js",
    ".min.map", ".bundle.map",
)

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
    ".log", ".out",
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


def walk_project(project_root: Path, namespace: str = "") -> Iterator[Chunk]:
    """
    Yield Chunk objects for every readable text file under project_root.
    Read-only: only Path.read_text() is ever called.

    Code files are routed through the language-aware code chunker.
    Log files are routed through the log parser.

    When namespace is provided, updates the symbol graph during ingestion.
    """
    root = project_root.resolve()

    # Stats counters for debug logging
    file_count = 0
    total_chunks = 0
    functions_detected = 0
    classes_detected = 0
    methods_detected = 0
    chunks_per_file: Counter = Counter()

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

            # Skip binary / model weight extensions
            if filepath.suffix.lower() in SKIP_EXTENSIONS:
                continue

            # Skip known noise filenames (lockfiles, OS metadata)
            if filename in SKIP_FILENAMES:
                continue

            # Skip minified / bundled files by suffix pattern
            fname_lower = filename.lower()
            if any(fname_lower.endswith(sfx) for sfx in SKIP_SUFFIXES):
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

            file_count += 1
            relative = str(filepath.relative_to(root))
            file_chunk_count = 0

            try:
                # Route log files through the log parser for per-entry chunking
                if _is_log_file(filepath):
                    for chunk in parse_log_file(filepath, root):
                        file_chunk_count += 1
                        yield chunk
                else:
                    # Use code-aware chunker for all code files
                    for chunk in chunk_file_code_aware(filepath, root, namespace=namespace):
                        file_chunk_count += 1
                        # Count code structures
                        kind = getattr(chunk, 'chunk_kind', '')
                        if kind == 'function':
                            functions_detected += 1
                        elif kind == 'class':
                            classes_detected += 1
                        elif kind == 'method':
                            methods_detected += 1
                        yield chunk
            except Exception as exc:
                logger.warning("Skipping file %s: %s", relative, exc)
                continue

            chunks_per_file[relative] = file_chunk_count
            total_chunks += file_chunk_count

    # Debug summary
    logger.info(
        "Code-aware ingestion complete: %d file(s), %d chunk(s) "
        "[%d function(s), %d class(es), %d method(s)]",
        file_count, total_chunks,
        functions_detected, classes_detected, methods_detected,
    )
    # Log per-file breakdown (top 20 files by chunk count)
    if chunks_per_file:
        top_files = chunks_per_file.most_common(20)
        logger.info(
            "Chunks per file (top %d): %s",
            len(top_files),
            {f: c for f, c in top_files},
        )


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
            if filepath.suffix.lower() in SKIP_EXTENSIONS:
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
