"""
api/routes/index.py
─────────────────────────────────────────────────────────────────────────────
POST /index          — ingest a local directory (path on the host/container)
POST /index/github   — ingest a GitHub repository (public or token-authenticated)
POST /index/upload   — ingest uploaded files via multipart form data
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from pydantic import BaseModel

logger = logging.getLogger("bakup.ingestion")

router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class IndexLocalRequest(BaseModel):
    path: str               # Absolute path to a local project directory
    log_path: Optional[str] = None  # Optional: absolute path to a log file
    namespace: Optional[str] = None # Optional: override default namespace


class IndexGitHubRequest(BaseModel):
    repo_url: str           # HTTPS clone URL (credentials may be embedded for private repos)
    branch: str = "HEAD"    # Branch or ref to checkout
    namespace: Optional[str] = None


class IndexResponse(BaseModel):
    status: str
    namespace: str
    chunks_stored: int
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────

# Docker volume mount points that local paths must reside under (if set).
# When BAKUP_DOCKER_VOLUMES is set (comma-separated), only paths inside those
# directories are allowed.  Unset = no restriction (native / dev mode).
_DOCKER_VOLUMES: list[str] = [
    v.strip()
    for v in os.environ.get("BAKUP_DOCKER_VOLUMES", "").split(",")
    if v.strip()
]


def _normalize_path(raw: str) -> str:
    """
    Normalize a user-supplied path string.
    Handles Windows backslashes, forward slashes, mixed separators,
    and resolves to an absolute path.
    Works on Windows, macOS, and Linux.
    """
    # Replace any forward slashes with OS separator for consistency
    cleaned = raw.strip().strip('"').strip("'")
    # os.path.normpath handles // and mixed separators
    normed = os.path.normpath(cleaned)
    # os.path.abspath resolves relative paths against cwd
    absolute = os.path.abspath(normed)
    return absolute


def _validate_path(raw: str, *, kind: str = "directory") -> str:
    """
    Normalize *raw*, validate that it exists and is the expected *kind*
    ('directory' or 'file'), and enforce Docker volume restrictions.
    Returns the normalized absolute path string.
    Raises HTTPException with a clear message on any failure.
    """
    logger.info("Path received (raw): %r", raw)

    if not raw or not raw.strip():
        raise HTTPException(
            status_code=400,
            detail="Path is empty. Please provide a valid absolute path.",
        )

    resolved = _normalize_path(raw)
    logger.info("Path resolved (absolute): %s", resolved)

    # ── Existence check ──────────────────────────────────────────────────
    if not os.path.exists(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid path. The {kind} does not exist: {resolved}\n"
                "Ensure the full absolute path is provided and the "
                f"{kind} is accessible by the application."
            ),
        )

    # ── Type check ───────────────────────────────────────────────────────
    if kind == "directory" and not os.path.isdir(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Path is not a directory: {resolved}\n"
                "Please provide the path to a project folder, not a single file."
            ),
        )
    if kind == "file" and not os.path.isfile(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Path is not a file: {resolved}\n"
                "Please provide the path to a log file."
            ),
        )

    # ── Docker volume restriction ────────────────────────────────────────
    if _DOCKER_VOLUMES:
        inside = any(
            resolved.startswith(os.path.normpath(vol))
            for vol in _DOCKER_VOLUMES
        )
        if not inside:
            allowed = ", ".join(_DOCKER_VOLUMES)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Path is outside the allowed mounted volumes.\n"
                    f"Allowed: {allowed}\n"
                    f"Received: {resolved}\n"
                    "When running inside Docker, place your project files "
                    "inside the mounted directory and reference the container path."
                ),
            )

    return resolved


def _derive_namespace(path: str) -> str:
    """Stable namespace from a path string."""
    return hashlib.sha256(path.strip().encode()).hexdigest()[:24]


def _run_local_ingestion(path: str, log_path: Optional[str], namespace: str) -> int:
    """
    Walk a local project, parse any log file, embed all chunks,
    and store them in ChromaDB. Returns total chunks stored.

    Enriched with:
      - Severity distribution debug logging
      - Per-file chunk counts
    """
    from core.ingestion.file_walker import walk_project
    from core.ingestion.log_parser import parse_log_file
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks
    from core.ingestion.chunker import Chunk
    from collections import Counter

    project_root = Path(path).resolve()
    logger.info("Ingesting project root: %s", project_root)
    chunks: list[Chunk] = list(walk_project(project_root))
    logger.info("Walk complete: %d chunk(s) from source files", len(chunks))

    if log_path:
        log_file = Path(log_path).resolve()
        logger.info("Log file path (resolved): %s", log_file)
        if log_file.is_file():
            log_chunks = parse_log_file(log_file, project_root)
            logger.info("Log parser: %d chunk(s) from %s", len(log_chunks), log_file.name)
            chunks.extend(log_chunks)
        else:
            logger.warning("Log file does not exist or is not a file: %s", log_file)

    if not chunks:
        logger.warning("No chunks produced from %s", project_root)
        return 0

    # ── Debug: severity distribution ──────────────────────────────────────
    sev_counts = Counter(c.severity for c in chunks if c.source_type == "log")
    file_counts = Counter(c.file_name for c in chunks if c.file_name)
    logger.info(
        "Severity distribution: %s | Files indexed: %d | File breakdown: %s",
        dict(sev_counts),
        len(file_counts),
        dict(file_counts.most_common(10)),
    )

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    stored = add_chunks(chunks, embeddings, namespace=namespace)
    logger.info("Stored %d chunks in namespace %s", stored, namespace)
    return stored


def _run_github_ingestion(repo_url: str, branch: str, namespace: str) -> int:
    from core.ingestion.github_ingester import ingest_github_repo
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks

    chunks = ingest_github_repo(repo_url, branch=branch)
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    return add_chunks(chunks, embeddings, namespace=namespace)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/index", response_model=IndexResponse, tags=["ingestion"])
async def index_local(body: IndexLocalRequest) -> IndexResponse:
    """
    Read-only scan of a local project directory.
    Optionally also indexes a log file.
    """
    # ── Normalize and validate project path ──────────────────────────────
    resolved_path = _validate_path(body.path, kind="directory")

    # ── Normalize and validate optional log file path ────────────────────
    resolved_log: Optional[str] = None
    if body.log_path and body.log_path.strip():
        resolved_log = _validate_path(body.log_path, kind="file")

    namespace = body.namespace or _derive_namespace(resolved_path)

    try:
        stored = _run_local_ingestion(
            path=resolved_path,
            log_path=resolved_log,
            namespace=namespace,
        )
    except Exception as exc:
        logger.exception("Ingestion failed for path: %s", resolved_path)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ingestion failed: {exc}\n"
                "Ensure the folder exists and is accessible by the application."
            ),
        )

    return IndexResponse(
        status="ok",
        namespace=namespace,
        chunks_stored=stored,
        message=f"Indexed {stored} chunks from {resolved_path}.",
    )


@router.post("/index/github", response_model=IndexResponse, tags=["ingestion"])
async def index_github(body: IndexGitHubRequest) -> IndexResponse:
    """
    Clone a GitHub repository (shallow, read-only) and index its contents.
    For private repos, embed a token in the URL:
      https://<token>@github.com/owner/repo.git
    The token is never logged or stored.
    """
    from core.ingestion.github_ingester import validate_github_url
    if not validate_github_url(body.repo_url):
        raise HTTPException(status_code=400, detail="Invalid repo URL. Must be an HTTPS URL.")

    namespace = body.namespace or _derive_namespace(body.repo_url)

    try:
        stored = _run_github_ingestion(
            repo_url=body.repo_url,
            branch=body.branch,
            namespace=namespace,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GitHub ingestion failed: {exc}")

    return IndexResponse(
        status="ok",
        namespace=namespace,
        chunks_stored=stored,
        message=f"Indexed {stored} chunks from {body.repo_url}.",
    )


# ── Helpers for upload ────────────────────────────────────────────────────────

_ALLOWED_LOG_SUFFIXES = {".log", ".txt", ".json", ".csv", ".out", ".err"}
_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB per file


def _safe_suffix(filename: str) -> bool:
    """Allow code and log files. Block executables and archives."""
    p = Path(filename)
    return p.suffix.lower() not in {".exe", ".dll", ".so", ".zip", ".tar", ".gz", ".7z", ".bin"}


def _run_upload_ingestion(
    project_files: List[tuple[str, bytes]],
    log_files: List[tuple[str, bytes]],
    namespace: str,
) -> int:
    """
    Write uploaded files to a temp dir, run standard ingestion, clean up.
    """
    from core.ingestion.file_walker import walk_project
    from core.ingestion.log_parser import parse_log_file
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks
    from core.ingestion.chunker import Chunk

    with tempfile.TemporaryDirectory(prefix="bakup_upload_") as tmpdir:
        tmp = Path(tmpdir)
        src_dir = tmp / "src"
        logs_dir = tmp / "logs"
        src_dir.mkdir()
        logs_dir.mkdir()

        # Write project files
        for name, content in project_files:
            dest = src_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # Write log files
        for name, content in log_files:
            dest = logs_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # Ingest code
        chunks: list[Chunk] = []
        if any(src_dir.iterdir()):
            chunks.extend(walk_project(src_dir))

        # Ingest logs
        for log_file in logs_dir.rglob("*"):
            if log_file.is_file():
                chunks.extend(parse_log_file(log_file, tmp))

        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        return add_chunks(chunks, embeddings, namespace=namespace)


@router.post("/index/upload", response_model=IndexResponse, tags=["ingestion"])
async def index_upload(
    project_files: List[UploadFile] = File(default=[]),
    log_files: List[UploadFile] = File(default=[]),
    namespace: Optional[str] = Form(default=None),
    project_name: Optional[str] = Form(default=None),
) -> IndexResponse:
    """
    Upload local files via multipart form for indexing.

    - project_files: source code files (.py, .js, .ts, etc.)
    - log_files: log files (.log, .txt, .json)
    - namespace: optional override
    - project_name: optional project label (used to derive namespace)

    This endpoint is the preferred way to index when running in Docker,
    since the backend container cannot see arbitrary host paths.
    """
    if not project_files and not log_files:
        raise HTTPException(status_code=400, detail="No files uploaded. Attach project_files or log_files.")

    # Read all files into memory (capped at _MAX_UPLOAD_SIZE per file)
    proj_data: List[tuple[str, bytes]] = []
    log_data: List[tuple[str, bytes]] = []

    for f in project_files:
        if not _safe_suffix(f.filename or "unnamed"):
            continue
        content = await f.read()
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename} ({len(content)} bytes)")
        proj_data.append((f.filename or "unnamed", content))

    for f in log_files:
        content = await f.read()
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {f.filename} ({len(content)} bytes)")
        log_data.append((f.filename or "unnamed.log", content))

    label = project_name or "upload"
    ns = namespace or _derive_namespace(f"upload:{label}:{len(proj_data)}:{len(log_data)}")

    try:
        stored = _run_upload_ingestion(proj_data, log_data, ns)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload ingestion failed: {exc}")

    total_files = len(proj_data) + len(log_data)
    return IndexResponse(
        status="ok",
        namespace=ns,
        chunks_stored=stored,
        message=f"Indexed {stored} chunks from {total_files} uploaded file(s).",
    )
