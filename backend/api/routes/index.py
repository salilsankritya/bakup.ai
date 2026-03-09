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
                f"Invalid path. The {kind} does not exist.\n"
                "Ensure the full absolute path is provided and the "
                f"{kind} is accessible by the application."
            ),
        )

    # ── Type check ───────────────────────────────────────────────────────
    if kind == "directory" and not os.path.isdir(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                "Path is not a directory.\n"
                "Please provide the path to a project folder, not a single file."
            ),
        )
    if kind == "file" and not os.path.isfile(resolved):
        raise HTTPException(
            status_code=400,
            detail=(
                "Path is not a file.\n"
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
            raise HTTPException(
                status_code=400,
                detail=(
                    "Path is outside the allowed mounted volumes.\n"
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
      - Symbol graph building
      - Architecture summary generation
    """
    from core.ingestion.file_walker import walk_project
    from core.ingestion.log_parser import parse_log_file
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks
    from core.ingestion.chunker import Chunk
    from collections import Counter

    project_root = Path(path).resolve()
    logger.info("Ingesting project root: %s", project_root)
    chunks: list[Chunk] = list(walk_project(project_root, namespace=namespace))
    logger.info("Walk complete: %d chunk(s) from source files", len(chunks))

    if log_path:
        log_target = Path(log_path).resolve()
        logger.info("Log path (resolved): %s", log_target)
        if log_target.is_file():
            log_chunks = parse_log_file(log_target, project_root)
            logger.info("Log parser: %d chunk(s) from %s", len(log_chunks), log_target.name)
            chunks.extend(log_chunks)
        elif log_target.is_dir():
            # Recursively parse all log files in the directory
            log_file_count = 0
            for log_file in sorted(log_target.rglob("*")):
                if log_file.is_file() and log_file.suffix.lower() in {".log", ".out", ".txt", ".err"}:
                    try:
                        lc = parse_log_file(log_file, project_root)
                        chunks.extend(lc)
                        log_file_count += 1
                        logger.info("Log parser: %d chunk(s) from %s", len(lc), log_file.name)
                    except Exception as exc:
                        logger.warning("Skipping log file %s: %s", log_file.name, exc)
            logger.info("Log folder: %d files parsed under %s", log_file_count, log_target)
        else:
            logger.warning("Log path does not exist: %s", log_target)

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

    # ── Build architecture summary ────────────────────────────────────────
    try:
        from core.ingestion.symbol_graph import get_graph
        from core.analysis.architecture import build_architecture_summary
        from core.ingestion.code_parser import parse_file, detect_language
        from collections import Counter as ImportCounter

        graph = get_graph(namespace)
        file_paths = list(set(c.source_file for c in chunks if c.source_type == "code"))

        # Collect import counts from the symbol graph
        import_counter = ImportCounter()
        for file_path, imports in graph._file_imports.items():
            for imp in imports:
                import_counter[imp] += 1

        # Collect units by file (from symbol graph nodes)
        units_by_file = {}
        for file_path in file_paths:
            symbols = graph.symbols_in_file(file_path)
            if symbols:
                from core.ingestion.code_parser import CodeUnit
                units_by_file[file_path] = [
                    CodeUnit(
                        kind=s.kind, name=s.name, text="",
                        start_line=s.line_start, end_line=s.line_end,
                        language=s.language,
                    )
                    for s in symbols
                ]

        arch_summary = build_architecture_summary(
            file_paths=file_paths,
            units_by_file=units_by_file,
            import_counter=import_counter,
            project_name=Path(path).name,
            namespace=namespace,
        )
        logger.info(
            "Architecture summary: %d modules, %d entry points, %d deps",
            len(arch_summary.modules), len(arch_summary.entry_points),
            len(arch_summary.core_dependencies),
        )
        logger.info(
            "Symbol graph: %d nodes, %d edges",
            graph.node_count, graph.edge_count,
        )
    except Exception as exc:
        logger.warning("Architecture summary failed: %s", exc)

    return stored


def _run_github_ingestion(repo_url: str, branch: str, namespace: str) -> int:
    from core.ingestion.github_ingester import ingest_github_repo
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks

    chunks = ingest_github_repo(repo_url, branch=branch, namespace=namespace)
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

    # ── Normalize and validate optional log path (file or directory) ────
    resolved_log: Optional[str] = None
    if body.log_path and body.log_path.strip():
        raw_log = _normalize_path(body.log_path)
        if os.path.isdir(raw_log):
            resolved_log = _validate_path(body.log_path, kind="directory")
        else:
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
        logger.exception("GitHub ingestion failed for %s", body.repo_url)
        raise HTTPException(status_code=500, detail=f"GitHub ingestion failed: {exc}")

    if stored == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No chunks were produced from this repository. "
                "Possible causes:\n"
                "• The repository could not be cloned (check the URL and branch).\n"
                "• The repository is empty or contains no supported file types.\n"
                "• For private repos, embed a personal access token in the URL: "
                "https://<token>@github.com/owner/repo.git"
            ),
        )

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


def _sanitize_filename(raw: str) -> str:
    """
    Strip directory traversal components from an uploaded filename.

    Takes only the final component (e.g. '../../evil.py' → 'evil.py'),
    rejects empty or hidden names, and normalises separators.
    """
    # Use PurePosixPath to handle both / and \ separators
    from pathlib import PurePosixPath
    name = PurePosixPath(raw.replace("\\", "/")).name
    # Reject empty, dot-only, or hidden filenames
    if not name or name.startswith(".") or name in (".", ".."):
        name = "unnamed"
    return name


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

        # Write project files (filenames already sanitised)
        for name, content in project_files:
            dest = (src_dir / name).resolve()
            # Confinement check — dest must stay inside src_dir
            try:
                dest.relative_to(src_dir.resolve())
            except ValueError:
                logger.warning("Upload path escape blocked: %s", name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)

        # Write log files (filenames already sanitised)
        for name, content in log_files:
            dest = (logs_dir / name).resolve()
            try:
                dest.relative_to(logs_dir.resolve())
            except ValueError:
                logger.warning("Upload path escape blocked: %s", name)
                continue
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
        safe_name = _sanitize_filename(f.filename or "unnamed")
        if not _safe_suffix(safe_name):
            continue
        content = await f.read()
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large ({len(content)} bytes, max {_MAX_UPLOAD_SIZE}).")
        proj_data.append((safe_name, content))

    for f in log_files:
        safe_name = _sanitize_filename(f.filename or "unnamed.log")
        content = await f.read()
        if len(content) > _MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large ({len(content)} bytes, max {_MAX_UPLOAD_SIZE}).")
        log_data.append((safe_name, content))

    label = project_name or "upload"
    ns = namespace or _derive_namespace(f"upload:{label}:{len(proj_data)}:{len(log_data)}")

    try:
        stored = _run_upload_ingestion(proj_data, log_data, ns)
    except Exception as exc:
        logger.error("Upload ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail="Upload ingestion failed. Check server logs for details.")

    total_files = len(proj_data) + len(log_data)
    return IndexResponse(
        status="ok",
        namespace=ns,
        chunks_stored=stored,
        message=f"Indexed {stored} chunks from {total_files} uploaded file(s).",
    )
