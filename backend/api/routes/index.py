"""
api/routes/index.py
─────────────────────────────────────────────────────────────────────────────
POST /index          — ingest a local directory (path on the host/container)
POST /index/github   — ingest a GitHub repository (public or token-authenticated)
POST /index/upload   — ingest uploaded files via multipart form data
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from pydantic import BaseModel

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

def _derive_namespace(path: str) -> str:
    """Stable namespace from a path string."""
    return hashlib.sha256(path.strip().encode()).hexdigest()[:24]


def _run_local_ingestion(path: str, log_path: Optional[str], namespace: str) -> int:
    """
    Walk a local project, parse any log file, embed all chunks,
    and store them in ChromaDB. Returns total chunks stored.
    """
    from core.ingestion.file_walker import walk_project
    from core.ingestion.log_parser import parse_log_file
    from core.embeddings.embedder import embed_texts
    from core.retrieval.vector_store import add_chunks
    from core.ingestion.chunker import Chunk

    project_root = Path(path).resolve()
    chunks: list[Chunk] = list(walk_project(project_root))

    if log_path:
        log_file = Path(log_path).resolve()
        if log_file.is_file():
            chunks.extend(parse_log_file(log_file, project_root))

    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    return add_chunks(chunks, embeddings, namespace=namespace)


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
    project_path = Path(body.path)
    if not project_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {body.path}")
    if not project_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {body.path}")

    namespace = body.namespace or _derive_namespace(body.path)

    try:
        stored = _run_local_ingestion(
            path=body.path,
            log_path=body.log_path,
            namespace=namespace,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return IndexResponse(
        status="ok",
        namespace=namespace,
        chunks_stored=stored,
        message=f"Indexed {stored} chunks from {body.path}.",
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
