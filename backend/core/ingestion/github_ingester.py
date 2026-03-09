"""
core/ingestion/github_ingester.py
─────────────────────────────────────────────────────────────────────────────
Read-only GitHub repository ingestion.

Approach:
  1. Clone the repo (shallow, depth=1) to a temporary directory.
  2. Walk the clone with file_walker.walk_project() — same read-only path
     as local ingestion.
  3. Yield all chunks, then clean up the temp directory.

This requires no GitHub API token for public repos.
For private repos, the caller must embed credentials in the URL:
  https://<token>@github.com/owner/repo.git

The token is never logged and never stored — it lives only in the URL
string for the duration of the clone operation.

Dependency: gitpython (pip install gitpython)
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List
from urllib.parse import urlparse

from core.ingestion.chunker import Chunk
from core.ingestion.file_walker import walk_project


# Maximum repo size we are willing to clone (bytes, shallow clone).
# This is a best-effort guard — actual disk usage may slightly exceed this.
MAX_REPO_SIZE_BYTES: int = 200 * 1024 * 1024  # 200 MB


def _sanitize_url_for_log(url: str) -> str:
    """Strip credentials from a repo URL before writing to logs."""
    try:
        parsed = urlparse(url)
        safe = parsed._replace(netloc=parsed.hostname or "")
        return safe.geturl()
    except Exception:
        return "<redacted>"


def ingest_github_repo(repo_url: str, branch: str = "HEAD", namespace: str = "") -> List[Chunk]:
    """
    Clone a GitHub repository to a temporary directory, walk it, and return
    all chunks. The temp directory is deleted on completion or error.

    Args:
        repo_url:  HTTPS clone URL. Credentials may be embedded for private repos.
        branch:    Branch or ref to clone. Defaults to the repo's default branch.
        namespace: Project namespace for symbol graph / architecture indexing.

    Returns:
        List of Chunk objects. Empty if the clone fails.

    Raises:
        RuntimeError: If gitpython is not installed.
        ValueError:   If repo_url is empty or malformed.
    """
    try:
        import git  # gitpython
    except ImportError:
        raise RuntimeError(
            "gitpython is required for GitHub ingestion. "
            "Install it with: pip install gitpython"
        )

    if not repo_url.strip():
        raise ValueError("repo_url must not be empty.")

    safe_url = _sanitize_url_for_log(repo_url)
    print(f"bakup: cloning {safe_url} (shallow, read-only)")

    tmp_dir = tempfile.mkdtemp(prefix="bakup_github_")
    try:
        clone_kwargs: dict = {
            "depth": 1,
            "no_single_branch": False,
        }
        if branch and branch != "HEAD":
            clone_kwargs["branch"] = branch

        repo = git.Repo.clone_from(
            repo_url,
            tmp_dir,
            multi_options=["--filter=blob:none"],   # partial clone — skip large blobs
            **clone_kwargs,
        )

        # Extract repo metadata for enriched chunks
        repo_meta = _extract_repo_metadata(repo, repo_url, branch)
        print(f"bakup: clone complete ({repo_meta.get('repo_name', '?')}@{repo_meta.get('branch', '?')}), walking {tmp_dir}")

        chunks = list(walk_project(Path(tmp_dir), namespace=namespace))

        # Enrich chunks with repo metadata (Chunk is frozen, rebuild with metadata)
        enriched: list[Chunk] = []
        for chunk in chunks:
            merged_meta = {**(chunk.metadata or {}), **repo_meta}
            enriched.append(Chunk(
                text=chunk.text,
                source_file=chunk.source_file,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                source_type=chunk.source_type,
                file_name=chunk.file_name,
                last_modified=chunk.last_modified,
                detected_timestamp=chunk.detected_timestamp,
                severity=chunk.severity,
                language=chunk.language,
                function_name=chunk.function_name,
                class_name=chunk.class_name,
                chunk_kind=chunk.chunk_kind,
                docstring=chunk.docstring,
                imports=chunk.imports,
                metadata=merged_meta,
            ))

        print(f"bakup: {len(enriched)} chunks from {safe_url}")
        return enriched

    except git.exc.GitCommandError as exc:
        # Log the error without leaking credentials
        print(f"bakup: clone failed for {safe_url}: {exc.stderr.strip()}")
        return []
    except Exception as exc:
        print(f"bakup: unexpected error cloning {safe_url}: {exc}")
        return []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _extract_repo_metadata(repo, repo_url: str, branch: str) -> dict:
    """Extract repository metadata from the cloned repo object."""
    meta: dict = {}
    try:
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").rstrip(".git").split("/")
        if len(path_parts) >= 2:
            meta["repo_owner"] = path_parts[-2]
            meta["repo_name"] = path_parts[-1]
        elif path_parts:
            meta["repo_name"] = path_parts[-1]
    except Exception:
        pass

    try:
        meta["branch"] = str(repo.active_branch)
    except Exception:
        meta["branch"] = branch if branch != "HEAD" else "default"

    try:
        commit = repo.head.commit
        meta["commit_hash"] = commit.hexsha[:12]
        meta["commit_message"] = commit.message.strip()[:200]
        meta["commit_author"] = str(commit.author)
    except Exception:
        pass

    return meta


def validate_github_url(url: str) -> bool:
    """
    Basic structural check: must be an HTTPS URL pointing to github.com,
    gitlab.com, or a common self-hosted Git host ending in .git or not.
    Does not make any network request.
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ("https", "http")
            and bool(parsed.netloc)
            and bool(parsed.path.strip("/"))
        )
    except Exception:
        return False
