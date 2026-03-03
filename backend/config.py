"""
bakup/config.py
─────────────────────────────────────────────────────────────────────────────
Application settings loaded from environment variables.

All settings have explicit types and documented defaults.
No setting silently falls back to an insecure value.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # ── Access control ────────────────────────────────────────────────────────
    # Validated at startup by core/access.py — not re-read here.
    access_key: str = field(default="")

    # ── Project ingestion ─────────────────────────────────────────────────────
    # Absolute path to a local project directory Bakup is allowed to read.
    # Optional — if not set, users can upload files through the browser UI.
    project_path: Path | None = None

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"   # Localhost only. Never bind to 0.0.0.0 in preview.
    port: int = 8000
    log_level: str = "info"

    # ── Vector database ───────────────────────────────────────────────────────
    chroma_persist_dir: Path = field(default_factory=lambda: Path("vectordb"))
    chroma_collection: str = "bakup"

    # ── Embedding model ───────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    model_cache_dir: Path = field(default_factory=lambda: Path("model-weights"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model_file: str = "llama-3.2-3b-instruct.Q4_K_M.gguf"
    llm_context_window: int = 4096
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1   # Low temp — we want citation accuracy, not creativity.

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    confidence_threshold: float = 0.35  # Results below this are labeled low-confidence.


def _require(name: str, value: str, hint: str = "") -> str:
    if not value.strip():
        msg = f"\nbakup: Required environment variable {name!r} is not set."
        if hint:
            msg += f"\n  {hint}"
        print(msg, file=sys.stderr)
        sys.exit(1)
    return value


def load_settings() -> Settings:
    """
    Build Settings from environment. Call once at startup, after check_access_key().
    """
    # BAKUP_PROJECT_PATH is optional — users can upload files via the browser.
    raw_project_path = os.environ.get("BAKUP_PROJECT_PATH", "").strip()
    project_path: Path | None = None

    if raw_project_path:
        project_path = Path(raw_project_path).resolve()
        if not project_path.exists():
            print(
                f"\nbakup: BAKUP_PROJECT_PATH does not exist: {project_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not project_path.is_dir():
            print(
                f"\nbakup: BAKUP_PROJECT_PATH must be a directory, got: {project_path}",
                file=sys.stderr,
            )
            sys.exit(1)

    return Settings(
        access_key=os.environ.get("BAKUP_ACCESS_KEY", ""),
        project_path=project_path,
        host=os.environ.get("BAKUP_HOST", "127.0.0.1"),
        port=int(os.environ.get("BAKUP_PORT", "8000")),
        log_level=os.environ.get("BAKUP_LOG_LEVEL", "info").lower(),
        chroma_persist_dir=Path(os.environ.get("BAKUP_CHROMA_DIR", "vectordb")),
        chroma_collection=os.environ.get("BAKUP_CHROMA_COLLECTION", "bakup"),
        embedding_model=os.environ.get("BAKUP_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        model_cache_dir=Path(os.environ.get("BAKUP_MODEL_CACHE_DIR", "model-weights")),
        llm_model_file=os.environ.get("BAKUP_LLM_MODEL_FILE", "llama-3.2-3b-instruct.Q4_K_M.gguf"),
        llm_context_window=int(os.environ.get("BAKUP_LLM_CONTEXT_WINDOW", "4096")),
        llm_max_tokens=int(os.environ.get("BAKUP_LLM_MAX_TOKENS", "512")),
        llm_temperature=float(os.environ.get("BAKUP_LLM_TEMPERATURE", "0.1")),
        retrieval_top_k=int(os.environ.get("BAKUP_RETRIEVAL_TOP_K", "8")),
        confidence_threshold=float(os.environ.get("BAKUP_CONFIDENCE_THRESHOLD", "0.35")),
    )


# Module-level singleton — imported by all other modules.
settings: Settings = None  # type: ignore[assignment]
# Populated by main.py after startup checks pass.
