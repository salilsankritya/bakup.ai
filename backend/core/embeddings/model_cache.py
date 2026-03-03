"""
core/embeddings/model_cache.py
─────────────────────────────────────────────────────────────────────────────
Ensures local model weights are present before the app begins serving requests.
Downloads once to BAKUP_MODEL_CACHE_DIR; subsequent starts are instant.
"""
import sys
from pathlib import Path

from config import settings


def ensure_models_downloaded() -> None:
    """
    Check that required model weights are present in model_cache_dir.

    The LLM model is OPTIONAL. If absent, the RAG pipeline runs in
    extractive mode (returns best-matching chunks without LLM synthesis).
    This allows the backend to start and serve requests without a 3GB download.

    The embedding model (sentence-transformers) is downloaded automatically
    by the library on first use — no manual step required.
    """
    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    llm_path = cache_dir / settings.llm_model_file
    if not llm_path.exists():
        print(
            f"bakup: LLM model not found at {llm_path} — running in extractive mode.\n"
            f"  To enable LLM-augmented answers:\n"
            f"    bash scripts/download-models.sh",
            file=sys.stderr,
        )
        # Non-fatal: extractive mode is fully functional for the Developer Preview.
        return

    print(f"bakup: LLM model weights OK ({llm_path.name})")
