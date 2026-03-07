"""
core/embeddings/embedder.py
─────────────────────────────────────────────────────────────────────────────
Sentence-Transformers wrapper for local embeddings.

- Model is loaded once at module level and reused for the lifetime of the process.
- No API calls. No network access after the initial model download.
- Thread-safe: SentenceTransformer.encode() releases the GIL internally.

Model weights are cached by HuggingFace's local cache mechanism
(~/.cache/huggingface or BAKUP_MODEL_CACHE_DIR via HF_HOME env override).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np


# ── Lazy singleton ─────────────────────────────────────────────────────────────

_model = None


def _ensure_torch_cuda_stub():
    """Ensure torch accelerator modules are importable (stubs for CPU-only builds)."""
    import importlib, types as _t, sys as _sys
    try:
        import torch
    except ImportError:
        return
    for mod_name in ("torch.cuda", "torch.xpu", "torch.mps", "torch.mtia"):
        attr = mod_name.split(".")[1]
        if importlib.util.find_spec(mod_name) is not None:
            continue
        stub = _t.ModuleType(mod_name)
        stub.is_available = lambda: False
        stub.device_count = lambda: 0
        stub.current_device = lambda: -1
        stub.get_device_name = lambda *a, **kw: ""
        stub.FloatTensor = None
        _sys.modules[mod_name] = stub
        if not hasattr(torch, attr):
            setattr(torch, attr, stub)


def _get_model():
    """Load the embedding model once. Subsequent calls return the cached instance."""
    global _model
    if _model is None:
        _ensure_torch_cuda_stub()
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("BAKUP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        cache_dir = os.environ.get("BAKUP_MODEL_CACHE_DIR", "model-weights")

        # Point HuggingFace cache at our controlled directory
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_HOME", cache_dir)

        print(f"bakup: loading embedding model '{model_name}' from cache '{cache_dir}'")
        _model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print(f"bakup: embedding model ready (dim={_model.get_sentence_embedding_dimension()})")

    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed a list of strings and return a list of float vectors.

    Args:
        texts:      Non-empty list of strings to embed.
        batch_size: Number of texts processed per forward pass.

    Returns:
        List of float lists, one per input string.
        All vectors have the same dimension (384 for all-MiniLM-L6-v2).
    """
    if not texts:
        return []

    model = _get_model()
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalize → cosine similarity == dot product
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string. Convenience wrapper around embed_texts.
    """
    results = embed_texts([query])
    if not results:
        raise ValueError("Embedding returned empty result for query.")
    return results[0]


def embedding_dimension() -> int:
    """Return the vector dimension of the loaded model."""
    return _get_model().get_sentence_embedding_dimension()
