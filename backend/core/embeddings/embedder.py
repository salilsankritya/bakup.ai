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
    """Pre-register stubs for excluded torch GPU modules (frozen builds only).

    In PyInstaller builds, torch.cuda is excluded to save space.  But
    torch.__init__.py does ``from torch import cuda`` on import, which would
    crash with 'No module named torch.cuda'.  A MetaPathFinder installed
    BEFORE the first ``import torch`` intercepts those imports and returns
    lightweight stubs that report no GPU available.

    In dev mode (not frozen), the real torch.cuda is available, so this
    function is a no-op.
    """
    import sys as _sys

    # Only needed inside a frozen (PyInstaller) binary
    if not getattr(_sys, 'frozen', False):
        return

    # Idempotency: skip if already installed
    if any(getattr(f, '_bakup_torch_stub', False) for f in _sys.meta_path):
        return

    import importlib, importlib.abc, importlib.machinery, types as _t

    _ROOTS = frozenset({
        # GPU backends (torch.__init__ imports these unconditionally)
        "torch.cuda", "torch.xpu", "torch.mps", "torch.mtia",
        # Dev/compile modules excluded by PyInstaller spec
        "torch._dynamo", "torch._inductor", "torch._export",
        "torch.onnx", "torch.package",
    })

    class _NullStub:
        """Falsy callable placeholder for dynamic attribute access."""
        __slots__ = ()
        def __bool__(self):         return False
        def __call__(self, *a, **kw): return _NullStub()
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _NullStub()
        def __iter__(self):         return iter(())
        def __len__(self):          return 0
        def __enter__(self):        return self
        def __exit__(self, *a):     return None
        def __repr__(self):         return "<NullStub>"
        def __getitem__(self, k):   return _NullStub()
        def __or__(self, other):    return other
        def __ror__(self, other):   return other
        def __ior__(self, other):   return other
        def __and__(self, other):   return False
        def __rand__(self, other):  return False
        def __iand__(self, other):  return False
        def __int__(self):          return 0
        def __float__(self):        return 0.0
        def __index__(self):        return 0

    class _GPUStub(_t.ModuleType):
        """Stub that reports no GPU and dynamically handles any attr access."""
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__package__ = name
            self.__all__ = []
        @staticmethod
        def is_available(): return False
        @staticmethod
        def device_count(): return 0
        @staticmethod
        def current_device(): return -1
        @staticmethod
        def get_device_name(*a, **kw): return ""
        FloatTensor = None
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _NullStub()

    class _Finder(importlib.abc.MetaPathFinder):
        _bakup_torch_stub = True
        def find_spec(self, fullname, path, target=None):
            if fullname in _ROOTS or any(fullname.startswith(r + ".") for r in _ROOTS):
                return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
            return None

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return _GPUStub(spec.name)
        def exec_module(self, module):
            pass

    _sys.meta_path.insert(0, _Finder())


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
