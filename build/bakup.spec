# -*- mode: python ; coding: utf-8 -*-
"""
bakup.spec — PyInstaller spec file for bakup.ai
─────────────────────────────────────────────────────────────────────────────
Compiles the backend into a single-folder distribution.
UI static files are bundled as data.

Usage:
    cd build
    pyinstaller bakup.spec --noconfirm
"""

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# ── Paths ─────────────────────────────────────────────────────────────────────
# SPECPATH in PyInstaller resolves to cwd when spec is a relative path.
# Use os.getcwd() which is the build/ directory when run correctly.
SPEC_DIR = os.getcwd()
# If cwd ends with "build", we're in the right place; otherwise find it
if not SPEC_DIR.endswith("build"):
    # Try to locate build/ relative to SPECPATH
    _candidate = os.path.dirname(os.path.abspath(SPECPATH))
    if os.path.basename(_candidate) == "build":
        SPEC_DIR = _candidate
    else:
        SPEC_DIR = os.path.join(_candidate, "build")

PROJECT_ROOT = os.path.dirname(SPEC_DIR)
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
UI_DIR = os.path.join(PROJECT_ROOT, "ui")
ENTRY_POINT = os.path.join(SPEC_DIR, "bakup_server.py")

# Verify paths
for label, path in [("SPEC_DIR", SPEC_DIR), ("PROJECT_ROOT", PROJECT_ROOT),
                     ("BACKEND_DIR", BACKEND_DIR), ("UI_DIR", UI_DIR),
                     ("ENTRY_POINT", ENTRY_POINT)]:
    exists = os.path.exists(path)
    print(f"  {label}: {path} (exists={exists})")
    if not exists:
        raise FileNotFoundError(f"Build path not found: {label}={path}")

# ── Hidden imports ────────────────────────────────────────────────────────────
# Modules that PyInstaller can't detect through static analysis
hidden_imports = [
    # FastAPI & web
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "multipart",
    "python_multipart",

    # Backend modules (dynamically imported)
    "api",
    "api.routes",
    "api.routes.health",
    "api.routes.index",
    "api.routes.query",
    "api.routes.llm_config",
    "api.routes.debug",
    "api.routes.download",
    "core",
    "core.access",
    "core.analysis",
    "core.analysis.clusters",
    "core.analysis.confidence",
    "core.analysis.file_aggregation",
    "core.analysis.trends",
    "core.classifier",
    "core.classifier.query_classifier",
    "core.embeddings",
    "core.embeddings.embedder",
    "core.embeddings.model_cache",
    "core.ingestion",
    "core.ingestion.chunker",
    "core.ingestion.file_walker",
    "core.ingestion.github_ingester",
    "core.ingestion.log_parser",
    "core.llm",
    "core.llm.config_store",
    "core.llm.llm_service",
    "core.llm.prompt_templates",
    "core.retrieval",
    "core.retrieval.models",
    "core.retrieval.rag",
    "core.retrieval.ranker",
    "core.retrieval.vector_store",
    "config",

    # sentence-transformers
    "sentence_transformers",

    # chromadb
    "chromadb",
    "chromadb.config",
    "chromadb.api",
    "chromadb.db",
    "chromadb.db.impl",
    "chromadb.db.impl.sqlite",

    # torch CPU
    "torch",

    # Other
    "chardet",
    "httpx",
    "gitpython",
    "git",
    "pydantic",
    "pydantic_core",
    "dotenv",
    "numpy",
    "tqdm",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
]

# Collect all submodules for complex packages
hidden_imports += collect_submodules("chromadb")
hidden_imports += collect_submodules("sentence_transformers")
hidden_imports += collect_submodules("uvicorn")
hidden_imports += collect_submodules("fastapi")
hidden_imports += collect_submodules("starlette")
hidden_imports += collect_submodules("pydantic")
hidden_imports += collect_submodules("pydantic_core")

# ── Data files ────────────────────────────────────────────────────────────────
datas = [
    # Bundle the entire backend source tree (needed for module imports)
    (BACKEND_DIR, "backend"),
    # Bundle UI static files
    (UI_DIR, "ui"),
]

# Collect data files from packages that need them
chromadb_datas, chromadb_binaries, chromadb_hiddenimports = collect_all("chromadb")
st_datas, st_binaries, st_hiddenimports = collect_all("sentence_transformers")

datas += chromadb_datas
datas += st_datas
hidden_imports += chromadb_hiddenimports
hidden_imports += st_hiddenimports

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [ENTRY_POINT],
    pathex=[BACKEND_DIR],
    binaries=chromadb_binaries + st_binaries,
    datas=datas,
    hiddenimports=list(set(hidden_imports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude GUI toolkits we don't need
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "jupyter",
        "notebook",
        # Exclude test frameworks
        "pytest",
        "_pytest",
    ],
    noarchive=False,
)

# ── Build ─────────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="bakup-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console app so users can see server logs
    icon=None,     # Add icon later: icon='../assets/bakup.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="bakup-server",
)
