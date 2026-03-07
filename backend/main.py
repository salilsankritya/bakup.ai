"""
bakup/main.py
─────────────────────────────────────────────────────────────────────────────
Application entry point.

Startup order is strict:
    1. Access key check       — exits before anything else if key is invalid
    2. Settings load          — reads and validates all env vars
    3. FastAPI app creation   — registers routes and middleware
    4. uvicorn server start   — binds to localhost only

Nothing is imported from the rest of the codebase before step 1 completes.
This ensures no route, DB connection, or model load runs on an unauthorised
instance.
"""

# ── Step 1: Access key check ──────────────────────────────────────────────────
# Import and call before anything else. sys.exit(1) on failure.
from core.access import check_access_key
check_access_key()

# ── Step 2: Load and validate settings ───────────────────────────────────────
import config as _config
_config.settings = _config.load_settings()

# ── Step 2a: Port auto-detection ─────────────────────────────────────────────
# Must happen before FastAPI app construction so the CORS origin list and
# uvicorn bind use the same (possibly adjusted) port.
from core.net import resolve_port
_resolved_port = resolve_port(_config.settings.host, _config.settings.port)
if _resolved_port != _config.settings.port:
    # Rebuild settings with the new port so everything downstream sees it
    import os as _os
    _os.environ["BAKUP_PORT"] = str(_resolved_port)
    _config.settings = _config.load_settings()

# ── Step 3: Build the FastAPI application ─────────────────────────────────────
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before first request) and once at shutdown.
    Model loading and ChromaDB initialisation happen here so that
    the /health route can accurately reflect readiness.
    """
    # Deferred imports — only after the access check has passed.
    from core.embeddings.model_cache import ensure_models_downloaded
    from core.retrieval.vector_store import init_vector_store

    print(f"bakup: starting - project path: {settings.project_path or '(none - use browser upload)'}")
    ensure_models_downloaded()
    init_vector_store()
    print("bakup: ready.")

    yield

    # Shutdown — nothing to flush for now.
    print("bakup: shutting down.")


app = FastAPI(
    title="Bakup.ai",
    version="0.1.0",
    description="Local-first incident intelligence. All processing runs on this machine.",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allows localhost origins for local dev and any production domains
# configured via the BAKUP_CORS_ORIGINS env var (comma-separated).
#
# Example:  BAKUP_CORS_ORIGINS=https://bakup.example.com,https://beta.bakup.ai
import os as _os_cors
_extra_origins = [
    o.strip()
    for o in _os_cors.environ.get("BAKUP_CORS_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        f"http://localhost:{settings.port}",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1",
        f"http://127.0.0.1:{settings.port}",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
        *_extra_origins,
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
from api.routes.health     import router as health_router
from api.routes.index      import router as index_router
from api.routes.query      import router as query_router
from api.routes.llm_config import router as llm_router
from api.routes.debug      import router as debug_router
from api.routes.download   import router as download_router

app.include_router(health_router)
app.include_router(index_router)
app.include_router(query_router)
app.include_router(llm_router)
app.include_router(debug_router)
app.include_router(download_router)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,           # 127.0.0.1 — never exposed externally
        port=settings.port,
        log_level=settings.log_level,
        reload=False,                 # No hot-reload in preview builds
        access_log=True,
    )
