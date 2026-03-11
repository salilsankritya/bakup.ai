"""
bakup_server.py — PyInstaller-compatible entry point for bakup.ai backend.

This replaces main.py's "main:app" uvicorn string import pattern
(which doesn't work in frozen executables) with a direct app reference.

Also serves the UI static files and manages runtime data directories.
"""

import os
import sys
import webbrowser
import threading

# ── Frozen-app path setup ─────────────────────────────────────────────────────
# When running as a PyInstaller bundle, sys._MEIPASS points to the temp
# extraction directory. We need to set cwd and paths accordingly.

def _get_app_dir():
    """Return the directory where the .exe lives (not the temp extraction dir)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

APP_DIR = _get_app_dir()

# Ensure runtime data directories exist alongside the executable
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

VECTORDB_DIR = os.path.join(DATA_DIR, "vectordb")
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "model-weights")
os.makedirs(VECTORDB_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Set environment variables before any bakup imports
os.environ.setdefault("BAKUP_ACCESS_KEY", "tango")
os.environ.setdefault("BAKUP_CHROMA_DIR", VECTORDB_DIR)
os.environ.setdefault("BAKUP_MODEL_CACHE_DIR", MODEL_CACHE_DIR)
os.environ.setdefault("BAKUP_HOST", "127.0.0.1")
os.environ.setdefault("BAKUP_PORT", "8000")

# If running frozen, add the backend source to sys.path
if getattr(sys, 'frozen', False):
    _backend_path = os.path.join(sys._MEIPASS, "backend")
    if os.path.isdir(_backend_path):
        sys.path.insert(0, _backend_path)
    else:
        sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ── Torch GPU stub (MetaPathFinder) ───────────────────────────────────────────
# The PyInstaller build excludes torch.cuda / xpu / mps / mtia to save ~700 MB.
# torch.__init__.py internally does `from torch import cuda`, so the stubs MUST
# be registered in sys.modules BEFORE `import torch` runs.  A MetaPathFinder
# intercepts any import under the excluded roots and returns a lightweight stub
# module.  The stub uses __getattr__ to safely handle ANY attribute access
# (e.g. torch.cuda.Event, torch.cuda.amp.autocast) without enumerating them.
import importlib, importlib.abc, importlib.machinery, types

_TORCH_STUB_ROOTS = frozenset({
    # GPU backends (torch.__init__ imports these unconditionally)
    "torch.cuda", "torch.xpu", "torch.mps", "torch.mtia",
    # Dev/compile modules excluded by PyInstaller spec (imported
    # conditionally by transformers — safe to return NullStub)
    "torch._dynamo", "torch._inductor", "torch._export",
    "torch.onnx", "torch.package",
})


class _NullStub:
    """Falsy callable/subscriptable/iterable placeholder.

    Any attribute access, call, or boolean test returns a safe no-op value.
    This prevents guard checks like ``if torch.cuda._is_compiled():`` from
    accidentally evaluating as True.
    """
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
    # bitwise / augmented operators (used by transformers: _is_tracing |= ...)
    def __or__(self, other):    return other
    def __ror__(self, other):   return other
    def __ior__(self, other):   return other
    def __and__(self, other):   return False
    def __rand__(self, other):  return False
    def __iand__(self, other):  return False
    def __int__(self):          return 0
    def __float__(self):        return 0.0
    def __index__(self):        return 0

_null = _NullStub()


class _TorchGPUStub(types.ModuleType):
    """Module stub that reports no GPU and dynamically handles any attribute."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []            # behave like a package
        self.__package__ = name
        self.__all__ = []

    # ── common fast-path attributes ──
    @staticmethod
    def is_available(): return False
    @staticmethod
    def device_count(): return 0
    @staticmethod
    def current_device(): return -1
    @staticmethod
    def get_device_name(*a, **kw): return ""
    FloatTensor = None
    HalfTensor = None
    DoubleTensor = None

    def __getattr__(self, name):
        """Return a safe falsy dummy for any attribute not explicitly defined."""
        # Only reject dunder attrs to signal unsupported protocols
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _NullStub()


class _TorchStubFinder(importlib.abc.MetaPathFinder):
    """Intercept imports of excluded torch GPU modules and return CPU-only stubs."""
    _bakup_torch_stub = True  # sentinel for idempotency checks

    def find_spec(self, fullname, path, target=None):
        if fullname in _TORCH_STUB_ROOTS or any(
            fullname.startswith(r + ".") for r in _TORCH_STUB_ROOTS
        ):
            return importlib.machinery.ModuleSpec(
                fullname, _TorchStubLoader(), is_package=True,
            )
        return None


class _TorchStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _TorchGPUStub(spec.name)

    def exec_module(self, module):
        pass


# Install the finder at the FRONT of sys.meta_path so it runs before the
# default finders — this ensures stubs are in place when `import torch` runs.
sys.meta_path.insert(0, _TorchStubFinder())

# ── Frozen-build dispatch fix ─────────────────────────────────────────────────
# In PyInstaller builds, tensor operations (normal_, uniform_, fill_, etc.)
# dispatch through the torch._refs decomposition path instead of the C++ ATen
# backend, hitting an assertion in _prims_common/wrappers.py.
#
# Since we ONLY load pre-trained models (SentenceTransformer with saved
# checkpoints), random weight initialisation is immediately overwritten from
# the saved state dict.  Making every nn.init function a harmless no-op
# completely sidesteps the broken dispatch — with zero impact on inference.
if getattr(sys, 'frozen', False):
    import torch as _torch
    import torch.nn.init as _init

    def _noop_init(tensor, *_a, **_kw):
        return tensor

    for _fn in (
        # internal helpers
        '_no_grad_normal_', '_no_grad_uniform_', '_no_grad_fill_',
        '_no_grad_zero_',
        # public API
        'uniform_', 'normal_', 'constant_', 'ones_', 'zeros_',
        'eye_', 'dirac_', 'xavier_uniform_', 'xavier_normal_',
        'kaiming_uniform_', 'kaiming_normal_', 'orthogonal_', 'sparse_',
        'trunc_normal_',
    ):
        if hasattr(_init, _fn):
            setattr(_init, _fn, _noop_init)

# ── Import and run ────────────────────────────────────────────────────────────
# Now import the actual app (this triggers access check + config load)
from core.access import check_access_key
check_access_key()

import config as _config
_config.settings = _config.load_settings()

# ── Port auto-detection ───────────────────────────────────────────────────────
# Must happen before FastAPI app construction so the CORS origin list
# and uvicorn bind use the same (possibly adjusted) port.
from core.net import resolve_port
_actual_port = resolve_port(_config.settings.host, _config.settings.port)
if _actual_port != _config.settings.port:
    os.environ["BAKUP_PORT"] = str(_actual_port)
    _config.settings = _config.load_settings()

# Import after config is loaded
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.embeddings.model_cache import ensure_models_downloaded
    from core.retrieval.vector_store import init_vector_store

    print(f"bakup: starting - data dir: {DATA_DIR}")
    ensure_models_downloaded()
    init_vector_store()
    print("bakup: ready.")
    print(f"bakup: open http://127.0.0.1:{settings.port} in your browser")
    yield
    print("bakup: shutting down.")


app = FastAPI(
    title="Bakup.ai",
    version="0.2.0",
    description="Local-first incident intelligence.",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

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

app.include_router(health_router)
app.include_router(index_router)
app.include_router(query_router)
app.include_router(llm_router)
app.include_router(debug_router)

# ── Serve UI static files ─────────────────────────────────────────────────────
# In frozen mode, UI files are bundled inside the executable
if getattr(sys, 'frozen', False):
    _ui_dir = os.path.join(sys._MEIPASS, "ui")
else:
    _ui_dir = os.path.join(os.path.dirname(__file__), "ui")

if os.path.isdir(_ui_dir):
    app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
    print(f"bakup: serving UI from {_ui_dir}")


# ── Auto-open browser ────────────────────────────────────────────────────────
def _open_browser(port: int):
    """Poll the health endpoint until the server is ready, then open the browser."""
    import time
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}"
    health_url = f"{url}/health"
    max_wait = 120          # seconds — first run downloads ~90 MB model
    poll_interval = 1       # seconds between health checks
    elapsed = 0

    # Wait for uvicorn to start accepting connections
    time.sleep(1)

    while elapsed < max_wait:
        try:
            req = urllib.request.urlopen(health_url, timeout=2)
            if req.status == 200:
                print(f"bakup: server ready after {elapsed}s - opening {url}")
                webbrowser.open(url)
                return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval)
        elapsed += poll_interval

    # Fallback: open anyway so the user sees something
    print(f"bakup: server not ready after {max_wait}s - opening browser anyway")
    webbrowser.open(url)


if __name__ == "__main__":
    # Re-validate port right before starting (belt-and-suspenders)
    from core.net import is_port_available, find_free_port
    if not is_port_available(settings.host, settings.port):
        old_port = settings.port
        new_port = find_free_port(settings.host, start=old_port + 1)
        print(f"bakup: port {old_port} is in use -- switching to {new_port}")
        os.environ["BAKUP_PORT"] = str(new_port)
        _config.settings = _config.load_settings()

    # Launch browser in background thread (uses final port)
    browser_thread = threading.Thread(target=_open_browser, args=(settings.port,), daemon=True)
    browser_thread.start()

    # Run server with port-conflict retry as safety net
    max_port_retries = 5
    port = settings.port

    for attempt in range(max_port_retries):
        try:
            uvicorn.run(
                app,
                host=settings.host,
                port=port,
                log_level=settings.log_level,
                access_log=True,
            )
            break   # clean exit
        except OSError as e:
            # Errno 10048 (Windows) / 98 (Linux): address already in use
            if e.errno in (10048, 98) and attempt < max_port_retries - 1:
                port += 1
                print(f"bakup: port {port - 1} is in use -- retrying on {port}")
                os.environ["BAKUP_PORT"] = str(port)
                _config.settings = _config.load_settings()
            else:
                raise
