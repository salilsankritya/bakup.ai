# bakup.ai — Build Guide

How to generate the Windows installer for bakup.ai.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://www.python.org/downloads/) |
| PyInstaller | 6.x | `pip install pyinstaller` |
| Inno Setup 6 | 6.x | [jrsoftware.org/isdl.php](https://jrsoftware.org/isdl.php) |
| pip packages | per `backend/requirements.txt` | `pip install -r backend/requirements.txt` |

---

## Build Pipeline

The build produces:

```
dist/
├── bakup-server/          ← PyInstaller output (compiled binary + deps)
│   ├── bakup-server.exe   ← Main executable
│   ├── ui/                ← Bundled UI static files
│   └── ...                ← Python runtime, torch, chromadb, etc.
├── bakup-ai-installer.exe ← Inno Setup installer (distributable)
downloads/
└── bakup-ai-installer.exe ← Copy for website download
```

### One-Command Build

```powershell
.\build\build.ps1
```

This runs all five steps automatically:
1. Checks prerequisites (Python, PyInstaller, Inno Setup)
2. Cleans previous builds
3. Compiles backend with PyInstaller (~5-15 min first run)
4. Packages into installer with Inno Setup
5. Copies installer to `downloads/` for the website

### Build Options

```powershell
# Skip PyInstaller (reuse existing compiled binary)
.\build\build.ps1 -SkipPyInstaller

# Skip Inno Setup (compile binary only, no installer)
.\build\build.ps1 -SkipInstaller

# Verbose PyInstaller output
.\build\build.ps1 -Verbose
```

---

## Manual Build Steps

### Step 1: Compile Backend

```powershell
cd build
pyinstaller bakup.spec --noconfirm --distpath ..\dist --workpath .\build
```

Output: `dist/bakup-server/bakup-server.exe`

The spec file (`bakup.spec`) handles:
- Bundling all Python backend modules
- Including UI static files (`ui/`)
- Collecting hidden imports for `chromadb`, `sentence-transformers`, `torch`, `uvicorn`, `fastapi`
- Excluding unnecessary packages (`tkinter`, `matplotlib`, `pytest`)

### Step 2: Test the Binary

```powershell
# Set required env vars
$env:BAKUP_ACCESS_KEY = "tango"

# Run the compiled server
.\dist\bakup-server\bakup-server.exe
```

Open http://127.0.0.1:8000 — the UI should load and the API should respond.

### Step 3: Build Installer

```powershell
# Using Inno Setup command-line compiler
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" build\bakup-installer.iss
```

Output: `dist/bakup-ai-installer.exe`

### Step 4: Copy to Downloads

```powershell
Copy-Item dist\bakup-ai-installer.exe downloads\bakup-ai-installer.exe
```

---

## What the Installer Does

When a user runs `bakup-ai-installer.exe`:

1. **Installs** the compiled binary to `C:\Users\<user>\AppData\Local\Programs\bakup-ai\`
2. **Creates directories**: `data/vectordb/` and `data/model-weights/`
3. **Creates shortcuts**: Start menu entry + optional desktop shortcut
4. **Launches** the app (optional post-install step)

When bakup.ai runs:

1. `bakup-launcher.bat` sets environment variables and starts `bakup-server.exe`
2. The server loads the embedding model on first run (~90 MB download)
3. Browser opens automatically to `http://127.0.0.1:8000`
4. User points Bakup at their project via the browser UI

---

## Architecture

```
bakup-ai-installer.exe
└── Installs to: <user>\AppData\Local\Programs\bakup-ai\
    ├── bakup-launcher.bat          ← Entry point (Start menu / desktop shortcut)
    ├── bakup-server\
    │   ├── bakup-server.exe        ← Compiled Python backend
    │   ├── ui\                     ← Static HTML/CSS/JS (served by FastAPI)
    │   ├── backend\                ← Bundled backend modules
    │   ├── _internal\              ← Python runtime + wheel data
    │   └── *.dll / *.pyd           ← Native extensions (torch, chromadb, etc.)
    └── data\                       ← Runtime data (created on first run)
        ├── vectordb\               ← ChromaDB persistent storage
        └── model-weights\          ← Embedding model cache
```

---

## Build Files Reference

| File | Purpose |
|------|---------|
| `build/bakup_server.py` | PyInstaller-compatible entry point. Replaces `main.py`'s string-import pattern with direct app reference. Serves UI static files. Auto-opens browser. |
| `build/bakup.spec` | PyInstaller spec file. Defines hidden imports, data files, and build configuration. |
| `build/bakup-launcher.bat` | Windows launcher script. Sets env vars, creates data dirs, starts the server. |
| `build/bakup-installer.iss` | Inno Setup script. Packages compiled binary into a Windows installer. |
| `build/build.ps1` | Master build orchestrator. Runs PyInstaller → Inno Setup → copies to downloads/. |

---

## Updating the Build

When code changes:

1. Make changes to `backend/` or `ui/` as usual
2. Run `.\build\build.ps1` to regenerate the installer
3. The new installer appears at `downloads/bakup-ai-installer.exe`

When adding new Python modules:

1. Add the module to `backend/requirements.txt`
2. Add hidden imports to `build/bakup.spec` if needed
3. Rebuild

When changing the access key:

1. Update the hash in `backend/core/access.py`
2. Update the hash in `app.js` (landing page key gate)
3. Update the default in `build/bakup-launcher.bat`
4. Rebuild

---

## Troubleshooting

**PyInstaller fails with import errors:**
Add the missing module to `hidden_imports` in `bakup.spec`.

**Installer is too large:**
The compiled binary is ~500MB-1.5GB due to `torch` and `sentence-transformers`. This is expected. The Inno Setup LZMA compression reduces the installer to ~200-500MB.

**"Access key not set" on launch:**
The launcher script sets `BAKUP_ACCESS_KEY=tango`. If running the `.exe` directly, set the env var first.

**Model downloads on first run:**
The embedding model (~90MB) is downloaded automatically on first launch. The user needs internet access for this initial setup.
