<#
.SYNOPSIS
    bakup.ai — Full build pipeline: PyInstaller compile → Electron installer.

.DESCRIPTION
    This script:
    1. Compiles the Python backend into a standalone binary using PyInstaller
    2. Bundles the UI static files into the binary
    3. Packages everything into an Electron-wrapped Windows installer
    4. Copies the installer to the downloads/ directory for the website

.NOTES
    Prerequisites:
    - Python 3.10+ with pip
    - PyInstaller (pip install pyinstaller)
    - Node.js 18+ with npm

.EXAMPLE
    .\build\build.ps1
#>

param(
    [switch]$SkipPyInstaller,
    [switch]$SkipInstaller,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist"
$DownloadsDir = Join-Path $ProjectRoot "downloads"
$ElectronDir = Join-Path $ProjectRoot "electron"

Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Cyan
Write-Host "    bakup.ai — Build Pipeline" -ForegroundColor Cyan
Write-Host "  ==========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Verify prerequisites ─────────────────────────────────────────────
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Check Python
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.10+ and try again."
    exit 1
}
Write-Host "  Python: $pythonVersion" -ForegroundColor Gray

# Check PyInstaller
$pyiVersion = python -m PyInstaller --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  PyInstaller not found. Installing..." -ForegroundColor Yellow
    pip install pyinstaller --quiet
}
Write-Host "  PyInstaller: $(python -m PyInstaller --version 2>&1)" -ForegroundColor Gray

# Check Node.js
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Node.js not found. Install Node.js 18+ and try again."
    exit 1
}
Write-Host "  Node.js: $nodeVersion" -ForegroundColor Gray

# Check npm
$npmVersion = npm --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "npm not found. Install Node.js 18+ (includes npm) and try again."
    exit 1
}
Write-Host "  npm: $npmVersion" -ForegroundColor Gray

Write-Host "  Prerequisites OK." -ForegroundColor Green
Write-Host ""

# ── Step 2: Clean previous builds ────────────────────────────────────────────
Write-Host "[2/5] Cleaning previous builds..." -ForegroundColor Yellow

$CleanPaths = @(
    (Join-Path $DistDir "bakup-server"),
    (Join-Path $DistDir "electron-out"),
    (Join-Path $BuildDir "build"),
    (Join-Path $BuildDir "__pycache__")
)
foreach ($p in $CleanPaths) {
    if (Test-Path $p) {
        Remove-Item -Path $p -Recurse -Force
        Write-Host "  Removed: $p" -ForegroundColor Gray
    }
}
Write-Host "  Clean." -ForegroundColor Green
Write-Host ""

# ── Step 3: PyInstaller compile ──────────────────────────────────────────────
if (-not $SkipPyInstaller) {
    Write-Host "[3/5] Compiling backend with PyInstaller..." -ForegroundColor Yellow
    Write-Host "  This may take 5-15 minutes on first run." -ForegroundColor Gray
    Write-Host ""

    Push-Location $BuildDir
    try {
        $specFile = Join-Path $BuildDir "bakup.spec"

        $pyiArgs = @(
            $specFile,
            "--noconfirm",
            "--distpath", $DistDir,
            "--workpath", (Join-Path $BuildDir "build")
        )
        if (-not $Verbose) {
            $pyiArgs += "--log-level", "WARN"
        }

        python -m PyInstaller @pyiArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Error "PyInstaller compilation failed."
            exit 1
        }
    } finally {
        Pop-Location
    }

    $exePath = Join-Path $DistDir "bakup-server\bakup-server.exe"
    if (-not (Test-Path $exePath)) {
        Write-Error "Build output not found at: $exePath"
        exit 1
    }

    $exeSize = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
    Write-Host "  Compiled: $exePath ($exeSize MB)" -ForegroundColor Green

    # ── Post-compile cleanup: remove test/debug artifacts from dist ──────────
    Write-Host "  Cleaning test and debug artifacts from dist..." -ForegroundColor Gray
    $internalDir = Join-Path $DistDir "bakup-server\_internal"
    $cleanupDirs = @(
        # pytest cache shipped from backend source
        (Join-Path $internalDir "backend\.pytest_cache"),
        # chromadb test suite
        (Join-Path $internalDir "chromadb\test"),
        # torch subsystems — stubbed at runtime by bakup_server.py
        (Join-Path $internalDir "torch\_inductor"),
        (Join-Path $internalDir "torch\_dynamo"),
        (Join-Path $internalDir "torch\onnx"),
        (Join-Path $internalDir "torch\_export"),
        (Join-Path $internalDir "torch\package"),
        (Join-Path $internalDir "torch\compiler")
        # NOTE: do NOT delete distributed, testing, ao, profiler
        # — they are needed by torch.nn.parallel at import time
    )
    $removedMB = 0
    foreach ($d in $cleanupDirs) {
        if (Test-Path $d) {
            $size = (Get-ChildItem $d -Recurse -File -ErrorAction SilentlyContinue |
                     Measure-Object -Property Length -Sum).Sum
            $removedMB += $size
            Remove-Item $d -Recurse -Force
        }
    }
    $removedMB = [math]::Round($removedMB / 1MB, 1)
    Write-Host "  Removed $removedMB MB of test/debug artifacts." -ForegroundColor Gray

    Write-Host ""
} else {
    Write-Host "[3/5] Skipping PyInstaller (--SkipPyInstaller)." -ForegroundColor Gray
    Write-Host ""
}

# ── Step 4: Electron installer ───────────────────────────────────────────────
if (-not $SkipInstaller) {
    Write-Host "[4/5] Building Electron installer..." -ForegroundColor Yellow

    # Verify bakup-server was compiled
    $exePath = Join-Path $DistDir "bakup-server\bakup-server.exe"
    if (-not (Test-Path $exePath)) {
        Write-Error "bakup-server.exe not found at: $exePath. Run without -SkipPyInstaller first."
        exit 1
    }

    Push-Location $ElectronDir
    try {
        Write-Host "  Installing Electron dependencies..." -ForegroundColor Gray
        npm install --no-audit --no-fund 2>&1 | Out-Null

        Write-Host "  Packaging with electron-builder..." -ForegroundColor Gray
        $env:ELECTRON_BUILDER_DIST = $DistDir
        $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"

        # Pre-populate winCodeSign cache to work around symlink privilege error
        # (darwin symlinks in the archive fail on Windows without Developer Mode)
        $wcsCache = Join-Path $env:LOCALAPPDATA "electron-builder\Cache\winCodeSign\winCodeSign-2.6.0"
        if (-not (Test-Path $wcsCache)) {
            Write-Host "  Pre-extracting winCodeSign cache..." -ForegroundColor Gray
            $wcsCacheDir = Split-Path $wcsCache -Parent
            New-Item -ItemType Directory -Path $wcsCacheDir -Force | Out-Null
            $wcsUrl = "https://github.com/electron-userland/electron-builder-binaries/releases/download/winCodeSign-2.6.0/winCodeSign-2.6.0.7z"
            $wcsArchive = Join-Path $wcsCacheDir "winCodeSign-2.6.0.7z"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $wcsUrl -OutFile $wcsArchive -UseBasicParsing
            $7zaPath = Join-Path $ElectronDir "node_modules\7zip-bin\win\x64\7za.exe"
            & $7zaPath x -bd -y $wcsArchive "-o$wcsCache" 2>&1 | Out-Null
            Remove-Item $wcsArchive -Force -ErrorAction SilentlyContinue
        }

        npx electron-builder --win --publish never 2>&1 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }

        if ($LASTEXITCODE -ne 0) {
            Write-Error "electron-builder packaging failed."
            exit 1
        }
    } finally {
        Pop-Location
    }

    # Find the built installer
    $electronOut = Join-Path $DistDir "electron-out"
    $installerPath = Get-ChildItem -Path $electronOut -Filter "bakup-ai-installer.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $installerPath) {
        # Fallback: look for any .exe installer
        $installerPath = Get-ChildItem -Path $electronOut -Filter "*.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if (-not $installerPath) {
        Write-Error "Installer not found in: $electronOut"
        exit 1
    }

    # Copy to dist root with canonical name
    $finalInstaller = Join-Path $DistDir "bakup-ai-installer.exe"
    Copy-Item -Path $installerPath.FullName -Destination $finalInstaller -Force

    $installerSize = [math]::Round((Get-Item $finalInstaller).Length / 1MB, 1)
    Write-Host "  Installer: $finalInstaller ($installerSize MB)" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[4/5] Skipping Electron installer (--SkipInstaller)." -ForegroundColor Gray
    Write-Host ""
}

# ── Step 5: Copy to downloads directory ──────────────────────────────────────
Write-Host "[5/5] Copying to downloads directory..." -ForegroundColor Yellow

New-Item -ItemType Directory -Path $DownloadsDir -Force | Out-Null

$installerPath = Join-Path $DistDir "bakup-ai-installer.exe"
if (Test-Path $installerPath) {
    Copy-Item -Path $installerPath -Destination (Join-Path $DownloadsDir "bakup-ai-installer.exe") -Force
    Write-Host "  Copied: downloads\bakup-ai-installer.exe" -ForegroundColor Green
} else {
    Write-Host "  No installer to copy. Run full build first." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  ==========================================" -ForegroundColor Green
Write-Host "    Build complete!" -ForegroundColor Green
Write-Host "  ==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Output:" -ForegroundColor Cyan
Write-Host "    Binary:    dist\bakup-server\bakup-server.exe" -ForegroundColor Gray
if (Test-Path (Join-Path $DistDir "bakup-ai-installer.exe")) {
    Write-Host "    Installer: dist\bakup-ai-installer.exe" -ForegroundColor Gray
    Write-Host "    Download:  downloads\bakup-ai-installer.exe" -ForegroundColor Gray
}
Write-Host ""
Write-Host "  To test the server directly:" -ForegroundColor Cyan
Write-Host "    .\dist\bakup-server\bakup-server.exe" -ForegroundColor Gray
Write-Host ""
