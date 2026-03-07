<#
.SYNOPSIS
    bakup.ai — Full build pipeline: PyInstaller compile → Inno Setup installer.

.DESCRIPTION
    This script:
    1. Compiles the Python backend into a standalone binary using PyInstaller
    2. Bundles the UI static files into the binary
    3. Packages everything into a Windows installer using Inno Setup
    4. Copies the installer to the downloads/ directory for the website

.NOTES
    Prerequisites:
    - Python 3.10+ with pip
    - PyInstaller (pip install pyinstaller)
    - Inno Setup 6 (https://jrsoftware.org/isdl.php)

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

# Check Inno Setup
$InnoCompiler = $null
$InnoSearchPaths = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)
foreach ($path in $InnoSearchPaths) {
    if (Test-Path $path) {
        $InnoCompiler = $path
        break
    }
}
if (-not $SkipInstaller -and -not $InnoCompiler) {
    Write-Host "  WARNING: Inno Setup 6 not found. Installer step will be skipped." -ForegroundColor Yellow
    Write-Host "  Download from: https://jrsoftware.org/isdl.php" -ForegroundColor Gray
    $SkipInstaller = $true
} elseif ($InnoCompiler) {
    Write-Host "  Inno Setup: $InnoCompiler" -ForegroundColor Gray
}

Write-Host "  Prerequisites OK." -ForegroundColor Green
Write-Host ""

# ── Step 2: Clean previous builds ────────────────────────────────────────────
Write-Host "[2/5] Cleaning previous builds..." -ForegroundColor Yellow

$CleanPaths = @(
    (Join-Path $DistDir "bakup-server"),
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
        # torch subsystems excluded by spec but may leak via collect_all
        (Join-Path $internalDir "torch\testing"),
        (Join-Path $internalDir "torch\distributed"),
        (Join-Path $internalDir "torch\_inductor"),
        (Join-Path $internalDir "torch\_dynamo"),
        (Join-Path $internalDir "torch\onnx"),
        (Join-Path $internalDir "torch\_export"),
        (Join-Path $internalDir "torch\profiler"),
        (Join-Path $internalDir "torch\package"),
        (Join-Path $internalDir "torch\compiler")
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

# ── Step 4: Inno Setup installer ─────────────────────────────────────────────
if (-not $SkipInstaller) {
    Write-Host "[4/5] Building installer with Inno Setup..." -ForegroundColor Yellow

    $issFile = Join-Path $BuildDir "bakup-installer.iss"
    & $InnoCompiler $issFile

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Inno Setup compilation failed."
        exit 1
    }

    $installerPath = Join-Path $DistDir "bakup-ai-installer.exe"
    if (-not (Test-Path $installerPath)) {
        Write-Error "Installer not found at: $installerPath"
        exit 1
    }

    $installerSize = [math]::Round((Get-Item $installerPath).Length / 1MB, 1)
    Write-Host "  Installer: $installerPath ($installerSize MB)" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[4/5] Skipping installer (Inno Setup not available)." -ForegroundColor Gray
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
