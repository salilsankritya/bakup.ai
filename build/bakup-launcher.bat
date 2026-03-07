@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: bakup.ai — Windows Launcher
:: ─────────────────────────────────────────────────────────────────────────────
:: This script starts the bakup.ai server and opens the browser.
:: It is placed in the installation directory by the installer.
:: ─────────────────────────────────────────────────────────────────────────────

title bakup.ai — Local Incident Intelligence
echo.
echo  ========================================
echo    bakup.ai — Local Incident Intelligence
echo  ========================================
echo.
echo  Starting server...
echo  Press Ctrl+C to stop.
echo.

:: Set the working directory to where this script lives
cd /d "%~dp0"

:: Set environment variables
set BAKUP_ACCESS_KEY=tango
set BAKUP_HOST=127.0.0.1
set BAKUP_PORT=8000
set BAKUP_CHROMA_DIR=%~dp0data\vectordb
set BAKUP_MODEL_CACHE_DIR=%~dp0data\model-weights

:: Create data directories if they don't exist
if not exist "%~dp0data" mkdir "%~dp0data"
if not exist "%~dp0data\vectordb" mkdir "%~dp0data\vectordb"
if not exist "%~dp0data\model-weights" mkdir "%~dp0data\model-weights"

:: Start the server
:: The compiled binary serves both the API and the UI
"%~dp0bakup-server\bakup-server.exe"

:: If the server exits, pause so the user can see any errors
echo.
echo  Server stopped. Press any key to close.
pause >nul
