/**
 * bakup.ai — Electron main process
 *
 * Responsibilities:
 *  1. Spawn the compiled bakup-server.exe backend
 *  2. Wait for the health endpoint to respond
 *  3. Load the UI served by the backend into a BrowserWindow
 *  4. Kill the backend when the app closes
 */

const { app, BrowserWindow, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const net = require("net");
const http = require("http");

// ── Paths ────────────────────────────────────────────────────────────────────

/** Resolve the bakup-server directory depending on dev vs packaged mode. */
function getServerDir() {
  if (app.isPackaged) {
    // electron-builder puts extraResources alongside the asar
    return path.join(process.resourcesPath, "bakup-server");
  }
  // Dev: use the dist folder built by PyInstaller
  return path.join(__dirname, "..", "dist", "bakup-server");
}

const SERVER_EXE = path.join(getServerDir(), "bakup-server.exe");

// ── Port helpers ─────────────────────────────────────────────────────────────

function isPortFree(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.once("listening", () => {
      srv.close(() => resolve(true));
    });
    srv.listen(port, "127.0.0.1");
  });
}

async function findFreePort(start = 8000, max = 8020) {
  for (let p = start; p <= max; p++) {
    if (await isPortFree(p)) return p;
  }
  throw new Error(`No free port found between ${start} and ${max}`);
}

// ── Health check ─────────────────────────────────────────────────────────────

function checkHealth(port) {
  return new Promise((resolve) => {
    const req = http.get(
      `http://127.0.0.1:${port}/health`,
      { timeout: 2000 },
      (res) => {
        let body = "";
        res.on("data", (d) => (body += d));
        res.on("end", () => resolve(res.statusCode === 200));
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForServer(port, maxWait = 120000) {
  const interval = 1000;
  const start = Date.now();
  while (Date.now() - start < maxWait) {
    if (await checkHealth(port)) return true;
    await new Promise((r) => setTimeout(r, interval));
  }
  return false;
}

// ── Backend process management ───────────────────────────────────────────────

let backendProcess = null;
let backendPort = 8000;

function startBackend(port) {
  const serverDir = getServerDir();
  const dataDir = path.join(serverDir, "data");

  const env = Object.assign({}, process.env, {
    BAKUP_ACCESS_KEY: process.env.BAKUP_ACCESS_KEY || "tango",
    BAKUP_HOST: "127.0.0.1",
    BAKUP_PORT: String(port),
    BAKUP_CHROMA_DIR: path.join(dataDir, "vectordb"),
    BAKUP_MODEL_CACHE_DIR: path.join(dataDir, "model-weights"),
    // Prevent bakup-server from opening a browser on its own
    BAKUP_NO_BROWSER: "1",
  });

  backendProcess = spawn(SERVER_EXE, [], {
    cwd: serverDir,
    env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  backendProcess.stdout.on("data", (d) => {
    process.stdout.write(`[bakup-server] ${d}`);
  });
  backendProcess.stderr.on("data", (d) => {
    process.stderr.write(`[bakup-server] ${d}`);
  });

  backendProcess.on("error", (err) => {
    console.error("Failed to start bakup-server:", err.message);
  });

  backendProcess.on("exit", (code, signal) => {
    console.log(
      `bakup-server exited (code=${code}, signal=${signal})`
    );
    backendProcess = null;
  });

  return backendProcess;
}

function killBackend() {
  if (!backendProcess) return;
  try {
    // On Windows, child_process.kill() sends SIGTERM which doesn't work
    // for console-less processes. Use taskkill instead.
    const { execSync } = require("child_process");
    execSync(`taskkill /F /T /PID ${backendProcess.pid}`, {
      stdio: "ignore",
      windowsHide: true,
    });
  } catch {
    // Process may already have exited
    try {
      backendProcess.kill("SIGKILL");
    } catch {
      /* ignore */
    }
  }
  backendProcess = null;
}

// ── Window management ────────────────────────────────────────────────────────

let mainWindow = null;
let splashWindow = null;

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 420,
    height: 320,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: false,
    show: false,
    title: "bakup.ai",
    icon: getIconPath(),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  splashWindow.loadFile(path.join(__dirname, "splash.html"));
  splashWindow.once("ready-to-show", () => splashWindow.show());
  return splashWindow;
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    title: "bakup.ai",
    icon: getIconPath(),
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
      devTools: !app.isPackaged,
    },
  });

  mainWindow.loadURL(`http://127.0.0.1:${port}`);

  mainWindow.once("ready-to-show", () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
    mainWindow.focus();
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  return mainWindow;
}

function getIconPath() {
  const iconFile = path.join(__dirname, "icons", "icon.ico");
  try {
    require("fs").accessSync(iconFile);
    return iconFile;
  } catch {
    return undefined;
  }
}

// ── App lifecycle ────────────────────────────────────────────────────────────

// Prevent multiple instances
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

app.whenReady().then(async () => {
  try {
    // 1. Show splash
    createSplashWindow();

    // 2. Find a free port
    backendPort = await findFreePort(8000);
    console.log(`bakup.ai: using port ${backendPort}`);

    // 3. Verify the server binary exists
    const fs = require("fs");
    if (!fs.existsSync(SERVER_EXE)) {
      const msg =
        `Could not find the bakup.ai backend at:\n${SERVER_EXE}\n\n` +
        `Please reinstall the application.`;
      dialog.showErrorBox("bakup.ai — Backend Not Found", msg);
      app.quit();
      return;
    }

    // 4. Start backend
    startBackend(backendPort);

    // 5. Wait for health
    const ready = await waitForServer(backendPort, 120000);
    if (!ready) {
      const msg =
        "The bakup.ai backend did not start within 2 minutes.\n\n" +
        "This can happen on first launch while the embedding model\n" +
        "downloads (~90 MB). Please try again.";
      dialog.showErrorBox("bakup.ai — Startup Timeout", msg);
      killBackend();
      app.quit();
      return;
    }

    // 6. Show main window
    createMainWindow(backendPort);
  } catch (err) {
    dialog.showErrorBox(
      "bakup.ai — Error",
      `An unexpected error occurred:\n\n${err.message}`
    );
    killBackend();
    app.quit();
  }
});

app.on("window-all-closed", () => {
  killBackend();
  app.quit();
});

app.on("before-quit", () => {
  killBackend();
});

process.on("exit", () => {
  killBackend();
});
