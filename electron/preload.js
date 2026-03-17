// preload.js — Electron context-isolation bridge for bakup.ai
//
// Exposes a minimal API to the renderer so the UI can use
// native OS dialogs (folder picker) instead of the limited
// browser <input webkitdirectory> approach.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("bakupElectron", {
  /** Open a native OS folder-picker dialog. Returns the chosen path or null. */
  selectFolder: () => ipcRenderer.invoke("select-folder"),

  /** Open a native OS file-picker dialog. Returns the chosen path or null. */
  selectFile: (filters) => ipcRenderer.invoke("select-file", filters),

  /** True when running inside the Electron shell (vs. a plain browser). */
  isElectron: true,
});
