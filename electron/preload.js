// preload.js — minimal preload script for context isolation
// No node APIs are exposed to the renderer.
window.addEventListener("DOMContentLoaded", () => {
  // Nothing needed — the UI is served by the backend and
  // communicates via fetch/XHR to the same origin.
});
