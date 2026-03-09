/**
 * Bakup Local UI — app.js  (chat interface)
 *
 * Architecture: session-based chat, localStorage persistence.
 * Zero external dependencies. No analytics. No tracking.
 */
'use strict';

// ─── Config ───────────────────────────────────────────────────────────────────
// When served behind nginx on 8080, use same-origin (proxy handles routing).
// When running dev mode on 3000/5500, talk directly to backend on 8000.
const API_BASE       = ['3000','5500'].includes(location.port)
                       ? `${location.protocol}//${location.hostname}:8000`
                       : '';
const HEALTH_POLL_MS = 15_000;
const DEFAULT_MODELS = { openai: 'gpt-4o-mini', anthropic: 'claude-sonnet-4-20250514', azure_openai: 'gpt-4o-mini', ollama: 'llama3' };
const LS_SESSIONS    = 'bakup_sessions_v2';
const LS_ACTIVE      = 'bakup_active_session_v2';

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $q = sel => document.querySelector(sel);
const $$ = sel => document.querySelectorAll(sel);

// Sidebar
const sessionList       = $('session-list');
const newSessionBtn     = $('new-session-btn');
const sidebarStatusDot  = $('status-dot');
const sidebarStatusLbl  = $('status-label');

// Header
const sessionTitle      = $('session-title');
const headerNamespace   = $('header-namespace');
const headerStatusDot   = $('header-status-dot');
const headerStatusLbl   = $('header-status-label');
const openIndexBtnHdr   = $('open-index-btn-header');

// Chat
const chatArea          = $('chat-area');
const welcomeState      = $('welcome-state');
const messageList       = $('message-list');
const welcomeIndexBtn   = $('welcome-index-btn');

// Input
const chatInput         = $('chat-input');
const sendBtn           = $('send-btn');
const nsHint            = $('input-namespace-hint');

// Index panel
const panelOverlay      = $('panel-overlay');
const indexPanel        = $('index-panel');
const closeIndexBtn     = $('close-index-btn');
const openIndexBtnSide  = $('open-index-btn');
const indexLocalForm    = $('index-local-form');
const indexGithubForm   = $('index-github-form');
const iPath             = $('i-path');
const iLog              = $('i-log');
const folderPicker      = $('folder-picker');
const filePicker        = $('file-picker');
const browseFolderBtn   = $('browse-folder-btn');
const browseFileBtn     = $('browse-file-btn');
const iNamespace        = $('i-namespace');
const iSubmit           = $('i-submit');
const gUrl              = $('g-url');
const gBranch           = $('g-branch');
const gSubmit           = $('g-submit');
const indexResult       = $('index-result');
const indexResultInner  = $('index-result-inner');
const panelLoader       = $('panel-loader');
const panelLoaderMsg    = $('panel-loader-msg');

// Upload tab
const indexUploadForm   = $('index-upload-form');
const uProjectFiles     = $('u-project-files');
const uLogFiles         = $('u-log-files');
const uProjectCount     = $('u-project-count');
const uLogCount         = $('u-log-count');
const uNamespace        = $('u-namespace');
const uSubmit           = $('u-submit');

// Rename modal
const renameModal       = $('rename-modal');
const renameInput       = $('rename-input');
const renameConfirmBtn  = $('rename-confirm-btn');
const renameCancelBtn   = $('rename-cancel-btn');
const closeRenameBtn    = $('close-rename-btn');

// LLM setup modal
const llmSetupModal    = $('llm-setup-modal');
const closeLlmBtn      = $('close-llm-setup-btn');
const llmSetupBtn      = $('llm-setup-btn');
const sidebarLlmBtn    = $('sidebar-llm-btn');
const llmStatusDot     = $('llm-status-dot');
const llmStatusLabel   = $('llm-status-label');
const llmSetupForm     = $('llm-setup-form');
const llmProvider      = $('llm-provider');
const llmModel         = $('llm-model');
const llmApiKey        = $('llm-api-key');
const llmAzureEndpoint = $('llm-azure-endpoint');
const llmAzureVersion  = $('llm-azure-version');
const llmOllamaUrl     = $('llm-ollama-url');
const llmTestBtn       = $('llm-test-btn');
const llmTestResult    = $('llm-test-result');
const llmSaveBtn       = $('llm-save-btn');
const llmSkipBtn       = $('llm-skip-btn');
const llmKeyGroup      = $('llm-key-group');
const llmAzureGroup    = $('llm-azure-group');
const llmOllamaGroup   = $('llm-ollama-group');

// ─── Session State ────────────────────────────────────────────────────────────
let sessions    = [];     // Session[]
let activeId    = null;   // string

function loadState() {
  try {
    sessions = JSON.parse(localStorage.getItem(LS_SESSIONS) || '[]');
    activeId = localStorage.getItem(LS_ACTIVE) || null;
  } catch { sessions = []; activeId = null; }
}

function saveState() {
  localStorage.setItem(LS_SESSIONS, JSON.stringify(sessions));
  if (activeId) localStorage.setItem(LS_ACTIVE, activeId);
}

function getActive() {
  return sessions.find(s => s.id === activeId) || null;
}

function createSession(name = 'Untitled Session') {
  const s = {
    id:        crypto.randomUUID(),
    name,
    namespace: '',
    messages:  [],
    createdAt: Date.now(),
  };
  sessions.unshift(s);
  saveState();
  return s;
}

function setActive(id) {
  activeId = id;
  localStorage.setItem(LS_ACTIVE, id);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function fmtDate(ts) {
  const d = new Date(ts);
  const today = new Date();
  if (d.toDateString() === today.toDateString()) return 'Today';
  const yesterday = new Date(); yesterday.setDate(today.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

// ─── API ──────────────────────────────────────────────────────────────────────
async function apiPost(path, body) {
  const res  = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = data?.detail;
    const msg = Array.isArray(detail)
      ? detail.map(e => e?.msg ?? JSON.stringify(e)).join('; ')
      : (typeof detail === 'object' && detail !== null)
        ? JSON.stringify(detail)
        : (detail ?? `HTTP ${res.status}`);
    throw new Error(msg);
  }
  return data;
}

async function apiGet(path) {
  const res  = await fetch(`${API_BASE}${path}`);
  const data = await res.json();
  if (!res.ok) {
    const detail = data?.detail;
    const msg = Array.isArray(detail)
      ? detail.map(e => e?.msg ?? JSON.stringify(e)).join('; ')
      : (typeof detail === 'object' && detail !== null)
        ? JSON.stringify(detail)
        : (detail ?? `HTTP ${res.status}`);
    throw new Error(msg);
  }
  return data;
}

// ─── Sidebar render ───────────────────────────────────────────────────────────
function renderSidebar() {
  sessionList.innerHTML = '';

  if (sessions.length === 0) {
    const empty = document.createElement('div');
    empty.style.cssText = 'padding:8px 12px; font-size:0.76rem; color:var(--text-muted);';
    empty.textContent = 'No sessions yet.';
    sessionList.appendChild(empty);
    return;
  }

  sessions.forEach(s => {
    const el = document.createElement('div');
    el.className = 'session-item' + (s.id === activeId ? ' session-item--active' : '');
    el.dataset.id = s.id;

    el.innerHTML = `
      <svg class="session-item__icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
        <path d="M2 4h12M2 8h9M2 12h6"/>
      </svg>
      <div class="session-item__body">
        <div class="session-item__name">${esc(s.name)}</div>
        <div class="session-item__date">${fmtDate(s.createdAt)}</div>
      </div>
      <button class="session-item__del" data-del="${s.id}" title="Delete session">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M4 4l8 8M12 4l-8 8"/>
        </svg>
      </button>
    `;

    el.addEventListener('click', e => {
      if (e.target.closest('[data-del]')) return;
      switchSession(s.id);
    });

    el.querySelector('[data-del]').addEventListener('click', e => {
      e.stopPropagation();
      deleteSession(s.id);
    });

    sessionList.appendChild(el);
  });
}

// ─── Header update ────────────────────────────────────────────────────────────
function updateHeader() {
  const s = getActive();
  sessionTitle.textContent = s ? s.name : 'Untitled Session';
  headerNamespace.textContent = s?.namespace || '';
  updateNsHint(s?.namespace || '');
}

function updateNsHint(ns) {
  if (ns) {
    nsHint.textContent = `ns: ${ns}`;
    nsHint.classList.add('has-ns');
  } else {
    nsHint.textContent = 'No namespace — index a project first';
    nsHint.classList.remove('has-ns');
  }
}

// ─── Switch / create session ─────────────────────────────────────────────────
function switchSession(id) {
  const s = sessions.find(s => s.id === id);
  if (!s) return;
  setActive(id);
  renderSidebar();
  updateHeader();
  renderMessages();
}

function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  if (activeId === id) {
    activeId = sessions[0]?.id || null;
    if (!activeId) {
      const ns = createSession();
      activeId = ns.id;
    }
    localStorage.setItem(LS_ACTIVE, activeId || '');
  }
  saveState();
  renderSidebar();
  updateHeader();
  renderMessages();
}

// ─── Message render ───────────────────────────────────────────────────────────
function renderMessages() {
  const s = getActive();
  messageList.innerHTML = '';

  if (!s || s.messages.length === 0) {
    welcomeState.hidden   = false;
    messageList.hidden    = true;
    return;
  }

  welcomeState.hidden = true;
  messageList.hidden  = false;

  s.messages.forEach(msg => {
    const el = buildMessageEl(msg);
    messageList.appendChild(el);
  });

  scrollToBottom();
}

function scrollToBottom() {
  chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
}

// ─── Build message DOM ────────────────────────────────────────────────────────
function buildMessageEl(msg) {
  const el = document.createElement('div');
  el.className = `message message--${msg.role}`;
  el.dataset.msgId = msg.id;

  // Avatar
  const avatarContent = msg.role === 'user'
    ? 'YOU'
    : `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
         <rect x="2" y="2" width="5" height="5" rx="1"/>
         <rect x="9" y="2" width="5" height="5" rx="1"/>
         <rect x="2" y="9" width="5" height="5" rx="1"/>
         <path d="M11.5 9v5M9 11.5h5"/>
       </svg>`;

  const roleLabel  = msg.role === 'user' ? 'You' : 'Bakup';
  const timeStr    = msg.timestamp ? fmtTime(msg.timestamp) : '';

  let bodyHTML = `
    <div class="msg-header">
      <div class="msg-avatar">${avatarContent}</div>
      <span class="msg-role">${roleLabel}</span>
      <span class="msg-time">${timeStr}</span>
    </div>
    <div class="msg-body">
  `;

  if (msg.isThinking) {
    bodyHTML += `<div class="msg-bubble msg-bubble--thinking">
      <span class="thinking-status" style="font-size:0.85em;opacity:0.7;display:block;margin-bottom:4px;min-height:1.2em"></span>
      <span class="thinking-dots">
        <span>●</span><span>●</span><span>●</span>
      </span>
    </div>`;
  } else if (msg.isError) {
    bodyHTML += `<div class="error-bubble">${esc(msg.content)}</div>`;
  } else if (msg.no_data && msg.mode !== 'clarification' && !msg.content) {
    bodyHTML += `<div class="no-data-bubble">
      <strong>No similar incident found.</strong><br>
      No indexed content was similar enough to answer this. Try rephrasing, or index more context.
    </div>`;
  } else {
    bodyHTML += `<div class="msg-bubble">${esc(msg.content)}</div>`;

    if (msg.role === 'assistant' && msg.confidence != null) {
      const tier = msg.confidenceTier ?? tierFromScore(msg.confidence);
      const pct  = Math.round(msg.confidence * 100);
      bodyHTML += `
        <div class="confidence-row">
          <span class="conf-badge conf-badge--${tier}">${pct}% ${tier}</span>
          ${msg.mode ? `<span class="mode-badge">${esc(msg.mode)}</span>` : ''}
        </div>
      `;
    }

    if (msg.role === 'assistant' && msg.sources?.length > 0) {
      bodyHTML += buildSourcesHTML(msg.sources, msg.id);
    }
  }

  if (msg.role === 'assistant' && !msg.isThinking && !msg.isError) {
    bodyHTML += `
      <div class="msg-actions">
        <button class="action-btn" data-copy="${msg.id}" title="Copy answer">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4">
            <rect x="4" y="4" width="9" height="10" rx="1"/>
            <path d="M2 2h8v2H4v8H2V2z"/>
          </svg>Copy
        </button>
      </div>
    `;
  }

  bodyHTML += `</div>`;
  el.innerHTML = bodyHTML;

  // Wire copy button
  const copyBtn = el.querySelector('[data-copy]');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => copyMessage(msg.id, copyBtn));
  }

  // Wire sources toggle
  const toggle = el.querySelector('.sources-toggle');
  if (toggle) {
    const list = el.querySelector('.sources-list');
    toggle.addEventListener('click', () => {
      const open = toggle.classList.toggle('expanded');
      if (list) list.hidden = !open;
    });
  }

  // Wire excerpt toggles
  el.querySelectorAll('.source-card__toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const pre = btn.previousElementSibling;
      if (pre) {
        const exp = pre.classList.toggle('expanded');
        btn.textContent = exp ? 'Show less' : 'Show more';
      }
    });
  });

  return el;
}

function buildSourcesHTML(sources, msgId) {
  const cards = sources.map((src, i) => {
    const tier = src.confidence_label ?? tierFromScore(src.confidence);
    const pct  = Math.round(src.confidence * 100);
    const excerpt = (src.excerpt || '').trim();
    const isLong  = excerpt.split('\n').length > 3 || excerpt.length > 200;
    return `
      <div class="source-card source-card--${tier}">
        <div class="source-card__header">
          <span class="source-card__file" title="${esc(src.file)}">${esc(src.file)}</span>
          <span class="source-card__lines">L${src.line_start}–${src.line_end}</span>
          <span class="source-card__type source-card__type--${src.source_type}">${src.source_type}</span>
          <span class="source-card__conf">${pct}% ${tier}</span>
        </div>
        ${excerpt ? `
          <pre class="source-card__excerpt">${esc(excerpt)}</pre>
          ${isLong ? `<button class="source-card__toggle">Show more</button>` : ''}
        ` : ''}
      </div>
    `;
  }).join('');

  return `
    <div class="sources-block">
      <button class="sources-toggle">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M6 4l4 4-4 4"/>
        </svg>
        Sources <span class="sources-count">${sources.length}</span>
      </button>
      <div class="sources-list" hidden>${cards}</div>
    </div>
  `;
}

function tierFromScore(score) {
  if (score >= 0.70) return 'high';
  if (score >= 0.45) return 'medium';
  return 'low';
}

// ─── Copy to clipboard ────────────────────────────────────────────────────────
function copyMessage(msgId, btn) {
  const s   = getActive();
  const msg = s?.messages.find(m => m.id === msgId);
  if (!msg) return;

  navigator.clipboard.writeText(msg.content).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 9l3 3 7-7"/></svg>Copied`;
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="4" y="4" width="9" height="10" rx="1"/><path d="M2 2h8v2H4v8H2V2z"/></svg>Copy`;
    }, 2000);
  });
}

// ─── Append message ───────────────────────────────────────────────────────────
function appendMessage(msg) {
  const s = getActive();
  if (!s) return;

  if (!msg.isThinking) {
    s.messages.push(msg);
    saveState();
  }

  welcomeState.hidden = true;
  messageList.hidden  = false;

  const el = buildMessageEl(msg);
  el.dataset.transient = msg.isThinking ? '1' : '';
  messageList.appendChild(el);
  scrollToBottom();

  return el;
}

function removeThinkingEl() {
  const el = messageList.querySelector('[data-transient="1"]');
  if (el) el.remove();
}

// ─── SSE streaming helper ─────────────────────────────────────────────────────
async function askWithStreaming(question, namespace, thinkingEl) {
  const statusEl = thinkingEl.querySelector('.thinking-status');

  try {
    const response = await fetch(`${API_BASE}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, namespace, top_k: 8 }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (!response.body) throw new Error('No response body');

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let finalData = null;
    let buffer    = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();   // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'step' && statusEl) {
            statusEl.textContent = event.message;
            scrollToBottom();
          } else if (event.type === 'result') {
            finalData = event.data;
          }
        } catch {}
      }
    }

    if (finalData?.error) throw new Error(finalData.error);
    if (finalData) return finalData;
    throw new Error('No result received from stream');

  } catch (err) {
    // Fallback to regular /ask endpoint
    console.warn('SSE stream failed, falling back to /ask:', err.message);
    return await apiPost('/ask', { question, namespace, top_k: 8 });
  }
}

// ─── Send message ─────────────────────────────────────────────────────────────
async function sendMessage(text) {
  const s = getActive();
  if (!s || !s.namespace) return;

  const question = text.trim();
  if (!question) return;

  // Clear input
  chatInput.value = '';
  autoResize();
  sendBtn.disabled = true;

  // User bubble
  const userMsg = {
    id:        crypto.randomUUID(),
    role:      'user',
    content:   question,
    timestamp: Date.now(),
  };
  appendMessage(userMsg);

  // Thinking bubble (transient)
  const thinkingMsg = { id: 'thinking', role: 'assistant', isThinking: true };
  const thinkingEl  = appendMessage(thinkingMsg);

  try {
    const data = await askWithStreaming(question, s.namespace, thinkingEl);

    removeThinkingEl();

    const assistantMsg = {
      id:             crypto.randomUUID(),
      role:           'assistant',
      content:        data.answer ?? '',
      timestamp:      Date.now(),
      confidence:     data.confidence,
      confidenceTier: data.sources?.[0]?.confidence_label ?? tierFromScore(data.confidence),
      mode:           data.mode,
      sources:        data.sources ?? [],
      no_data:        data.no_data ?? false,
    };
    appendMessage(assistantMsg);

  } catch (err) {
    removeThinkingEl();
    appendMessage({
      id:        crypto.randomUUID(),
      role:      'assistant',
      content:   err.message,
      timestamp: Date.now(),
      isError:   true,
    });
  } finally {
    sendBtn.disabled = !s?.namespace;
  }
}

// ─── Input auto-resize ────────────────────────────────────────────────────────
function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
}

chatInput.addEventListener('input', () => {
  autoResize();
  const s = getActive();
  sendBtn.disabled = !chatInput.value.trim() || !s?.namespace;
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage(chatInput.value);
  }
});

sendBtn.addEventListener('click', () => {
  if (!sendBtn.disabled) sendMessage(chatInput.value);
});

// ─── Welcome chips ────────────────────────────────────────────────────────────
$$('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const s = getActive();
    if (!s?.namespace) { openIndexPanel(); return; }
    chatInput.value = chip.dataset.q;
    autoResize();
    sendBtn.disabled = false;
    chatInput.focus();
  });
});

// ─── New session ──────────────────────────────────────────────────────────────
newSessionBtn.addEventListener('click', () => {
  const s = createSession();
  setActive(s.id);
  renderSidebar();
  updateHeader();
  renderMessages();
});

// ─── Session rename ───────────────────────────────────────────────────────────
sessionTitle.addEventListener('click', openRenameModal);

function openRenameModal() {
  const s = getActive();
  if (!s) return;
  renameInput.value = s.name;
  renameModal.hidden = false;
  renameInput.focus();
  renameInput.select();
}

function closeRenameModal() {
  renameModal.hidden = true;
}

function confirmRename() {
  const s = getActive();
  if (!s) return;
  const newName = renameInput.value.trim();
  if (newName) s.name = newName;
  saveState();
  updateHeader();
  renderSidebar();
  closeRenameModal();
}

renameConfirmBtn.addEventListener('click', confirmRename);
renameCancelBtn.addEventListener('click', closeRenameModal);
closeRenameBtn.addEventListener('click', closeRenameModal);
renameInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') confirmRename();
  if (e.key === 'Escape') closeRenameModal();
});

// ─── Index panel ──────────────────────────────────────────────────────────────
function openIndexPanel() {
  panelOverlay.hidden = false;
  indexPanel.classList.add('open');
  indexPanel.setAttribute('aria-hidden', 'false');
  indexResult.hidden    = true;
  panelLoader.hidden    = true;
}

function closeIndexPanel() {
  panelOverlay.hidden = true;
  indexPanel.classList.remove('open');
  indexPanel.setAttribute('aria-hidden', 'true');
}

openIndexBtnSide.addEventListener('click', openIndexPanel);
openIndexBtnHdr.addEventListener('click', openIndexPanel);
welcomeIndexBtn.addEventListener('click', openIndexPanel);
closeIndexBtn.addEventListener('click', closeIndexPanel);
panelOverlay.addEventListener('click', closeIndexPanel);

// ── Browse buttons ──────────────────────────────────────────────────
browseFolderBtn.addEventListener('click', () => folderPicker.click());
browseFileBtn.addEventListener('click',   () => filePicker.click());

folderPicker.addEventListener('change', e => {
  const files = e.target.files;
  if (!files || !files.length) return;
  const first = files[0];
  if (first.path && first.path.length > 0) {
    // Electron / local Chromium: .path gives the full absolute path
    const fullPath = first.path;
    // Strip the filename to get the parent directory
    const sepIdx = Math.max(fullPath.lastIndexOf('/'), fullPath.lastIndexOf('\\'));
    iPath.value = sepIdx > 0 ? fullPath.substring(0, sepIdx) : fullPath;
  } else if (first.webkitRelativePath) {
    // Standard browser fallback — webkitRelativePath only gives relative paths
    // (e.g. "my-folder/file.txt"), NOT the full absolute path.
    // Pre-fill the folder name and alert the user to type the full path.
    const folderName = first.webkitRelativePath.split('/')[0];
    iPath.value = folderName;
    iPath.focus();
    iPath.setSelectionRange(0, folderName.length);
    showIndexError(
      'Browser cannot detect the full folder path. ' +
      'Please type the complete absolute path (e.g. C:\\projects\\' + folderName + ').'
    );
  }
  folderPicker.value = '';
});

filePicker.addEventListener('change', e => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  if (file.path && file.path.length > 0) {
    // Electron / local Chromium: full absolute path available
    iLog.value = file.path;
  } else {
    // Standard browser: only filename available — user must type full path
    iLog.value = file.name;
    iLog.focus();
    iLog.setSelectionRange(0, file.name.length);
    showIndexError(
      'Browser cannot detect the full file path. ' +
      'Please type the complete absolute path to the log file.'
    );
  }
  filePicker.value = '';
});

// Index tabs
$$('.index-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.index-tab').forEach(t => t.classList.remove('index-tab--active'));
    tab.classList.add('index-tab--active');
    $('tab-local').hidden  = tab.dataset.tab !== 'local';
    $('tab-upload').hidden = tab.dataset.tab !== 'upload';
    $('tab-github').hidden = tab.dataset.tab !== 'github';
  });
});

// ─── Index local form ─────────────────────────────────────────────────────────
indexLocalForm.addEventListener('submit', async e => {
  e.preventDefault();
  const path = iPath.value.trim();
  if (!path) { iPath.focus(); return; }

  const body = { path: path };
  if (iLog.value.trim())       body.log_path  = iLog.value.trim();
  if (iNamespace.value.trim()) body.namespace  = iNamespace.value.trim();

  iSubmit.disabled  = true;
  panelLoader.hidden = false;
  panelLoaderMsg.textContent = 'Indexing project…';
  indexResult.hidden = true;

  try {
    const data = await apiPost('/index', body);
    onIndexSuccess(data);
  } catch (err) {
    showIndexError(err.message);
  } finally {
    iSubmit.disabled  = false;
    panelLoader.hidden = true;
  }
});

// ─── Index GitHub form ────────────────────────────────────────────────────────
indexGithubForm.addEventListener('submit', async e => {
  e.preventDefault();
  const url = gUrl.value.trim();
  if (!url) { gUrl.focus(); return; }

  const body = { repo_url: url };
  if (gBranch.value.trim()) body.branch = gBranch.value.trim();

  gSubmit.disabled   = true;
  panelLoader.hidden  = false;
  panelLoaderMsg.textContent = 'Cloning and indexing…';
  indexResult.hidden  = true;

  try {
    const data = await apiPost('/index/github', body);
    onIndexSuccess(data);
  } catch (err) {
    showIndexError(err.message);
  } finally {
    gSubmit.disabled   = false;
    panelLoader.hidden  = true;
  }
});

// ─── Upload form ──────────────────────────────────────────────────────────────
uProjectFiles.addEventListener('change', () => {
  const n = uProjectFiles.files.length;
  uProjectCount.textContent = n ? `${n} file(s) selected` : '';
});
uLogFiles.addEventListener('change', () => {
  const n = uLogFiles.files.length;
  uLogCount.textContent = n ? `${n} file(s) selected` : '';
});

indexUploadForm.addEventListener('submit', async e => {
  e.preventDefault();
  const projFiles = uProjectFiles.files;
  const logFiles  = uLogFiles.files;
  if (!projFiles.length && !logFiles.length) {
    showIndexError('Select at least one source or log file to upload.');
    return;
  }

  const form = new FormData();
  for (const f of projFiles) form.append('project_files', f);
  for (const f of logFiles)  form.append('log_files', f);
  if (uNamespace.value.trim()) form.append('namespace', uNamespace.value.trim());

  uSubmit.disabled   = true;
  panelLoader.hidden  = false;
  panelLoaderMsg.textContent = 'Uploading and indexing…';
  indexResult.hidden  = true;

  try {
    const res = await fetch(`${API_BASE}/index/upload`, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      const detail = data?.detail;
      const msg = Array.isArray(detail)
        ? detail.map(e => e?.msg ?? JSON.stringify(e)).join('; ')
        : (typeof detail === 'object' && detail !== null)
          ? JSON.stringify(detail)
          : (detail ?? `HTTP ${res.status}`);
      throw new Error(msg);
    }
    onIndexSuccess(data);
  } catch (err) {
    showIndexError(err.message);
  } finally {
    uSubmit.disabled   = false;
    panelLoader.hidden  = true;
  }
});

function onIndexSuccess(data) {
  const s = getActive();
  if (data.chunks_stored > 0 && s) {
    s.namespace = data.namespace;
    saveState();
    updateHeader();
    updateNsHint(s.namespace);
    sendBtn.disabled = !chatInput.value.trim();
  }

  if (data.chunks_stored === 0) {
    indexResultInner.className   = 'index-result__inner err';
    indexResultInner.textContent =
      `✗ No chunks indexed\nThe repository may be empty, inaccessible, or contain no supported files.\nFor private repos, embed a token in the URL.`;
  } else {
    indexResultInner.className   = 'index-result__inner ok';
    indexResultInner.textContent =
      `✓ Indexed\nNamespace : ${data.namespace}\nChunks    : ${data.chunks_stored}`;
  }
  indexResult.hidden = false;
}

function showIndexError(msg) {
  indexResultInner.className   = 'index-result__inner err';
  indexResultInner.textContent = `✗ ${msg}`;
  indexResult.hidden = false;
}

// ─── Health poll ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const dots   = [sidebarStatusDot, headerStatusDot];
  const labels = [sidebarStatusLbl, headerStatusLbl];

  dots.forEach(d   => { d.className = 'status-dot status-dot--checking'; });
  labels.forEach(l => { l.textContent = 'Checking…'; });
  updateLlmStatus('checking', 'Checking…');

  try {
    const data = await apiGet('/health');
    dots.forEach(d   => { d.className = 'status-dot status-dot--ok'; });
    labels.forEach(l => { l.textContent = 'Online'; });
    updateLlmStatus(data.llm_status, data.llm_message);
    return data;
  } catch {
    dots.forEach(d   => { d.className = 'status-dot status-dot--error'; });
    labels.forEach(l => { l.textContent = 'Offline'; });
    updateLlmStatus('error', 'Backend offline');
    return null;
  }
}

function updateLlmStatus(status, message) {
  const map = {
    ready:          'status-dot--ok',
    not_configured: 'status-dot--warn',
    error:          'status-dot--error',
  };
  llmStatusDot.className = 'status-dot ' + (map[status] || 'status-dot--checking');
  llmStatusLabel.textContent = status === 'ready'          ? 'AI Ready'
                             : status === 'not_configured' ? 'AI Not Set'
                             : status === 'error'          ? 'AI Error'
                             : 'AI…';
  llmStatusDot.title = message || '';
}

// ─── LLM setup modal ──────────────────────────────────────────────────────────────────────
function openLlmSetupModal() {
  llmSetupModal.hidden = false;
  clearTestResult();
  apiGet('/llm/config').then(cfg => {
    llmProvider.value       = cfg.provider || 'openai';
    llmModel.value          = cfg.model    || DEFAULT_MODELS[cfg.provider] || '';
    llmApiKey.value         = cfg.api_key_set ? '••••••••' : '';
    llmAzureEndpoint.value  = cfg.azure_endpoint    || '';
    llmAzureVersion.value   = cfg.azure_api_version || '2024-02-01';
    llmOllamaUrl.value      = cfg.ollama_base_url   || 'http://localhost:11434';
    syncProviderFields();
  }).catch(() => syncProviderFields());
}

function closeLlmSetup() {
  llmSetupModal.hidden = true;
  clearTestResult();
}

function syncProviderFields() {
  const p = llmProvider.value;
  if (!llmModel.value || Object.values(DEFAULT_MODELS).includes(llmModel.value)) {
    llmModel.value = DEFAULT_MODELS[p] || '';
  }
  llmKeyGroup.hidden    = (p === 'ollama');
  llmAzureGroup.hidden  = (p !== 'azure_openai');
  llmOllamaGroup.hidden = (p !== 'ollama');
  // Update API key placeholder per provider
  llmApiKey.placeholder = p === 'anthropic' ? 'sk-ant-…' : 'sk-…';
}

function clearTestResult() {
  llmTestResult.hidden = true;
  llmTestResult.textContent = '';
  llmTestResult.className = 'setup-test-result';
}

function buildLlmConfigBody() {
  const p    = llmProvider.value;
  const body = { provider: p, model: llmModel.value.trim() || DEFAULT_MODELS[p] };
  const key  = llmApiKey.value.trim();
  if (key && !key.startsWith('•')) body.api_key = key;
  if (p === 'azure_openai') {
    body.azure_endpoint     = llmAzureEndpoint.value.trim();
    body.azure_api_version  = llmAzureVersion.value.trim() || '2024-02-01';
  }
  if (p === 'ollama') {
    body.ollama_base_url = llmOllamaUrl.value.trim() || 'http://localhost:11434';
  }
  return body;
}

[llmSetupBtn, sidebarLlmBtn].forEach(btn =>
  btn.addEventListener('click', openLlmSetupModal)
);
closeLlmBtn.addEventListener('click', closeLlmSetup);
llmSkipBtn.addEventListener('click', closeLlmSetup);

llmProvider.addEventListener('change', () => {
  syncProviderFields();
  clearTestResult();
});

llmTestBtn.addEventListener('click', async () => {
  const body = buildLlmConfigBody();
  llmTestBtn.disabled = true;
  llmTestResult.hidden = false;
  llmTestResult.className = 'setup-test-result setup-test-result--checking';
  llmTestResult.textContent = 'Testing…';
  try {
    const r = await apiPost('/llm/test', body);
    llmTestResult.className = 'setup-test-result setup-test-result--ok';
    llmTestResult.textContent = r.message || 'Connection successful!';
  } catch (err) {
    llmTestResult.className = 'setup-test-result setup-test-result--error';
    llmTestResult.textContent = err.message;
  } finally {
    llmTestBtn.disabled = false;
  }
});

llmSetupForm.addEventListener('submit', async e => {
  e.preventDefault();
  const body = buildLlmConfigBody();
  llmSaveBtn.disabled = true;
  llmTestResult.hidden = false;
  llmTestResult.className = 'setup-test-result setup-test-result--checking';
  llmTestResult.textContent = 'Saving…';
  try {
    await apiPost('/llm/config', body);
    llmTestResult.className = 'setup-test-result setup-test-result--ok';
    llmTestResult.textContent = 'Saved!';
    setTimeout(() => { closeLlmSetup(); checkHealth(); }, 800);
  } catch (err) {
    llmTestResult.className = 'setup-test-result setup-test-result--error';
    llmTestResult.textContent = err.message;
  } finally {
    llmSaveBtn.disabled = false;
  }
});

setInterval(checkHealth, HEALTH_POLL_MS);

// ─── Boot ─────────────────────────────────────────────────────────────────────
(async function init() {
  loadState();

  // Ensure at least one session
  if (sessions.length === 0) {
    const s = createSession('Session 1');
    activeId = s.id;
    saveState();
  } else if (!activeId || !sessions.find(s => s.id === activeId)) {
    activeId = sessions[0].id;
    saveState();
  }

  renderSidebar();
  updateHeader();
  renderMessages();
  syncProviderFields();

  const health = await checkHealth();
  // Show setup modal on first run if AI is not configured
  if (health?.llm_status === 'not_configured') {
    openLlmSetupModal();
  }
})();
