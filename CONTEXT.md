# bakup.ai — AI Context File

> **Last updated:** 2026-03-07
> **Purpose:** Provides a complete snapshot of the project for any AI assistant to resume work without loss of context.

---

## 1. What is bakup.ai?

bakup.ai is a **local-first AI project intelligence and production support assistant** for developers and DevOps engineers. It indexes project source code and log files, stores them as vector embeddings, and lets engineers ask natural-language questions about errors, incidents, code quality, and architecture — with answers grounded strictly in their own data. It reasons over structured evidence like an experienced engineer, not a keyword search tool. No data ever leaves the machine (unless an external LLM like OpenAI is configured).

### Core Workflow
```
User uploads/indexes a project folder
  → Files are scanned recursively (code + logs)
  → Text is chunked, embedded (all-MiniLM-L6-v2, dim=384), stored in ChromaDB
  → Code is parsed for functions, classes, imports, dependencies
  → Symbol graph + architecture summary auto-generated
  → User asks questions via chat UI
  → Query is routed by hybrid router (LLM + rule-based fallback):
      → project_query → planner + agent retrieval pipeline
      → conversational → LLM direct response (or canned if no LLM)
      → unrelated → polite scope message
  → For project queries:
      → Agentic planner generates multi-step retrieval plan
      → Agent executor runs plan: semantic search → keyword search → code refs → deps → cross-analysis
      → Structured evidence built: logs, code, deps, architecture, cross-analysis, trends, clusters
      → Full evidence context sent to LLM with mode-appropriate prompt
      → Quality gate checks response completeness; retries if truncated
      → Structured answer returned with confidence scores and source citations
```

### Vision
bakup.ai is designed to become the **best AI production support assistant in the market**. It should reason over structured evidence, correlate logs with code, trace dependency chains, and produce actionable root-cause analysis — not behave like a keyword search tool.

---

## 2. Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Backend** | Python 3.10+ / FastAPI | Port 8000, uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS | Port 3000 (dev) or nginx (Docker) |
| **Embeddings** | sentence-transformers | `all-MiniLM-L6-v2`, dim=384 |
| **Vector DB** | ChromaDB | Persistent at `./vectordb`, cosine distance |
| **LLM Providers** | OpenAI, Anthropic, Azure OpenAI, Ollama | Configurable via UI or API |
| **Containerization** | Docker Compose | backend + nginx (ui) |
| **Git** | GitHub | `https://github.com/salilsankritya/bakup.ai.git`, branch `main` |

---

## 3. Directory Structure

```
bakup.ai/
├── CONTEXT.md              ← THIS FILE
├── .context.md             ← Detailed installer/build/pipeline context
├── FEATURES.md             ← Complete feature list
├── HOSTING.md              ← Landing page hosting & distribution guide
├── DEPLOY.md               ← Production deployment guide (VPS/Railway/Render)
├── TESTING_GUIDE.md        ← Comprehensive test instructions
├── .nojekyll               ← Prevents Jekyll processing on GitHub Pages
├── .env / .env.example     ← Environment variables
├── .gitignore
├── docker-compose.yml      ← Two services: backend + ui (nginx)
├── README.md
├── index.html              ← Landing page (GitHub Pages)
├── styles.css              ← Landing page styles
├── fonts.css               ← Landing page fonts
├── app.js                  ← Landing page key-gate + download logic
│
├── ui/                     ← Frontend (static files)
│   ├── index.html          ← 302 lines — single-page shell
│   ├── app.js              ← 935 lines — chat UI, indexing, SSE, LLM config
│   ├── styles.css           ← 1174 lines — full stylesheet
│   ├── fonts.css            ← 22 lines — font-face declarations
│   └── nginx.conf           ← 49 lines — reverse proxy config
│
├── backend/                ← Python backend
│   ├── main.py             ← 93 lines — FastAPI app entry point
│   ├── config.py           ← 87 lines — env-based settings
│   ├── Dockerfile          ← 30 lines — container build
│   ├── requirements.txt    ← pinned dependencies
│   │
│   ├── api/routes/
│   │   ├── health.py       ← GET /health
│   │   ├── index.py        ← POST /index, /index/github, /index/upload
│   │   ├── query.py        ← POST /ask, /ask/stream (SSE) — routes via brain controller
│   │   ├── llm_config.py   ← GET/PUT /llm/config, /llm/status, /llm/test
│   │   └── debug.py        ← Diagnostic endpoints incl. /debug/brain
│   │
│   ├── core/access.py      ← Startup access key gate
│   │
│   ├── core/brain/
│   │   ├── __init__.py     ← Brain module init
│   │   ├── brain.py        ← Brain controller — LLM-orchestrated tool loop + fallback
│   │   ├── tools.py        ← Tool interface layer (8 tools wrapping retrieval/analysis)
│   │   └── prompt_templates.py ← Brain system prompt for tool-calling orchestration
│   │
│   ├── core/ingestion/
│   │   ├── chunker.py      ← Chunk dataclass + text chunking (line-window)
│   │   ├── file_walker.py  ← Recursive project scanner with security guards
│   │   ├── log_parser.py   ← Log file parser with severity tagging
│   │   └── github_ingester.py ← GitHub repo shallow clone ingestion
│   │
│   ├── core/embeddings/
│   │   ├── embedder.py     ← Sentence-transformers wrapper
│   │   └── model_cache.py  ← Local model weight management
│   │
│   ├── core/retrieval/
│   │   ├── models.py       ← RetrievedChunk dataclass
│   │   ├── vector_store.py ← ChromaDB interface (add, query, keyword, severity search)
│   │   ├── ranker.py       ← Distance-to-confidence conversion + threshold gate
│   │   ├── rag.py          ← Full RAG pipeline (classify → plan → agent → evidence → answer)
│   │   ├── planner.py      ← Agentic question type classifier + multi-step plan generator
│   │   └── agent.py        ← Multi-step agent executor + structured evidence builder
│   │
│   ├── core/analysis/
│   │   ├── confidence.py   ← 6-factor confidence scoring (incl. cross-file)
│   │   ├── trends.py       ← Error trend detection (hourly counts, spikes, repeating failures)
│   │   ├── clusters.py     ← Timeline clustering (temporal + keyword merge)
│   │   └── file_aggregation.py ← Cross-file error distribution analysis
│   │
│   ├── core/classifier/
│   │   └── query_classifier.py ← 4-category query classification (project/greeting/conversational/off_topic)
│   │
│   ├── core/router/
│   │   └── router.py       ← Hybrid query router (LLM + rule-based fallback, confidence threshold)
│   │
│   └── core/llm/
│       ├── config_store.py ← JSON storage for LLM config (API keys masked)
│       ├── llm_service.py  ← LLM abstraction (OpenAI, Anthropic, Azure, Ollama) + tool calling + quality gate
│       └── prompt_templates.py ← All system prompts (RAG, agentic reasoning, code review, log summary, cross-analysis, conversational)
│
├── sample-project/         ← Test project with logs and source code
│   ├── logs/app.log        ← Sample log with errors (timeout, NoneType, etc.)
│   └── src/auth/session.py ← Sample Python source (SessionStore class)
│
├── scripts/
│   ├── generate_key_hash.py
│   └── generate-key-hash.js
│
├── vectordb/               ← ChromaDB persistent storage (gitignored)
├── model-weights/          ← Embedding model cache (gitignored)
└── logs/                   ← Application runtime logs
```

**Total:** ~54 Python files, ~12,000 lines backend / ~2,400 lines frontend

---

## 4. Key Configuration

| Setting | Value | Source |
|---------|-------|--------|
| Access Key | `tango` | `BAKUP_ACCESS_KEY` env var (SHA-256 hashed at startup) |
| Backend Port | 8000 | uvicorn default |
| UI Port | 3000 (dev) / 8080 (Docker) | http.server / nginx |
| Embedding Model | `all-MiniLM-L6-v2` | dim=384, cached in `model-weights/` |
| ChromaDB Dir | `./vectordb` | `BAKUP_CHROMA_DIR` env var |
| Confidence Threshold | 0.35 | `BAKUP_CONFIDENCE_THRESHOLD` env var |
| LLM Context Window | 16384 | `BAKUP_LLM_CONTEXT_WINDOW` env var |
| LLM Max Tokens | 2048 | `BAKUP_LLM_MAX_TOKENS` env var |
| LLM Temperature | 0.1 | `BAKUP_LLM_TEMPERATURE` env var |
| Router Confidence Threshold | 0.60 | Below this → falls back to project_query |
| Max Tool Calls | 5 | `BAKUP_MAX_TOOL_CALLS` — brain tool budget per query |
| App Mode | local | `BAKUP_APP_MODE` — local or cloud |
| LLM Context Results | 10 | Max chunks sent to LLM per query |
| Max Chunk Chars | 1200 | Per-chunk character limit in LLM context |
| Evidence Logs Cap | 12 | Max log chunks in evidence context |
| Evidence Code Cap | 12 | Max code chunks in evidence context |
| Architecture Cap | 2500 chars | Architecture summary truncation limit |
| Test Namespace | `d2a64b8f709574d8f1899038` | Derived from sample-project path |

---

## 5. How to Start the App

### Quick Start (Development)
```powershell
# 1. Start backend
cd "c:\Users\91876\Documents\bakup.ai\backend"
$env:BAKUP_ACCESS_KEY="tango"
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 2. Start UI (separate terminal)
python -m http.server 3000 --bind 127.0.0.1 --directory "c:\Users\91876\Documents\bakup.ai\ui"

# 3. Open browser → http://localhost:3000
```

### Restart Procedure
```powershell
# Kill all existing Python processes
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# Restart backend
cd "c:\Users\91876\Documents\bakup.ai\backend"
$env:BAKUP_ACCESS_KEY="tango"
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Restart UI (separate terminal)
python -m http.server 3000 --bind 127.0.0.1 --directory "c:\Users\91876\Documents\bakup.ai\ui"
```

### Docker Start
```powershell
cd "c:\Users\91876\Documents\bakup.ai"
docker compose up --build
# Backend: internal port 8000
# UI: http://localhost:8080
```

### Health Check
```powershell
# PowerShell
python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"

# Expected: {"status":"ok","version":"0.1.0","llm_status":"ready","llm_message":"ollama / llama3"}
```

---

## 6. Git History

| Hash | Message | Date |
|------|---------|------|
| `c97ddc1` | fix: port conflict crash + garbled Unicode in Windows console | Latest |
| `5dfb01d` | hosting: GitHub Pages + Releases setup for landing page distribution (tag: v0.2.0) | |
| `4dd669c` | v7: production deployment - nginx SSE/debug/download routes, named volumes, CORS env, DEPLOY.md | |
| `a83360f` | feat: causal confidence scoring + error pattern clustering (v4) | |
| `0831f23` | feat: agentic retrieval system with multi-step reasoning | |
| `a9353f0` | feat: project intelligence v2 - symbol graph, architecture summary, context bundling, log-code cross-analysis | |
| `dd508dd` | feat: code-aware ingestion with language parsing, metadata extraction, and retrieval boosting | |
| `5f7cca9` | fix: audit download endpoint - add HEAD support, nosniff, cache headers | |
| `22f1d99` | feat: auto-detect free port when default is busy | |
| `c53e240` | build: reduce AV false positives - disable UPX, add version info, expand excludes | |
| `2b3d75e` | fix: installer audit - desktop shortcut, browser polling, context file | |
| `5185951` | feat: compiled Windows installer distribution | |
| `835d7ea` | fix: audit and fix all landing page links, add download ZIP, changelog, and privacy sections | |
| `06c4eb5` | content: update landing page for multi-file log intelligence capabilities | |
| `1c4e97b` | content: update website copy for multi-file log intelligence capabilities | |
| `243d9f5` | feat: multi-file log intelligence, severity tagging, cross-file confidence, UI fixes & context doc | |
| `5693209` | feat: log intelligence upgrade - multi-factor confidence, trend detection, clustering | |
| `3197629` | feat: add query classifier with conversational/meta routing | |
| `4d43282` | feat: bakup.ai - full platform with hybrid retrieval, debug diagnostics & SSE streaming | |
| `0757dd7` | Initial commit | |

All changes are committed and pushed to GitHub. No uncommitted changes.

---

## 7. Architecture Deep Dive

### Data Flow: Ingestion
```
POST /index { path, log_path?, namespace? }
  → _validate_path() — normalize, existence check, Docker volume guard
  → _derive_namespace() — SHA-256 of path → 24-char hex
  → walk_project(root)
      → os.walk() with SKIP_DIRS pruning
      → For each file: extension filter → size filter → safety check
      → .log/.out files → parse_log_file() → per-entry chunks with severity
      → Other files → chunk_file() → line-window chunks (30 lines, 10 overlap)
  → embed_texts() → all-MiniLM-L6-v2 → 384-dim vectors
  → add_chunks() → ChromaDB upsert with metadata:
      {source_file, line_start, line_end, source_type, file_name, severity, detected_timestamp, last_modified}
  → Return { namespace, chunks_stored }
```

### Data Flow: Query
```
POST /ask { question, namespace, top_k?, debug? }
  → route_query() via hybrid router:
      1. Try LLM classification (if provider configured)
      2. If LLM confidence >= 0.6 → use LLM intent
      3. Otherwise → rule-based classification
      4. If rule confidence < 0.6 and intent != project_query → force project_query
  → If conversational → brain.process_query() → LLM direct or canned response
  → If unrelated → brain.process_query() → scope guard response
  → If project_query:
      → brain.process_query() decides mode:

      [BRAIN MODE — LLM configured + supports tool calling]
      → LLM receives user query + tool schemas + session context
      → LLM decides which tools to call:
          - search_logs, search_code, retrieve_dependencies
          - get_architecture_summary, get_error_clusters
          - get_file_context, query_symbol_graph, cross_analyse
      → Tools execute and return JSON evidence
      → Evidence fed back to LLM for more tool calls or final answer
      → Up to max_tool_calls (default 5) iterations
      → Final answer grounded in tool evidence with citations

      [FALLBACK MODE — no LLM or no tool calling]
      → Agentic planner classifies question type:
          LOG_ANALYSIS | CODE_ANALYSIS | CODE_REVIEW | ROOT_CAUSE | ARCHITECTURE | STRUCTURAL | GENERAL
      → Generates multi-step retrieval plan (2-5 steps per question type)
      → Agent executor runs each step:
          1. Semantic search → top chunks
          2. Keyword search → error/exception/traceback matches
          3. Code reference extraction → file:line from stack traces
          4. Dependency resolution → import/call graph traversal
          5. Cross-analysis → log-to-code linking
      → Structured evidence built and sent to LLM (or extractive fallback)

      → Result stored in brain debug cache
      → Turn stored in session memory for follow-up support
```

### Confidence Scoring (6 Factors)
| Factor | Weight (broad) | Weight (specific) | Description |
|--------|:-:|:-:|---|
| Similarity | 0.15 | 0.35 | Top + avg embedding distance |
| Volume | 0.20 | 0.12 | Number of relevant chunks (diminishing returns) |
| Severity | 0.20 | 0.18 | Highest severity in results (fatal→1.0, error→0.85, warn→0.55) |
| Recency | 0.10 | 0.10 | Age of timestamps (<1h→1.0, <24h→0.8, <7d→0.5) |
| Pattern | 0.15 | 0.12 | Consistency of error types across chunks |
| Cross-file | 0.20 | 0.13 | Errors spanning multiple files (4+→1.0, 2→0.65, 1→0.40) |

### Severity Tagging
Applied at ingestion time in `log_parser.py`:
- **error**: `ERROR`, `Exception`, `Traceback`, `Failed`, `Critical`, `Fatal`
- **warning**: `WARN`, `Warning`
- **info**: everything else

### LLM Providers
Configured via UI settings panel or `PUT /llm/config`:
- **Ollama** (default) — local, no API key needed, model: `llama3`
- **OpenAI** — requires API key, model: `gpt-4o-mini`
- **Anthropic (Claude)** — requires API key, model: `claude-sonnet-4-20250514`
- **Azure OpenAI** — requires API key + endpoint + deployment name

---

## 8. API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + LLM status |
| POST | `/index` | Index local directory |
| POST | `/index/github` | Index GitHub repo (shallow clone) |
| POST | `/index/upload` | Index uploaded files (multipart) |
| POST | `/ask` | Ask a question (JSON response) |
| POST | `/ask/stream` | Ask with SSE streaming |
| GET | `/llm/status` | LLM provider health check |
| GET | `/llm/config` | Get current LLM config (keys masked) |
| PUT | `/llm/config` | Update LLM provider config |
| POST | `/llm/test` | Test LLM connectivity |
| GET | `/debug/stats/{namespace}` | Collection stats |
| GET | `/debug/sample/{namespace}` | Sample chunks |
| POST | `/debug/router` | Hybrid router diagnostics |
| GET | `/debug/brain` | Brain controller state + tools |
| GET | `/debug/brain/{namespace}` | Last brain result for namespace |

---

## 9. Dependencies (requirements.txt)

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
pydantic==2.7.4
python-multipart==0.0.9
sentence-transformers==3.0.1
torch==2.3.1
transformers==4.44.2
chromadb==0.5.5
llama-cpp-python==0.2.88
gitpython==3.1.43
httpx==0.27.0
chardet==5.2.0
```

---

## 10. Known State & Notes

- **Ollama** must be running locally for LLM-augmented answers. If not available, the system falls back to **extractive mode** (returns structured top-5 chunk excerpts with formatting — no hallucination possible).
- The `model-weights/` directory caches the embedding model on first run (~80MB download).
- The `vectordb/` directory persists ChromaDB data between restarts.
- After code changes, **re-index** the project to populate new metadata fields (severity, file_name, etc.) in existing ChromaDB data.
- The UI communicates with the backend via `http://localhost:8000` (hardcoded in `ui/app.js`).
- All uncommitted changes listed in Section 6 are now **committed and pushed** as of 2026-03-07.
- **Production Deployment v7**: Docker Compose with named volumes, nginx with SSE/debug/download proxy routes, CORS env var, DEPLOY.md guide.
- **Hosting & Distribution v8**: Landing page live on GitHub Pages (`https://salilsankritya.github.io/bakup.ai`), installer on GitHub Releases (v0.2.0, 241 MB), HOSTING.md guide.
- **Port conflict fix**: Dual-phase port detection (connect + bind), uvicorn retry loop for Errno 10048, re-validates port before server start.
- **Unicode fix**: All print statements use ASCII dashes, launcher .bat sets `chcp 65001` for UTF-8 console.
- **Reasoning Engine v5**: All question types now receive the full evidence context (logs + code + deps + architecture + cross-analysis). No evidence is discarded at the last mile. The response quality gate ensures LLM output is complete. Token budget of 2048 allows deep multi-section analysis.
- **Hybrid Router v6**: Query routing uses LLM classification when available, with rule-based fallback. Conversational/unrelated queries are always short-circuited. Ingestion now excludes model-weights, vectordb, and binary files. Retrieval guard prevents low-confidence extractive matches from being surfaced.
- **Brain Architecture v9**: `/ask` route now goes through `brain.process_query()`. When an LLM with tool-calling support is configured, the brain orchestrates 8 tools (search_logs, search_code, retrieve_dependencies, get_architecture_summary, get_error_clusters, get_file_context, query_symbol_graph, cross_analyse) in an iterative loop (max 5 calls per query). Falls back to the deterministic planner→agent pipeline when no LLM is configured. Anthropic Claude added as a new provider alongside OpenAI/Azure/Ollama. Cloud mode config (`BAKUP_APP_MODE=local|cloud`) added. Debug endpoints: `GET /debug/brain` and `GET /debug/brain/{ns}`.

---

## 11. Testing

### Quick Smoke Test
```powershell
cd "c:\Users\91876\Documents\bakup.ai\backend"
python -c "
from core.ingestion.log_parser import _detect_severity
assert _detect_severity('ERROR: test') == 'error'
assert _detect_severity('WARNING: test') == 'warning'
assert _detect_severity('INFO: test') == 'info'
print('Severity tagging OK')

from core.analysis.file_aggregation import aggregate_by_file
from core.retrieval.models import RetrievedChunk
chunks = [
    RetrievedChunk('ERROR: x', 'a.log', 1, 1, 'log', 0.1, file_name='a.log', severity='error'),
    RetrievedChunk('WARN: y', 'b.log', 1, 1, 'log', 0.2, file_name='b.log', severity='warning'),
]
r = aggregate_by_file(chunks)
assert r.total_errors == 1
assert r.files_affected == 2
print('File aggregation OK')

print('All quick tests passed.')
"
```

### Full Smoke Test Suite
See `TESTING_GUIDE.md` for the 7-test smoke suite covering chunker, file walker, log parser, ranker, GitHub validator, threshold gate, and access key.

### Live End-to-End Test
```powershell
# With server running:
python -c "
import urllib.request, json
data = json.dumps({'question': 'any errors?', 'namespace': 'd2a64b8f709574d8f1899038', 'access_key': 'tango'}).encode()
req = urllib.request.Request('http://127.0.0.1:8000/ask', data=data, headers={'Content-Type': 'application/json'})
r = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
print(f'Mode: {r[\"mode\"]}, Confidence: {r[\"confidence\"]}, Sources: {len(r[\"sources\"])}')
"
```

---

## 12. What to Work On Next (suggestions)

Potential next steps, not yet started:
1. **Commit all uncommitted changes** — four feature sets ready to push (Enter-to-Send, Path Handling, Log Intelligence, Reasoning Engine v5)
2. **README.md** — currently a stub, needs proper documentation
3. **SSE streaming for agentic mode** — currently agentic answers wait for full LLM response; stream tokens as they arrive
4. **Folder dedup tracking** — prevent re-indexing the same folder (hash-based)
5. **Multi-log-folder support** — allow indexing dedicated log directories separately
6. **UI improvements** — file tree view, severity badges, error distribution charts, evidence chain visualization
7. **Prompt tuning per question type** — tailor system prompts further for LOG_ANALYSIS, CODE_REVIEW, GENERAL modes
8. **Auth improvements** — token-based auth instead of simple access key
9. **Rate limiting** — protect endpoints from abuse
10. **Automated tests** — pytest suite with fixtures covering the full pipeline
11. **Custom domain** — configure custom domain (e.g., bakup.ai) for GitHub Pages landing page
12. **Code signing** — sign the installer .exe to eliminate AV/SmartScreen warnings

---

*This file is meant to be read by AI assistants. Feed it at the start of a new session to restore full project context.*
