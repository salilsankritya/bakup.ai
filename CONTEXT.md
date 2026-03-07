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
| **LLM Providers** | OpenAI, Azure OpenAI, Ollama | Configurable via UI or API |
| **Containerization** | Docker Compose | backend + nginx (ui) |
| **Git** | GitHub | `https://github.com/salilsankritya/bakup.ai.git`, branch `main` |

---

## 3. Directory Structure

```
bakup.ai/
├── CONTEXT.md              ← THIS FILE
├── .env / .env.example     ← Environment variables
├── .gitignore
├── docker-compose.yml      ← Two services: backend + ui (nginx)
├── README.md
├── TESTING_GUIDE.md        ← Comprehensive test instructions
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
│   │   ├── query.py        ← POST /ask, /ask/stream (SSE)
│   │   ├── llm_config.py   ← GET/PUT /llm/config, /llm/status, /llm/test
│   │   └── debug.py        ← Diagnostic endpoints
│   │
│   ├── core/access.py      ← Startup access key gate
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
│       ├── llm_service.py  ← LLM abstraction (OpenAI, Azure, Ollama) + quality gate + unified agentic generation
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
| `a83360f` | feat: causal confidence scoring v4 — cluster ranking + structured reasoning | Latest committed |
| `0831f23` | feat: agentic retrieval v3 — planner + agent executor + session memory | |
| `a9353f0` | feat: agentic retrieval v2 — code review mode + evidence routing | |
| `5693209` | feat: log intelligence upgrade — multi-factor confidence, trend detection, clustering | |
| `3197629` | feat: add query classifier with conversational/meta routing | |
| `4d43282` | feat: bakup.ai — full platform with hybrid retrieval, debug diagnostics & SSE streaming | |
| `0757dd7` | Initial commit | |

### Uncommitted Changes (as of 2026-03-07)

These changes span **multiple feature sets** that haven't been committed yet:

#### A. Previously Uncommitted (Enter-to-Send, Path Handling, Multi-File Log Intelligence)
See prior CONTEXT.md versions for details — these remain uncommitted.

#### B. Query Classifier & Routing Fix
- `backend/core/classifier/query_classifier.py` — Default classification changed from CONVERSATIONAL to PROJECT; code review patterns added to `_PROJECT_STRONG`
- `backend/api/routes/query.py` — Only short-circuits GREETING (not conversational/off_topic when namespace exists); namespace-aware routing; pre_classified param; low-confidence → LLM routing

#### C. GitHub 0-Chunk Bug Fix
- `backend/api/routes/index.py` — Returns 422 when indexing produces 0 chunks
- `ui/app.js` — Shows user-friendly error on 0-chunk index result

#### D. Reasoning Engine Upgrade (v5 — Major)
Five critical pipeline fixes that transform the system from a search tool into a reasoning engine:

#### E. Hybrid Query Router (v6)
New pipeline entry point that replaces direct classifier routing:

1. **Router Module** — `core/router/router.py`
   - `route_query(query, namespace, session_context)` → `RoutingDecision(intent, confidence, source)`
   - LLM-based classification via small JSON prompt when provider is configured
   - Rule-based fallback using keyword banks + regex patterns (< 1 ms, no network)
   - Fallback protection: confidence < 0.6 → forced to `project_query`
   - Three intents: `project_query`, `conversational`, `unrelated`

2. **Ingestion Filtering** — `core/ingestion/file_walker.py`
   - Added `model-weights/`, `vectordb/` to `SKIP_DIRS`
   - New `SKIP_EXTENSIONS` set: `.bin`, `.model`, `.vocab`, `.onnx`, `.pt`, `.pth`, `.safetensors`, `.gguf`, `.ggml`, `.pkl`, `.exe`, `.dll`, `.so`, images, audio, fonts, archives
   - Both `walk_project()` and `list_indexed_files()` now check `SKIP_EXTENSIONS`

3. **Route Integration** — `api/routes/query.py`
   - Both `/ask` and `/ask/stream` now use `route_query()` as the first pipeline step
   - Conversational/unrelated always short-circuited regardless of namespace
   - Sub-classification via legacy classifier preserves greeting vs conversational distinction

4. **Retrieval Guard** — `core/llm/llm_service.py`
   - `_extractive_fallback()` now filters out chunks below confidence threshold
   - Returns clarification prompt instead of irrelevant matches

5. **Debug Endpoint** — `api/routes/debug.py`
   - `POST /debug/router` — shows intent, confidence, source (llm/rules), latency, routing decision

6. **Ignored Files for Ingestion**
   - Directories: `model-weights/`, `vectordb/`, `node_modules/`, `dist/`, `build/`, `.venv/`
   - Extensions: `.bin`, `.model`, `.vocab`, `.onnx`, `.pt`, `.pth`, `.safetensors`, `.gguf`, `.ggml`, `.pkl`, `.exe`, `.dll`, `.so`, `.dylib`, `.zip`, `.tar`, `.gz`, images, audio, fonts

1. **Token Budget Increase** — `config.py`, `llm_service.py`
   - `llm_context_window`: 4096 → 16384
   - `llm_max_tokens`: 512 → 2048
   - All 3 LLM providers (OpenAI, Azure, Ollama) updated

2. **Unified Evidence Routing** — `rag.py`, `llm_service.py`
   - ALL question types now route through `generate_agentic_answer()` with the full evidence context
   - Previously, only ROOT_CAUSE got the full context; LOG_ANALYSIS, CROSS_ANALYSIS, CODE_REVIEW, and GENERAL discarded deps, architecture, and cross-analysis
   - `generate_agentic_answer()` now supports 5 modes, each with an appropriate system prompt

3. **Extractive Fallback Depth** — `llm_service.py`
   - `_extractive_fallback()` shows top 5 chunks (was 1) with 1000 chars each (was 600)
   - Structured headers with confidence scores and code blocks

4. **Response Quality Gate** — `llm_service.py`
   - New `_quality_gate()` method detects truncated or too-short responses
   - Auto-retries with 2x token budget when LLM output is cut off
   - Checks for truncation signals and missing terminal punctuation

5. **Context Limits Increase** — `rag.py`, `llm_service.py`, `agent.py`
   - `_LLM_CONTEXT_RESULTS`: 5 → 10 chunks
   - `_MAX_CHUNK_CHARS`: 800 → 1200 per chunk
   - `build_evidence_context()`: logs/code cap 8 → 12, architecture cap 1500 → 2500 chars

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
      2. If LLM confidence ≥ 0.6 → use LLM intent
      3. Otherwise → rule-based classification
      4. If rule confidence < 0.6 and intent ≠ project_query → force project_query
  → If conversational → LLM direct call or canned response (no retrieval)
  → If unrelated → scope guard response
  → If project_query:
      → Agentic planner classifies question type:
          LOG_ANALYSIS | CODE_ANALYSIS | CODE_REVIEW | ROOT_CAUSE | ARCHITECTURE | STRUCTURAL | GENERAL
      → Generates multi-step retrieval plan (2–5 steps per question type)
      → Agent executor runs each step:
          1. Semantic search → top chunks
          2. Keyword search → error/exception/traceback matches
          3. Code reference extraction → file:line from stack traces
          4. Dependency resolution → import/call graph traversal
          5. Cross-analysis → log-to-code linking
      → Structured evidence built:
          - Log chunks (up to 12, 1200 chars each)
          - Code chunks (up to 12, 1200 chars each)
          - Dependencies list
          - Architecture summary (up to 2500 chars)
          - Cross-analysis context
          - Error clusters, trends, confidence scoring
      → build_evidence_context() → coherent context block for LLM
      → ALL question types route through generate_agentic_answer():
          - ROOT_CAUSE → SYSTEM_AGENTIC_REASONING prompt
          - CROSS_ANALYSIS → SYSTEM_CROSS_ANALYSIS prompt
          - LOG_ANALYSIS → SYSTEM_LOG_SUMMARY prompt
          - CODE_REVIEW → SYSTEM_CODE_REVIEW prompt
          - GENERAL → SYSTEM_RAG prompt
      → Quality gate checks response:
          - Min length (120 chars)
          - Truncation detection (missing terminal punctuation, truncation signals)
          - Auto-retry with 2x token budget if quality check fails
      → Structured answer returned with confidence scores and source citations
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
- All uncommitted changes listed in Section 6 are **tested and working** as of 2026-03-07.
- **Reasoning Engine v5**: All question types now receive the full evidence context (logs + code + deps + architecture + cross-analysis). No evidence is discarded at the last mile. The response quality gate ensures LLM output is complete. Token budget of 2048 allows deep multi-section analysis.
- **Hybrid Router v6**: Query routing uses LLM classification when available, with rule-based fallback. Conversational/unrelated queries are always short-circuited. Ingestion now excludes model-weights, vectordb, and binary files. Retrieval guard prevents low-confidence extractive matches from being surfaced.

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

---

*This file is meant to be read by AI assistants. Feed it at the start of a new session to restore full project context.*
