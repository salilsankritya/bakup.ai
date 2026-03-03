# bakup.ai — AI Context File

> **Last updated:** 2026-03-04
> **Purpose:** Provides a complete snapshot of the project for any AI assistant to resume work without loss of context.

---

## 1. What is bakup.ai?

bakup.ai is a **local-first AI incident intelligence tool** for developers. It indexes project source code and log files, stores them as vector embeddings, and lets engineers ask natural-language questions about errors, incidents, and code — with answers grounded strictly in their own data. No data ever leaves the machine (unless an external LLM like OpenAI is configured).

### Core Workflow
```
User uploads/indexes a project folder
  → Files are scanned recursively (code + logs)
  → Text is chunked, embedded (all-MiniLM-L6-v2, dim=384), stored in ChromaDB
  → User asks questions via chat UI
  → Query is classified (project / greeting / conversational / off_topic)
  → For project queries: embed → retrieve → rank → analyze → answer
  → For log queries: keyword search + severity search + trend analysis + clustering + file aggregation → LLM summary
```

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
│   │   └── rag.py          ← Full RAG pipeline (classify → embed → retrieve → rank → answer)
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
│   └── core/llm/
│       ├── config_store.py ← JSON storage for LLM config (API keys masked)
│       ├── llm_service.py  ← LLM abstraction (OpenAI, Azure, Ollama)
│       └── prompt_templates.py ← All system prompts and context builders
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

**Total:** ~34 Python files, ~4,400 lines backend / ~2,400 lines frontend

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
| `5693209` | feat: log intelligence upgrade — multi-factor confidence, trend detection, clustering | Latest committed |
| `3197629` | feat: add query classifier with conversational/meta routing | |
| `4d43282` | feat: bakup.ai — full platform with hybrid retrieval, debug diagnostics & SSE streaming | |
| `0757dd7` | Initial commit | |

### Uncommitted Changes (as of 2026-03-04)

These changes span **three feature sets** that haven't been committed yet:

#### A. Enter-to-Send UI Fix
- `ui/app.js` — Changed keydown from Ctrl+Enter to plain Enter (Shift+Enter for newline)
- `ui/index.html` — Updated button tooltip

#### B. Path Handling Fix
- `backend/api/routes/index.py` — Added `_normalize_path()`, `_validate_path()`, Docker volume enforcement
- `ui/app.js` — Fixed folder picker separator detection with `lastIndexOf`
- `docker-compose.yml` — Added `./projects` and `./logs` volume mounts, `BAKUP_DOCKER_VOLUMES` env var

#### C. Multi-File Log Intelligence Upgrade (Major)
Files modified:
- `backend/core/ingestion/chunker.py` — Added `file_name`, `last_modified`, `detected_timestamp`, `severity` to Chunk dataclass
- `backend/core/ingestion/log_parser.py` — Added `_detect_severity()`, `_extract_first_timestamp()`, `_get_file_mtime_iso()`, severity tagging per chunk, metadata enrichment in `_make_chunk()` and `_fallback_chunk()`
- `backend/core/ingestion/file_walker.py` — Added `.out` to `LOG_EXTENSIONS` and `TEXT_EXTENSIONS`
- `backend/core/retrieval/models.py` — Added `file_name`, `severity`, `detected_timestamp` to RetrievedChunk
- `backend/core/retrieval/vector_store.py` — Stores all new metadata; added `severity_search()` method
- `backend/core/retrieval/ranker.py` — Added `file_name`, `severity`, `detected_timestamp` to RankedResult; propagates through `rank_results()`
- `backend/core/retrieval/rag.py` — Wired severity search + file aggregation into both high/low confidence log paths; added enhanced debug logging
- `backend/core/analysis/confidence.py` — Added 6th factor: cross-file consistency (errors across multiple files boost confidence)
- `backend/core/analysis/__init__.py` — Re-exports `aggregate_by_file`, `FileAggregationReport`
- `backend/core/llm/prompt_templates.py` — Added "Error Distribution" section to SYSTEM_LOG_SUMMARY; `build_log_analysis_context()` accepts `file_aggregation_summary`
- `backend/core/llm/llm_service.py` — `generate_log_summary()` accepts `file_aggregation_summary` parameter
- `backend/api/routes/query.py` — Wired severity search + file aggregation into SSE streaming paths

New file:
- `backend/core/analysis/file_aggregation.py` — Cross-file error distribution (error counts per file, ranking, dominant source, summary text)

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
  → classify_query() → PROJECT | GREETING | CONVERSATIONAL | OFF_TOPIC
  → If GREETING → polite one-liner (no RAG)
  → If CONVERSATIONAL → LLM direct call (no retrieval)
  → If OFF_TOPIC → scope guard response
  → If PROJECT:
      → embed_query() → 384-dim vector
      → query_chunks() → top_k semantic matches
      → If log query:
          → keyword_search() — $contains for ERROR, Exception, Traceback, etc.
          → severity_search() — metadata filter for severity="error"
      → Merge + deduplicate
      → rank_results() → distance-to-confidence conversion, sort desc
      → If has_relevant_results (top confidence ≥ 0.35):
          → For log queries:
              → analyze_error_trends() → hourly counts, spikes, repeating failures
              → cluster_log_events() → temporal + keyword clustering
              → calculate_confidence() → 6-factor composite score
              → aggregate_by_file() → error distribution across files
              → generate_log_summary() → LLM structured report (or extractive fallback)
          → For code queries:
              → generate_response() → LLM answer with citations (or extractive)
      → If low confidence + log query → still run analysis pipeline, summarize
      → If low confidence + code query → ask clarifying question
      → If no results → "No similar incident found."
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
| POST | `/debug/search` | Raw similarity search |

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

- **Ollama** must be running locally for LLM-augmented answers. If not available, the system falls back to **extractive mode** (returns raw chunk excerpts — no hallucination possible).
- The `model-weights/` directory caches the embedding model on first run (~80MB download).
- The `vectordb/` directory persists ChromaDB data between restarts.
- After code changes, **re-index** the project to populate new metadata fields (severity, file_name, etc.) in existing ChromaDB data.
- The UI communicates with the backend via `http://localhost:8000` (hardcoded in `ui/app.js`).
- All uncommitted changes listed in Section 6 are **tested and working** as of 2026-03-04.

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
1. **Commit all uncommitted changes** — three feature sets ready to push
2. **README.md** — currently a stub, needs proper documentation
3. **Folder dedup tracking** — prevent re-indexing the same folder (hash-based)
4. **Multi-log-folder support** — allow indexing dedicated log directories separately
5. **UI improvements** — file tree view, severity badges, error distribution charts
6. **Auth improvements** — token-based auth instead of simple access key
7. **Rate limiting** — protect endpoints from abuse
8. **Automated tests** — pytest suite with fixtures

---

*This file is meant to be read by AI assistants. Feed it at the start of a new session to restore full project context.*
