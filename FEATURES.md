# bakup.ai — Feature List

Comprehensive list of all implemented features in bakup.ai — the AI-powered project intelligence and production support assistant.

---

## Core Platform

| Feature | Status | Description |
|---------|--------|-------------|
| FastAPI backend | ✅ | REST API with health, index, ask, debug, download routes |
| ChromaDB vector store | ✅ | Persistent vector storage with metadata-enriched chunks |
| Sentence-transformers embedding | ✅ | Local embedding model (downloaded on first run, ~90 MB) |
| Static UI serving | ✅ | HTML/CSS/JS app served via FastAPI StaticFiles mount |
| Landing page | ✅ | Separate marketing page with download link |
| Windows installer | ✅ | Inno Setup packaged binary (~240 MB, LZMA2 compressed) |
| Port auto-detection | ✅ | Connect + bind dual-phase check, scans 8000-8019, uvicorn retry loop |
| Port conflict recovery | ✅ | Catches Errno 10048 at bind time, retries up to 5 ports automatically |
| Access key authentication | ✅ | SHA-256 hash check at startup |

---

## Ingestion Pipeline

| Feature | Status | Description |
|---------|--------|-------------|
| Local project indexing | ✅ | Walks directory tree, chunks files, embeds, stores in ChromaDB |
| GitHub repository indexing | ✅ | Shallow clone → walk → chunk → embed → store (public + private repos) |
| GitHub repo metadata | ✅ | Extracts repo_name, branch, commit_hash, commit_author, commit_message |
| Log file ingestion | ✅ | Parses log files with timestamp/severity extraction |
| Language-aware code parsing | ✅ | Regex-based parser for 11 languages (Python, JS, TS, Go, Java, JSON, YAML, TOML, INI) |
| Code unit extraction | ✅ | Detects functions, classes, methods, imports, decorators, docstrings |
| Context header injection | ✅ | Prepends file/language/class/import info to each chunk |
| Code metadata enrichment | ✅ | 7 metadata fields: language, function_name, class_name, chunk_kind, docstring, imports, metadata |
| Symbol graph building | ✅ | In-memory graph of function calls, imports, class methods per namespace |
| Architecture summary | ✅ | Auto-generated project overview: entry points, modules, dependencies, dir tree |

---

## Retrieval & RAG Pipeline

| Feature | Status | Description |
|---------|--------|-------------|
| Semantic search | ✅ | Embedding-based similarity search via ChromaDB |
| Keyword-enhanced search | ✅ | Augments semantic search with log-keyword matching for error queries |
| Severity-based search | ✅ | Pulls error-severity chunks for cross-file distribution analysis |
| Confidence-based ranking | ✅ | Re-ranks results with code-aware boosting (+0.05 for functions/classes) |
| Context bundling | ✅ | Groups primary + sibling + import chunks for coherent LLM context |
| Symbol graph queries | ✅ | Answers structural questions without LLM (which files use X?, what depends on Y?) |
| Architecture queries | ✅ | Serves cached project overview for "explain the architecture" questions |
| Log-to-code cross analysis | ✅ | Links log errors to source code via stack trace / file:line extraction |
| **Agentic retrieval planner** | ✅ | Classifies question type and generates multi-step retrieval plans |
| **Multi-step agent executor** | ✅ | Executes retrieval plans sequentially, building structured evidence |
| **Structured evidence bundling** | ✅ | Organises evidence by category: logs, code, deps, architecture, cross-analysis |
| **Unified evidence routing** | ✅ | ALL question types receive the full evidence context — no data discarded at the last mile |
| **Root-cause reasoning mode** | ✅ | Auto-correlates stack traces with code, retrieves deps, asks LLM for root cause |
| **Code review mode** | ✅ | Broad analytical questions (optimize, review, improve) route through specialised code-review prompt |
| **Session memory** | ✅ | Per-namespace conversation context for follow-up question support |
| **Follow-up detection** | ✅ | Heuristic detection of follow-up questions using pronouns, short queries, markers |
| **Error pattern clustering** | ✅ | Groups similar errors by exception type, message similarity, stack trace, file+line |
| **Time-based trend detection** | ✅ | Per-cluster 1h/24h window analysis with spike, regression, and new error detection |
| **Causal confidence scoring** | ✅ | 5-factor weighted score (0–100): frequency, stack trace, code match, deps, trend |
| **Evidence ranking** | ✅ | Ranks clusters by severity × count × trend × reference clarity before LLM |
| **Structured reasoning input** | ✅ | Injects dominant cluster, frequency stats, time trends, confidence into LLM prompt |

---

## Query Classification

| Feature | Status | Description |
|---------|--------|-------------|
| Regex-based classifier | ✅ | Routes queries: project → RAG, greeting → polite response, off_topic → scope guard |
| Conversational handling | ✅ | Meta/personal questions routed to LLM without retrieval |
| Architecture detection | ✅ | Detects "architecture", "project structure", "explain the project" |
| Structural detection | ✅ | Detects "which files use/import X?", "what depends on X?" |
| Log query detection | ✅ | Detects error/exception/traceback/crash keywords |
| **Code review detection** | ✅ | Detects "optimized", "review", "improve", "best practices", "code quality" patterns |
| **Agentic question typing** | ✅ | Classifies: log_analysis, code_analysis, code_review, root_cause, architecture, structural, general |
| **Default-to-project routing** | ✅ | Ambiguous questions default to PROJECT classification instead of being rejected |

---

## LLM Integration

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-provider support | ✅ | OpenAI, Anthropic (Claude), Azure OpenAI, Ollama, local llama.cpp |
| Anthropic Claude support | ✅ | Native Messages API integration via urllib (no extra dependency) |
| Configurable via API | ✅ | `POST /llm/configure` with provider, model, API key |
| SSE streaming | ✅ | `POST /ask/stream` returns Server-Sent Events for real-time output |
| Extractive fallback | ✅ | Returns structured top-5 chunks when LLM is not configured (no hallucination) |
| Citation-enforced prompts | ✅ | System prompts require source citations and NO_ANSWER signal |
| Log summarization prompt | ✅ | Structured incident report: summary, findings, patterns, distribution |
| Cross-analysis prompt | ✅ | Root-cause analysis: error→code mapping, call chains, suggested fixes |
| Code review prompt | ✅ | Structured code quality review: strengths, issues, architecture, recommendations |
| Conversational prompt | ✅ | Professional identity-aware responses for meta questions |
| Clarification mode | ✅ | Asks user to refine query when confidence is low |
| **Agentic reasoning prompt** | ✅ | Multi-step root-cause analysis with evidence chain, impact assessment, recommendations |
| **Unified agentic answer generation** | ✅ | All question types route through a single reasoning method with mode-appropriate system prompts |
| **Response quality gate** | ✅ | Detects truncated or too-short responses and auto-retries with a larger token budget |
| **Deep context window** | ✅ | 16K context window with 2048 max output tokens (4x increase from initial config) |
| **Rich evidence context** | ✅ | Up to 12 log chunks + 12 code chunks + 2500-char architecture + deps + cross-analysis sent to LLM |

---

## Brain Architecture (LLM-Orchestrated Reasoning)

| Feature | Status | Description |
|---------|--------|-------------|
| **Brain controller** | ✅ | Central `process_query()` orchestrates LLM tool-calling loop or falls back to deterministic pipeline |
| **Tool interface layer** | ✅ | 8 tools wrapping retrieval/analysis: search_logs, search_code, retrieve_dependencies, get_architecture_summary, get_error_clusters, get_file_context, query_symbol_graph, cross_analyse |
| **LLM tool calling** | ✅ | `call_with_tools()` supports OpenAI, Anthropic, Azure OpenAI, Ollama function-calling formats |
| **Tool loop** | ✅ | Iterative tool-call loop: LLM requests tools → tools execute → results fed back → LLM reasons |
| **Tool budget** | ✅ | `max_tool_calls=5` per query prevents runaway tool usage |
| **Auto-inject namespace** | ✅ | Brain auto-injects namespace and top_k into tool arguments |
| **Graceful fallback** | ✅ | Falls back to deterministic RAG pipeline when LLM is not configured or doesn't support tools |
| **Brain debug endpoint** | ✅ | `GET /debug/brain` shows brain state, tools, provider; `GET /debug/brain/{ns}` shows last result |
| **Brain debug cache** | ✅ | Last brain result per namespace stored for inspection via debug API |
| **Provider tool format** | ✅ | Tool schemas auto-converted to OpenAI or Anthropic format based on provider |
| **Cloud mode config** | ✅ | `BAKUP_APP_MODE=local|cloud` env var for deployment mode selection |
| **Session context in brain** | ✅ | Follow-up detection injects prior conversation into brain's LLM context |

---

## Analysis Pipeline

| Feature | Status | Description |
|---------|--------|-------------|
| Error trend detection | ✅ | Hourly error counts, spike detection, repeating failure identification |
| Incident clustering | ✅ | Groups log events by temporal proximity and keywords |
| Multi-factor confidence scoring | ✅ | Combines embedding similarity, keyword overlap, cross-file factors |
| File-level error aggregation | ✅ | Ranks files by error count, identifies dominant error sources |
| Code reference extraction | ✅ | 7 regex patterns for Python/Java/JS/Go stack traces, file:line, Class.method |

---

## Debug & Diagnostics

| Feature | Status | Description |
|---------|--------|-------------|
| Index diagnostics | ✅ | `GET /debug/index/{ns}` — chunk count, source type breakdown, samples |
| Retrieval diagnostics | ✅ | `POST /debug/retrieval` — full pipeline trace without answer generation |
| Symbol graph diagnostics | ✅ | `GET /debug/symbols/{ns}` — node/edge counts, top files, top imports |
| Architecture diagnostics | ✅ | `GET /debug/architecture/{ns}` — full architecture summary as JSON |
| Pipeline step tracing | ✅ | `debug=true` on `/ask` returns step-by-step timing and data |
| **Agentic plan diagnostics** | ✅ | `POST /debug/plan` — shows retrieval plan without executing |
| **Session diagnostics** | ✅ | `GET /debug/session/{ns}` — shows session memory state and turns |
| **Session clear** | ✅ | `POST /debug/session/{ns}/clear` — clears session memory |
| **Agent step tracing** | ✅ | Each agent step logged with timing, chunk counts, and evidence state |
| **Error cluster diagnostics** | ✅ | `POST /debug/clusters` — error pattern clusters with signatures, counts |
| **Causal confidence diagnostics** | ✅ | `POST /debug/causal-confidence` — full scoring pipeline with factor breakdown |
| **Trend detection diagnostics** | ✅ | `POST /debug/trends` — per-cluster time trends, spike/regression alerts |
| **Causal confidence in pipeline trace** | ✅ | Score, dominant error, and trend alerts in debug trace |
| **Brain diagnostics** | ✅ | `GET /debug/brain` shows LLM state, tool list, brain mode; `GET /debug/brain/{ns}` shows last result with tool call trace |
| Console debug logging | ✅ | Detailed per-request logging: classification, retrieval, LLM calls |

---

## Security

| Feature | Status | Description |
|---------|--------|-------------|
| Localhost-only binding | ✅ | Server binds to 127.0.0.1, not 0.0.0.0 |
| Access key validation | ✅ | SHA-256 hash check before server starts |
| No credential logging | ✅ | GitHub URLs sanitized before logging |
| Download security headers | ✅ | X-Content-Type-Options, Cache-Control, Content-Disposition |
| HEAD support for /download | ✅ | CDN probes and link checkers handled correctly |
| No UPX packing | ✅ | Avoids AV false positives from packed binaries |
| Version info embedded | ✅ | Binary has CompanyName, FileDescription metadata |

---

## Build & Distribution

| Feature | Status | Description |
|---------|--------|-------------|
| PyInstaller compilation | ✅ | Folder distribution with hidden imports and data bundling |
| Inno Setup installer | ✅ | LZMA2 compressed, Start Menu + Desktop shortcuts |
| Build orchestrator script | ✅ | `build.ps1` with skip flags and verbose mode |
| Post-build cleanup | ✅ | Removes test/debug artifacts (~23 MB) |
| AV false-positive mitigation | ✅ | No UPX, version info, console mode, no GPU binaries |
| ASCII-safe console output | ✅ | All print statements use ASCII dashes (no Unicode em-dashes) |
| UTF-8 console support | ✅ | Launcher sets `chcp 65001` for Windows code page compatibility |

---

## Production Deployment (v7)

| Feature | Status | Description |
|---------|--------|-------------|
| Docker Compose volumes | ✅ | Named volumes (`bakup-vectordb`, `bakup-models`) for data persistence |
| Nginx SSE proxy | ✅ | `/ask/stream` with `proxy_buffering off` for real-time SSE |
| Nginx debug/download routes | ✅ | `/debug/*` and `/download` proxy routes added |
| CORS environment variable | ✅ | `BAKUP_CORS_ORIGINS` env var for production domain whitelisting |
| Memory limits | ✅ | Backend 4G, UI 2G container memory limits |
| DEPLOY.md guide | ✅ | Complete deployment guide for VPS/Railway/Render |
| `.dockerignore` | ✅ | Excludes vectordb, model-weights, .git from build context |

---

## Hosting & Distribution (v8)

| Feature | Status | Description |
|---------|--------|-------------|
| GitHub Pages landing page | ✅ | Live at `https://salilsankritya.github.io/bakup.ai` |
| GitHub Releases installer | ✅ | 241 MB installer hosted on GitHub Releases (v0.2.0) |
| Auto-follow latest release | ✅ | Download URL uses `/releases/latest/download/` pattern |
| Key-gated download | ✅ | Landing page requires access key before revealing download link |
| `.nojekyll` | ✅ | Prevents Jekyll processing on GitHub Pages |
| HOSTING.md guide | ✅ | Complete 0-to-100 hosting setup guide |

---

*Last updated: v9 Brain Architecture — LLM-orchestrated tool-calling + Anthropic provider*
