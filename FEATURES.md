# bakup.ai — Feature List

Comprehensive list of all implemented features in bakup.ai.

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
| Port auto-detection | ✅ | Scans 8000–8019 for free port, updates CORS origins |
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

---

## Query Classification

| Feature | Status | Description |
|---------|--------|-------------|
| Regex-based classifier | ✅ | Routes queries: project → RAG, greeting → polite response, off_topic → scope guard |
| Conversational handling | ✅ | Meta/personal questions routed to LLM without retrieval |
| Architecture detection | ✅ | Detects "architecture", "project structure", "explain the project" |
| Structural detection | ✅ | Detects "which files use/import X?", "what depends on X?" |
| Log query detection | ✅ | Detects error/exception/traceback/crash keywords |

---

## LLM Integration

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-provider support | ✅ | OpenAI, Anthropic, Google Gemini, OpenRouter, Ollama, local llama.cpp |
| Configurable via API | ✅ | `POST /llm/configure` with provider, model, API key |
| SSE streaming | ✅ | `POST /ask/stream` returns Server-Sent Events for real-time output |
| Extractive fallback | ✅ | Returns raw chunks when LLM is not configured (no hallucination) |
| Citation-enforced prompts | ✅ | System prompts require source citations and NO_ANSWER signal |
| Log summarization prompt | ✅ | Structured incident report: summary, findings, patterns, distribution |
| Cross-analysis prompt | ✅ | Root-cause analysis: error→code mapping, call chains, suggested fixes |
| Conversational prompt | ✅ | Professional identity-aware responses for meta questions |
| Clarification mode | ✅ | Asks user to refine query when confidence is low |

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

---

*Last updated: Project Intelligence v2 upgrade*
