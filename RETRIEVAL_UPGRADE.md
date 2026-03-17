# Retrieval Quality Upgrade — v2

This document describes the safe, non-breaking enhancements applied to
bakup.ai's retrieval pipeline to improve answer quality, evidence clarity,
and demo reliability.

## Summary of Changes

| Area | What Changed | Files Modified |
|------|-------------|----------------|
| Ingestion Filters | Expanded skip rules (dirs, extensions, filenames, suffixes) | `file_walker.py` |
| Multi-Query Retrieval | RRF-based multi-query expansion with rule-based variants | NEW: `multi_query.py` |
| Evidence Deduplication | Exact + overlap dedup, per-category limits | NEW: `dedup.py` |
| Evidence Reranking | Error keyword, stack trace, recency boosting | `ranker.py` |
| Structured Evidence | Safety-limited context bundling with token awareness | `agent.py` |
| Explanation Format | 5-section structured answer format | `prompt_templates.py` |
| Safety Limits | Max total context chars, chunk truncation | `agent.py` |
| Debug Visibility | Multi-query variants, evidence scores, context size logging | `rag.py` |

---

## 1. Ingestion Filters (`file_walker.py`)

### New Skip Directories
- `out`, `.output`, `.gradle`, `.maven`, `.cargo`, `vendor`,
  `bower_components`, `.terraform`, `.serverless`, `.webpack`,
  `.parcel-cache`, `site-packages`, `.sass-cache`, `tmp`, `.tmp`

### New Skip Extensions
- ML: `.h5`, `.hdf5`, `.tflite`, `.pb`
- Serialised: `.parquet`, `.feather`
- Binaries: `.o`, `.obj`, `.a`, `.lib`
- Archives: `.xz`, `.rar`
- Images: `.bmp`, `.tiff`, `.psd`
- AV: `.mov`, `.flv`, `.mkv`
- Fonts: `.otf`
- Source maps: `.map`
- Lock files: `.lock`
- Databases: `.db`, `.sqlite`, `.sqlite3`
- Compiled: `.pyc`, `.pyo`, `.class`, `.jar`, `.war`, `.ear`, `.nupkg`
- Coverage: `.coverage`, `.prof`, `.trace`

### New Skip Filenames
`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Pipfile.lock`,
`poetry.lock`, `Gemfile.lock`, `composer.lock`, `Cargo.lock`, `go.sum`,
`.DS_Store`, `Thumbs.db`, `desktop.ini`

### New Skip Suffixes
`.min.js`, `.min.css`, `.bundle.js`, `.chunk.js`, `.min.map`, `.bundle.map`

---

## 2. Multi-Query Retrieval (`multi_query.py`)

### How It Works
1. Takes the user's original question
2. Generates 2–4 deterministic query variants:
   - **Keyword-only**: strip stop-words, keep content words
   - **Synonym-expanded**: replace one keyword with a technical synonym
   - **Technical focus**: extract dotted identifiers, CamelCase names,
     file paths, and error patterns
3. Embeds each variant and runs vector search
4. Merges results using **Reciprocal Rank Fusion (RRF)** with k=60
5. Returns deduplicated chunks sorted by fused score

### Why Rule-Based (Not LLM)
Multi-query variants are generated deterministically so the feature works
even when no LLM provider is configured. This ensures consistent behaviour
in local/offline mode.

---

## 3. Evidence Deduplication (`dedup.py`)

### Three Strategies
1. **Exact dedup** — identical `(source_file, line_start, line_end)` tuples
2. **Overlap merging** — same file with overlapping line ranges → keep
   the higher-confidence chunk
3. **Limit enforcement** — per-category caps:
   - `max_logs = 5` (configurable via `BAKUP_MAX_EVIDENCE_LOGS`)
   - `max_code = 5` (configurable via `BAKUP_MAX_EVIDENCE_CODE`)

### Where Applied
- Agent pipeline (`agent.py`): `_execute_search_logs`, `_execute_search_code`
- SSE streaming route (`query.py`): `/ask/stream` step 6

---

## 4. Evidence Reranking (`ranker.py`)

### Multi-Signal Boosting (v2)
| Signal | Boost | Condition |
|--------|-------|-----------|
| Code structure | +0.05 | `chunk_kind` is function/method/class |
| Docstring | +0.02 | Chunk has a docstring |
| Error keyword | +0.08 | Text contains ERROR, Exception, Traceback, etc. |
| Stack trace | +0.06 | Text contains stack frame patterns |
| Recency (24h) | +0.03 | Timestamp within last 24 hours |
| Recency (7d) | +0.01 | Timestamp within last 7 days |

All boosts are additive and capped at 1.0.

### Error Keyword Patterns
`ERROR`, `Exception`, `Traceback`, `FATAL`, `CRITICAL`, `FAILED`,
`panic`, `segfault`, `OutOfMemoryError`, `NullPointerException`,
`StackOverflowError`, `RuntimeError`, `ValueError`, `TypeError`,
`KeyError`, `AttributeError`, `IOError`, `OSError`, `ConnectionError`,
`TimeoutError`

### Stack Trace Patterns
- Python: `File "...", line N`
- Java/JS: `at Module.func(file:line)`
- Go: `goroutine N`
- C/Rust: binary addresses

---

## 5. Explanation Format (`prompt_templates.py`)

### Structured Answer Sections
Every LLM answer now follows this format:

```
### Problem Summary
What the question is about and what was found.

### Evidence Found
- Specific evidence with citations (source: file, lines N–M)

### Root Cause Analysis
Why the issue occurs, based on evidence.

### Recommended Next Step
1–3 concrete actions.

### Confidence
**Confidence: High | Medium | Low**
```

---

## 6. Safety Limits (`agent.py`)

### Context Window Safety
- `max_total_chars = 12000` — total context sent to LLM is capped
- Sections are added in priority order (logs → code → deps → arch →
  cross-analysis → analysis → refs → structured reasoning)
- When the limit is hit, remaining sections are truncated with
  `[...context truncated for safety]`
- Each chunk is still individually capped at `max_chars = 1200`

### Why 12,000 chars?
At ~4 chars per token, this is ~3,000 tokens of context, leaving
ample room for the system prompt (~800 tokens) and response generation
(~2,048 tokens) within a 16K context window.

---

## 7. Debug Visibility (`rag.py`)

### New Debug Steps
| Step | What It Logs |
|------|-------------|
| `multi_query` | Generated query variants |
| `evidence_scores` | Top 5 evidence chunks with confidence scores |
| `context_size` | Final context chars and estimated token count |
| `agent_*` | Per-step execution time and evidence accumulation |

### Console Output
When `debug=True`, these steps appear in `debug_trace` on the API response.
They are also always printed to console (`[bakup:debug]` prefix).

---

## No Breaking Changes (Constraint #9)

### What Was NOT Modified
- Planner logic (`planner.py`)
- Agent workflow architecture (`agent.py` structure, `execute_plan` flow)
- Indexing pipeline (`index.py`, `chunker.py`, `code_parser.py`,
  `code_chunker.py`)
- Existing embeddings and ChromaDB storage format
- API request/response schemas (`QueryRequest`, `QueryResponse`,
  `SourceModel`)
- Brain controller logic (`brain.py`)
- Session management (`session.py`)

### Backward Compatibility
- All existing API contracts preserved
- All existing retrieval paths still work
- Multi-query is additive (takes single query, produces better results)
- Enhanced reranking only adds to confidence, never reduces it
- Dedup only removes redundancy, never removes unique evidence
