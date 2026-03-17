# Parsing Strategy & Known Limitations

> How bakup.ai ingests, chunks, and indexes source code for semantic retrieval.

---

## Architecture

```
  Source Files
       │
       ▼
  file_walker.py          ← discovers files, skips binaries/vendor dirs
       │
       ├── Code files ──▶  code_chunker.py
       │                        │
       │                        ▼
       │                   code_parser.py   ← language-aware structure extraction
       │                        │
       │                        ▼
       │                   Chunk objects (with function_name, class_name, imports…)
       │
       └── Log files ───▶  log_parser.py    ← timestamp-boundary splitting
                                │
                                ▼
                           Chunk objects (with severity, timestamps…)
```

All chunks flow into **ChromaDB** via `vector_store.py` for semantic search.

---

## Parsing Approach

bakup.ai uses **regex-based parsing with indentation / brace tracking**. There is no dependency on external AST libraries or tree-sitter. This keeps the installation simple (zero native dependencies) at the cost of some edge-case accuracy.

### What the Parser Extracts

For each supported language, the parser identifies:

| Element | Description |
|---------|-------------|
| **Functions** | Standalone functions, arrow functions, closures |
| **Classes** | Class/struct/trait/interface/enum/impl definitions |
| **Methods** | Functions defined inside classes |
| **Imports** | import / require / use / include statements |
| **Docstrings** | Python docstrings, JSDoc comments |
| **Decorators** | Python `@decorator` lines |
| **Module-level code** | Top-level statements not belonging to any function/class |

Each extracted unit becomes a **Chunk** with rich metadata:

```
Chunk(
    text          = "# File: api/routes/index.py\n# Language: python\nimport ...\ndef index_local(...):\n    ...",
    source_file   = "api/routes/index.py",
    line_start    = 45,
    line_end      = 98,
    source_type   = "code",
    language      = "python",
    function_name = "index_local",
    class_name    = "",
    chunk_kind    = "function",
    docstring     = "Index a local directory...",
    imports       = "from pathlib import Path\nimport logging",
)
```

### Context Headers

Every chunk is prepended with a context header so the embedding model and LLM can locate it:

```python
# File: core/retrieval/vector_store.py
# Language: python
# Class: VectorStore         ← only for methods
from __future__ import annotations
import chromadb
```

---

## Supported Languages

### Fully Supported (structural parsing)

| Language | Extensions | Parser | What's Extracted |
|----------|-----------|--------|-----------------|
| Python | `.py` | `parse_python` | classes, methods, functions, decorators, docstrings, imports |
| JavaScript | `.js` | `parse_javascript` | classes, named functions, arrow functions, JSDoc, imports |
| TypeScript | `.ts` | `parse_javascript` | (same as JS) |
| JSX / TSX | `.jsx`, `.tsx` | `parse_javascript` | (same as JS) |
| Go | `.go` | `parse_go` | structs, functions, imports |
| Java | `.java` | `parse_java` | classes, methods, imports |
| C | `.c`, `.h` | `parse_c_family` | structs, functions, includes |
| C++ | `.cpp`, `.hpp`, `.cc`, `.cxx` | `parse_c_family` | classes, functions, includes |
| C# | `.cs` | `parse_c_family` | classes, methods, usings |
| Rust | `.rs` | `parse_c_family` | structs, enums, traits, impls, fns, use statements |
| Kotlin | `.kt` | `parse_c_family` | classes, functions, imports |
| Swift | `.swift` | `parse_c_family` | classes/structs, functions |
| Scala | `.scala` | `parse_c_family` | classes/objects/traits, functions |
| Ruby | `.rb` | `parse_ruby` | classes, modules, defs, requires |
| PHP | `.php` | `parse_php` | classes, functions, use/require |
| Shell | `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1` | `parse_shell` | functions |
| JSON | `.json` | `parse_json` | top-level keys |
| YAML | `.yaml`, `.yml` | `parse_config` | sections |
| TOML | `.toml` | `parse_config` | tables |
| INI / Config | `.ini`, `.cfg`, `.conf`, `.env` | `parse_config` | sections |

### Fallback (line-window chunking)

Files with recognized text extensions but no language-specific parser fall back to **naive line-window chunking**: 40-line windows with 8-line overlap. These still get `# File:` context headers but have no `function_name` or `class_name` metadata.

Affected extensions: `.lua`, `.ex`, `.exs`, `.clj`, `.hs`, `.sql`, `.graphql`, `.proto`, `.html`, `.css`, `.scss`, `.xml`, `.md`, `.rst`, `.txt`

### Skipped Entirely

- **Directories**: `node_modules`, `.venv`, `__pycache__`, `dist`, `build`, `.git`, `.svn`, `.next`, `vendor`, `target`, `coverage`, `egg-info`, etc. (24 patterns)
- **Extensions**: `.pyc`, `.exe`, `.dll`, `.so`, `.png`, `.jpg`, `.pdf`, `.zip`, `.tar`, `.gguf`, `.bin`, `.model`, `.pt`, `.onnx`, etc. (30+ patterns)
- **Files > 512 KB**: Skipped to prevent memory issues

---

## Chunking Strategy

### Structure-Aware Chunking (default for supported languages)

1. **Parse** the file into logical `CodeUnit` objects (functions, classes, methods)
2. **Build context header** for each unit (file path, language, class name, imports)
3. **Split large units** (>80 lines) into overlapping chunks (8-line overlap)
4. **Merge small units** (<10 lines) with neighbors when adjacent
5. **Capture module-level code** (imports, constants, top-level statements) as a separate chunk

### Line-Window Chunking (fallback)

For unsupported languages:
- 40-line windows with 8-line overlap
- Minimum chunk size: 60 characters
- Context header still prepended

### Log File Chunking

For `.log` and `.out` files:
- Split on timestamp boundaries (detects common log timestamp patterns)
- Enrich with severity detection (error / warning / info)
- Capture timestamps for temporal queries

---

## API Response Metadata

The `/ask` endpoint returns sources with rich metadata:

```json
{
  "sources": [
    {
      "file": "core/retrieval/vector_store.py",
      "line_start": 60,
      "line_end": 66,
      "excerpt": "def _get_or_create_collection(name)...",
      "confidence": 0.68,
      "confidence_label": "medium",
      "source_type": "code",
      "function_name": "_get_or_create_collection",
      "class_name": "",
      "chunk_kind": "function",
      "language": "python"
    }
  ]
}
```

---

## Known Limitations

1. **No AST parsing** — The regex-based approach handles well-formatted code but may miss:
   - Heavily nested closures in JS/TS
   - Python multi-line function signatures across many lines
   - Decorator arguments spanning multiple lines
   - Template literals containing code-like patterns
   - Macro-heavy C/C++ code

2. **No cross-file analysis** — Each file is parsed independently. The parser does not resolve imports across files or build a call graph.

3. **Fallback languages** — Lua, Elixir, Clojure, Haskell, SQL, HTML, CSS, and Markdown get line-window chunking without structural metadata.

4. **Max file size** — Files larger than 512 KB are skipped entirely. This may exclude large generated files or data files.

5. **Docstring extraction** — Only Python triple-quote docstrings and JSDoc `/** */` comments are extracted as structured metadata. Other documentation patterns (Ruby YARD, Go comments, Javadoc) are included in chunk text but not in the `docstring` field.

---

## Performance Characteristics

Tested on bakup.ai's own backend (59 Python files):

| Metric | Value |
|--------|-------|
| Total chunks | 621 |
| Chunk kinds | class: 85, function: 268, method: 115, module: 153 |
| Fallback chunks | 0 (100% structure-aware) |
| Imports coverage | 99.5% |
| Docstring coverage | 41% |
| Context header coverage | 100% |
| Median chunk size | 788 chars |
| Avg chunk size | 1,280 chars |
| Distinct symbols | 32 classes, 247 functions, 115 methods |
