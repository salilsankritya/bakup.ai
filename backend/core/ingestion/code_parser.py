"""
core/ingestion/code_parser.py
─────────────────────────────────────────────────────────────────────────────
Language-aware code structure parser.

Detects language from file extension and extracts logical units:
  - functions, methods, classes, modules
  - imports, docstrings, comments
  - configuration blocks (JSON keys, YAML sections, TOML tables)

Uses regex-based parsing with indentation tracking — no external AST
dependencies, so it works with any Python installation.

Supported languages:
  Python, JavaScript, TypeScript, JSX, TSX, Go, Java, C, C++, C#,
  Rust, Kotlin, Swift, Scala, Ruby, PHP, Shell (bash/zsh/fish/ps1),
  JSON, and config files (.yaml, .yml, .toml, .ini, .cfg, .conf, .env).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Language detection ────────────────────────────────────────────────────────

LANGUAGE_MAP: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".jsx":   "jsx",
    ".ts":    "typescript",
    ".tsx":   "tsx",
    ".go":    "go",
    ".java":  "java",
    ".json":  "json",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".toml":  "toml",
    ".ini":   "config",
    ".cfg":   "config",
    ".conf":  "config",
    ".env":   "config",
    ".env.example": "config",
    # C-family / systems languages
    ".c":     "c",
    ".h":     "c",
    ".cpp":   "cpp",
    ".hpp":   "cpp",
    ".cc":    "cpp",
    ".cxx":   "cpp",
    ".cs":    "csharp",
    ".rs":    "rust",
    ".kt":    "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    # Scripting languages
    ".rb":    "ruby",
    ".php":   "php",
    ".sh":    "shell",
    ".bash":  "shell",
    ".zsh":   "shell",
    ".fish":  "shell",
    ".ps1":   "shell",
    # Remaining extensions get fallback "text" language
}


def detect_language(filepath: Path) -> str:
    """Detect programming language from file extension."""
    name = filepath.name.lower()
    # Check compound extensions first (e.g. .env.example)
    for ext, lang in LANGUAGE_MAP.items():
        if name.endswith(ext):
            return lang
    suffix = filepath.suffix.lower()
    return LANGUAGE_MAP.get(suffix, "text")


# ── Parsed structures ────────────────────────────────────────────────────────

@dataclass
class CodeUnit:
    """A single logical code unit extracted from a source file."""
    kind: str              # "function" | "class" | "method" | "module" | "config_block" | "import_block"
    name: str              # function/class name, or "" for module-level
    text: str              # full source text of this unit
    start_line: int        # 1-based
    end_line: int          # 1-based inclusive
    language: str          # "python", "javascript", etc.
    class_name: str = ""   # enclosing class name (for methods)
    docstring: str = ""    # extracted docstring/JSDoc
    imports: List[str] = field(default_factory=list)   # import lines in scope
    decorators: List[str] = field(default_factory=list) # decorators (Python)
    comments: str = ""     # leading comments


# ── Python parser ─────────────────────────────────────────────────────────────

_PY_DEF = re.compile(
    r"^([ \t]*)(async\s+)?def\s+(\w+)\s*\(",
    re.MULTILINE,
)
_PY_CLASS = re.compile(
    r"^([ \t]*)class\s+(\w+)\s*[\(:]",
    re.MULTILINE,
)
_PY_IMPORT = re.compile(
    r"^(?:from\s+\S+\s+)?import\s+.+",
    re.MULTILINE,
)
_PY_DECORATOR = re.compile(
    r"^([ \t]*)@\S+",
    re.MULTILINE,
)


def _extract_python_docstring(lines: List[str], start_idx: int) -> str:
    """Extract a triple-quoted docstring starting after a def/class line."""
    if start_idx >= len(lines):
        return ""
    line = lines[start_idx].strip()
    # Single-line docstring
    for q in ('"""', "'''"):
        if line.startswith(q) and line.endswith(q) and len(line) > 6:
            return line[3:-3].strip()
    # Multi-line docstring
    for q in ('"""', "'''"):
        if line.startswith(q):
            parts = [line[3:]]
            for i in range(start_idx + 1, min(start_idx + 30, len(lines))):
                if q in lines[i]:
                    parts.append(lines[i].split(q)[0])
                    return "\n".join(parts).strip()
                parts.append(lines[i].strip())
            return parts[0].strip() if parts else ""
    return ""


def _find_block_end_python(lines: List[str], start_idx: int, base_indent: int) -> int:
    """
    Find the last line of a Python block starting at start_idx.
    A block ends when a non-empty line at the same or lesser indent is found.
    """
    last = start_idx
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            # Blank line — could be inside block
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        if indent <= base_indent:
            break
        last = i
    return last


def _collect_leading_comments(lines: List[str], line_idx: int) -> str:
    """Collect comment lines immediately above a definition."""
    comments = []
    i = line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            comments.insert(0, stripped)
        elif not stripped:
            # Allow one blank line gap
            if i > 0 and lines[i - 1].strip().startswith("#"):
                i -= 1
                continue
            break
        else:
            break
        i -= 1
    return "\n".join(comments)


def _collect_decorators(lines: List[str], line_idx: int) -> List[str]:
    """Collect decorator lines immediately above a def/class."""
    decorators = []
    i = line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("@"):
            decorators.insert(0, stripped)
        elif not stripped:
            i -= 1
            continue
        else:
            break
        i -= 1
    return decorators


def parse_python(text: str, language: str = "python") -> List[CodeUnit]:
    """Parse Python source into logical code units."""
    lines = text.splitlines()
    if not lines:
        return []

    # Collect file-level imports
    file_imports = [
        m.group().strip()
        for m in re.finditer(r"^(?:from\s+\S+\s+)?import\s+.+", text, re.MULTILINE)
    ]

    units: List[CodeUnit] = []
    claimed: set[int] = set()  # line indices claimed by a unit

    # 1. Find classes and their methods
    for m in _PY_CLASS.finditer(text):
        class_indent = len(m.group(1))
        class_name = m.group(2)
        class_line_idx = text[:m.start()].count("\n")
        class_end_idx = _find_block_end_python(lines, class_line_idx, class_indent)

        # Find decorator start
        dec_start = class_line_idx
        decorators = _collect_decorators(lines, class_line_idx)
        if decorators:
            dec_start = class_line_idx - len(decorators)

        class_text = "\n".join(lines[dec_start : class_end_idx + 1])
        docstring = _extract_python_docstring(lines, class_line_idx + 1)
        comments = _collect_leading_comments(lines, dec_start)

        units.append(CodeUnit(
            kind="class",
            name=class_name,
            text=class_text,
            start_line=dec_start + 1,
            end_line=class_end_idx + 1,
            language=language,
            docstring=docstring,
            imports=file_imports,
            decorators=decorators,
            comments=comments,
        ))
        for i in range(dec_start, class_end_idx + 1):
            claimed.add(i)

        # Find methods inside this class
        class_body = "\n".join(lines[class_line_idx : class_end_idx + 1])
        for method_match in _PY_DEF.finditer(class_body):
            method_indent = len(method_match.group(1))
            if method_indent <= class_indent:
                continue  # not inside this class
            method_name = method_match.group(3)
            method_line_idx = class_line_idx + class_body[:method_match.start()].count("\n")
            method_end_idx = _find_block_end_python(lines, method_line_idx, method_indent)

            method_decorators = _collect_decorators(lines, method_line_idx)
            method_dec_start = method_line_idx - len(method_decorators) if method_decorators else method_line_idx

            method_text = "\n".join(lines[method_dec_start : method_end_idx + 1])
            method_docstring = _extract_python_docstring(lines, method_line_idx + 1)

            units.append(CodeUnit(
                kind="method",
                name=method_name,
                text=method_text,
                start_line=method_dec_start + 1,
                end_line=method_end_idx + 1,
                language=language,
                class_name=class_name,
                docstring=method_docstring,
                imports=file_imports,
                decorators=method_decorators,
            ))

    # 2. Find standalone functions (not inside a class)
    for m in _PY_DEF.finditer(text):
        indent = len(m.group(1))
        if indent > 0:
            continue  # inside a class or nested — already handled
        func_name = m.group(3)
        func_line_idx = text[:m.start()].count("\n")

        if func_line_idx in claimed:
            continue

        func_end_idx = _find_block_end_python(lines, func_line_idx, indent)
        decorators = _collect_decorators(lines, func_line_idx)
        dec_start = func_line_idx - len(decorators) if decorators else func_line_idx

        func_text = "\n".join(lines[dec_start : func_end_idx + 1])
        docstring = _extract_python_docstring(lines, func_line_idx + 1)
        comments = _collect_leading_comments(lines, dec_start)

        units.append(CodeUnit(
            kind="function",
            name=func_name,
            text=func_text,
            start_line=dec_start + 1,
            end_line=func_end_idx + 1,
            language=language,
            docstring=docstring,
            imports=file_imports,
            decorators=decorators,
            comments=comments,
        ))
        for i in range(dec_start, func_end_idx + 1):
            claimed.add(i)

    # 3. Collect unclaimed module-level code
    module_lines = []
    module_start = None
    for i, line in enumerate(lines):
        if i not in claimed:
            if module_start is None:
                module_start = i
            module_lines.append(line)

    if module_lines:
        module_text = "\n".join(module_lines).strip()
        if len(module_text) >= 30:
            units.append(CodeUnit(
                kind="module",
                name="<module>",
                text=module_text,
                start_line=(module_start or 0) + 1,
                end_line=len(lines),
                language=language,
                imports=file_imports,
            ))

    return units


# ── JavaScript/TypeScript parser (handles JSX/TSX too) ───────────────────────

_JS_FUNC = re.compile(
    r"^([ \t]*)(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*[\(<]",
    re.MULTILINE,
)
_JS_ARROW = re.compile(
    r"^([ \t]*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>",
    re.MULTILINE,
)
_JS_CLASS = re.compile(
    r"^([ \t]*)(?:export\s+)?(?:default\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_JS_IMPORT = re.compile(
    r"^(?:import\s+.+|const\s+\w+\s*=\s*require\(.+\))",
    re.MULTILINE,
)


def _find_brace_block_end(lines: List[str], start_idx: int) -> int:
    """Find end of a brace-delimited block (JS/TS/Go/Java)."""
    depth = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    return i
    return len(lines) - 1


def _extract_jsdoc(lines: List[str], line_idx: int) -> str:
    """Extract JSDoc comment block above a definition."""
    parts = []
    i = line_idx - 1
    while i >= 0 and not lines[i].strip():
        i -= 1  # skip blank lines
    # Look for /** ... */ block
    if i >= 0 and lines[i].strip().endswith("*/"):
        end_i = i
        while i >= 0:
            parts.insert(0, lines[i].strip())
            if "/**" in lines[i]:
                return "\n".join(parts)
            i -= 1
    return ""


def parse_javascript(text: str, language: str = "javascript") -> List[CodeUnit]:
    """Parse JS/TS/JSX/TSX source into logical code units."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _JS_IMPORT.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    # Classes
    for m in _JS_CLASS.finditer(text):
        class_name = m.group(2)
        class_line_idx = text[:m.start()].count("\n")
        class_end_idx = _find_brace_block_end(lines, class_line_idx)

        class_text = "\n".join(lines[class_line_idx : class_end_idx + 1])
        docstring = _extract_jsdoc(lines, class_line_idx)

        units.append(CodeUnit(
            kind="class",
            name=class_name,
            text=class_text,
            start_line=class_line_idx + 1,
            end_line=class_end_idx + 1,
            language=language,
            docstring=docstring,
            imports=file_imports,
        ))
        for i in range(class_line_idx, class_end_idx + 1):
            claimed.add(i)

    # Named functions
    for m in _JS_FUNC.finditer(text):
        func_name = m.group(2)
        func_line_idx = text[:m.start()].count("\n")
        if func_line_idx in claimed:
            continue
        func_end_idx = _find_brace_block_end(lines, func_line_idx)

        func_text = "\n".join(lines[func_line_idx : func_end_idx + 1])
        docstring = _extract_jsdoc(lines, func_line_idx)

        units.append(CodeUnit(
            kind="function",
            name=func_name,
            text=func_text,
            start_line=func_line_idx + 1,
            end_line=func_end_idx + 1,
            language=language,
            docstring=docstring,
            imports=file_imports,
        ))
        for i in range(func_line_idx, func_end_idx + 1):
            claimed.add(i)

    # Arrow functions assigned to const/let/var
    for m in _JS_ARROW.finditer(text):
        func_name = m.group(2)
        func_line_idx = text[:m.start()].count("\n")
        if func_line_idx in claimed:
            continue
        func_end_idx = _find_brace_block_end(lines, func_line_idx)
        # Arrow without braces — single expression
        if func_end_idx == func_line_idx:
            # Check for semicolon at end or next line
            func_end_idx = min(func_line_idx + 1, len(lines) - 1)

        func_text = "\n".join(lines[func_line_idx : func_end_idx + 1])
        docstring = _extract_jsdoc(lines, func_line_idx)

        units.append(CodeUnit(
            kind="function",
            name=func_name,
            text=func_text,
            start_line=func_line_idx + 1,
            end_line=func_end_idx + 1,
            language=language,
            docstring=docstring,
            imports=file_imports,
        ))
        for i in range(func_line_idx, func_end_idx + 1):
            claimed.add(i)

    # Module-level unclaimed code
    module_lines = []
    module_start = None
    for i, line in enumerate(lines):
        if i not in claimed:
            if module_start is None:
                module_start = i
            module_lines.append(line)

    if module_lines:
        module_text = "\n".join(module_lines).strip()
        if len(module_text) >= 30:
            units.append(CodeUnit(
                kind="module",
                name="<module>",
                text=module_text,
                start_line=(module_start or 0) + 1,
                end_line=len(lines),
                language=language,
                imports=file_imports,
            ))

    return units


# ── Go parser ─────────────────────────────────────────────────────────────────

_GO_FUNC = re.compile(
    r"^func\s+(?:\(\s*\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
    re.MULTILINE,
)
_GO_TYPE = re.compile(
    r"^type\s+(\w+)\s+struct\s*\{",
    re.MULTILINE,
)
_GO_IMPORT = re.compile(
    r'^import\s+(?:\([\s\S]*?\)|\".+\")',
    re.MULTILINE,
)


def parse_go(text: str) -> List[CodeUnit]:
    """Parse Go source into logical code units."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _GO_IMPORT.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    # Structs
    for m in _GO_TYPE.finditer(text):
        name = m.group(1)
        line_idx = text[:m.start()].count("\n")
        end_idx = _find_brace_block_end(lines, line_idx)

        unit_text = "\n".join(lines[line_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="class",
            name=name,
            text=unit_text,
            start_line=line_idx + 1,
            end_line=end_idx + 1,
            language="go",
            imports=file_imports,
        ))
        for i in range(line_idx, end_idx + 1):
            claimed.add(i)

    # Functions
    for m in _GO_FUNC.finditer(text):
        func_name = m.group(1)
        line_idx = text[:m.start()].count("\n")
        if line_idx in claimed:
            continue
        end_idx = _find_brace_block_end(lines, line_idx)

        # Collect preceding // comment block
        comments = _collect_leading_comments_cstyle(lines, line_idx)

        func_text = "\n".join(lines[line_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="function",
            name=func_name,
            text=func_text,
            start_line=line_idx + 1,
            end_line=end_idx + 1,
            language="go",
            imports=file_imports,
            comments=comments,
        ))
        for i in range(line_idx, end_idx + 1):
            claimed.add(i)

    # Module-level
    module_lines = [lines[i] for i in range(len(lines)) if i not in claimed]
    if module_lines:
        mod_text = "\n".join(module_lines).strip()
        if len(mod_text) >= 30:
            units.append(CodeUnit(
                kind="module", name="<module>", text=mod_text,
                start_line=1, end_line=len(lines), language="go",
                imports=file_imports,
            ))

    return units


def _collect_leading_comments_cstyle(lines: List[str], line_idx: int) -> str:
    """Collect // comment lines immediately above a definition."""
    comments = []
    i = line_idx - 1
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("//"):
            comments.insert(0, stripped)
        elif not stripped:
            if i > 0 and lines[i - 1].strip().startswith("//"):
                i -= 1
                continue
            break
        else:
            break
        i -= 1
    return "\n".join(comments)


# ── Java parser ───────────────────────────────────────────────────────────────

_JAVA_CLASS = re.compile(
    r"^([ \t]*)(?:public|private|protected)?\s*(?:abstract\s+)?(?:static\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_JAVA_METHOD = re.compile(
    r"^([ \t]+)(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?\w[\w<>\[\], ]*\s+(\w+)\s*\(",
    re.MULTILINE,
)
_JAVA_IMPORT = re.compile(
    r"^import\s+[\w.*]+;",
    re.MULTILINE,
)


def parse_java(text: str) -> List[CodeUnit]:
    """Parse Java source into logical code units."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _JAVA_IMPORT.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    # Classes
    for m in _JAVA_CLASS.finditer(text):
        class_name = m.group(2)
        line_idx = text[:m.start()].count("\n")
        end_idx = _find_brace_block_end(lines, line_idx)

        class_text = "\n".join(lines[line_idx : end_idx + 1])
        docstring = _extract_jsdoc(lines, line_idx)  # Javadoc uses same /** */ syntax

        units.append(CodeUnit(
            kind="class",
            name=class_name,
            text=class_text,
            start_line=line_idx + 1,
            end_line=end_idx + 1,
            language="java",
            docstring=docstring,
            imports=file_imports,
        ))
        for i in range(line_idx, end_idx + 1):
            claimed.add(i)

    # Methods (indented, inside classes)
    for m in _JAVA_METHOD.finditer(text):
        method_name = m.group(2)
        line_idx = text[:m.start()].count("\n")
        if line_idx in claimed:
            # Already part of a class — extract as method
            end_idx = _find_brace_block_end(lines, line_idx)
            method_text = "\n".join(lines[line_idx : end_idx + 1])
            docstring = _extract_jsdoc(lines, line_idx)

            # Find enclosing class
            enclosing_class = ""
            for u in units:
                if u.kind == "class" and u.start_line <= line_idx + 1 <= u.end_line:
                    enclosing_class = u.name
                    break

            units.append(CodeUnit(
                kind="method",
                name=method_name,
                text=method_text,
                start_line=line_idx + 1,
                end_line=end_idx + 1,
                language="java",
                class_name=enclosing_class,
                docstring=docstring,
                imports=file_imports,
            ))

    # Module-level
    module_lines = [lines[i] for i in range(len(lines)) if i not in claimed]
    if module_lines:
        mod_text = "\n".join(module_lines).strip()
        if len(mod_text) >= 30:
            units.append(CodeUnit(
                kind="module", name="<module>", text=mod_text,
                start_line=1, end_line=len(lines), language="java",
                imports=file_imports,
            ))

    return units


# ── JSON parser ───────────────────────────────────────────────────────────────

def parse_json(text: str) -> List[CodeUnit]:
    """Parse JSON into top-level key blocks."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [CodeUnit(
            kind="config_block", name="<invalid_json>", text=text,
            start_line=1, end_line=text.count("\n") + 1, language="json",
        )]

    if not isinstance(data, dict):
        return [CodeUnit(
            kind="config_block", name="<root>", text=text,
            start_line=1, end_line=text.count("\n") + 1, language="json",
        )]

    units: List[CodeUnit] = []
    lines = text.splitlines()

    for key in data:
        # Find the line where this key appears
        key_pattern = re.compile(rf'^\s*"{re.escape(key)}"\s*:', re.MULTILINE)
        m = key_pattern.search(text)
        if not m:
            continue
        key_line = text[:m.start()].count("\n")
        value = data[key]
        value_text = json.dumps({key: value}, indent=2)

        units.append(CodeUnit(
            kind="config_block",
            name=key,
            text=value_text,
            start_line=key_line + 1,
            end_line=key_line + value_text.count("\n") + 1,
            language="json",
        ))

    # If no keys found or too few, emit the whole file
    if not units:
        units.append(CodeUnit(
            kind="config_block", name="<root>", text=text,
            start_line=1, end_line=len(lines), language="json",
        ))

    return units


# ── Config file parser (YAML, TOML, INI) ─────────────────────────────────────

_YAML_SECTION = re.compile(r"^(\S[\w\-. ]*)\s*:", re.MULTILINE)
_INI_SECTION = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)
_TOML_TABLE = re.compile(r"^\[([^\]]+)\]", re.MULTILINE)


def parse_config(text: str, language: str = "config") -> List[CodeUnit]:
    """Parse config files into section blocks."""
    lines = text.splitlines()
    if not lines:
        return []

    units: List[CodeUnit] = []

    if language == "toml":
        sections = list(_TOML_TABLE.finditer(text))
    elif language == "config":  # INI-style
        sections = list(_INI_SECTION.finditer(text))
    else:  # YAML
        sections = list(_YAML_SECTION.finditer(text))

    if not sections:
        # Emit whole file as one block
        return [CodeUnit(
            kind="config_block", name="<root>", text=text,
            start_line=1, end_line=len(lines), language=language,
        )]

    for idx, m in enumerate(sections):
        name = m.group(1)
        start_line_idx = text[:m.start()].count("\n")
        if idx + 1 < len(sections):
            end_line_idx = text[:sections[idx + 1].start()].count("\n") - 1
        else:
            end_line_idx = len(lines) - 1

        section_text = "\n".join(lines[start_line_idx : end_line_idx + 1])
        units.append(CodeUnit(
            kind="config_block",
            name=name,
            text=section_text,
            start_line=start_line_idx + 1,
            end_line=end_line_idx + 1,
            language=language,
        ))

    return units


# ── C-family parser (C, C++, C#, Rust, Kotlin, Swift, Scala) ──────────────────
#
# Covers languages that use braces for blocks and have function/struct/class
# declarations recognisable by a simple regex + brace-counting approach.
# This is intentionally broad  — it extracts *structural boundaries*, not a
# full AST, which is good enough for chunking + retrieval.

_C_FUNC = re.compile(
    r"^[\w\s\*:&<>,\[\]~]+?\b(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?\{",
    re.MULTILINE,
)
_C_CLASS = re.compile(
    r"^(?:(?:public|private|protected|abstract|sealed|partial|static|final)\s+)*"
    r"(?:class|struct|interface|enum|trait|object)\s+(\w+)",
    re.MULTILINE,
)
_C_INCLUDE = re.compile(
    r"^(?:#include\s+[<\"].+?[>\"]|using\s+.+;|import\s+.+)",
    re.MULTILINE,
)
_RUST_FN = re.compile(
    r"^(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(\w+)",
    re.MULTILINE,
)
_RUST_STRUCT = re.compile(
    r"^(?:pub(?:\([\w:]+\))?\s+)?(?:struct|enum|trait|impl)\s+(\w+)",
    re.MULTILINE,
)


def parse_c_family(text: str, language: str = "c") -> List[CodeUnit]:
    """Parse C/C++/C#/Kotlin/Swift/Scala into logical units using brace matching."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _C_INCLUDE.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    # Classes / structs / interfaces
    for m in _C_CLASS.finditer(text):
        name = m.group(1)
        start_idx = text[:m.start()].count("\n")
        # Only match if there's an opening brace on or near this line
        end_idx = _find_brace_block_end(lines, start_idx)
        if end_idx <= start_idx:
            continue
        block_text = "\n".join(lines[start_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="class", name=name, text=block_text,
            start_line=start_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(start_idx, end_idx + 1):
            claimed.add(i)

    # Free functions (not inside classes already claimed)
    func_re = _RUST_FN if language == "rust" else _C_FUNC
    for m in func_re.finditer(text):
        func_name = m.group(1)
        func_idx = text[:m.start()].count("\n")
        if func_idx in claimed:
            continue
        end_idx = _find_brace_block_end(lines, func_idx)
        if end_idx < func_idx:
            continue
        block_text = "\n".join(lines[func_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="function", name=func_name, text=block_text,
            start_line=func_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(func_idx, end_idx + 1):
            claimed.add(i)

    # Rust structs/enums/traits/impls
    if language == "rust":
        for m in _RUST_STRUCT.finditer(text):
            name = m.group(1)
            start_idx = text[:m.start()].count("\n")
            if start_idx in claimed:
                continue
            end_idx = _find_brace_block_end(lines, start_idx)
            if end_idx <= start_idx:
                # Could be a unit struct without braces
                end_idx = start_idx
            block_text = "\n".join(lines[start_idx : end_idx + 1])
            units.append(CodeUnit(
                kind="class", name=name, text=block_text,
                start_line=start_idx + 1, end_line=end_idx + 1,
                language=language, imports=file_imports,
            ))
            for i in range(start_idx, end_idx + 1):
                claimed.add(i)

    # Module-level unclaimed code
    unclaimed_lines = [i for i in range(len(lines)) if i not in claimed and lines[i].strip()]
    if unclaimed_lines:
        module_text = "\n".join(lines)
        units.append(CodeUnit(
            kind="module", name="<module>", text=module_text,
            start_line=1, end_line=len(lines),
            language=language, imports=file_imports,
        ))

    return units


# ── Ruby parser ───────────────────────────────────────────────────────────────

_RUBY_CLASS = re.compile(r"^[ \t]*(?:class|module)\s+(\w+)", re.MULTILINE)
_RUBY_DEF = re.compile(r"^[ \t]*def\s+(\w+[\?\!]?)", re.MULTILINE)
_RUBY_IMPORT = re.compile(r"^(?:require|require_relative|include|extend)\s+.+", re.MULTILINE)


def parse_ruby(text: str, language: str = "ruby") -> List[CodeUnit]:
    """Parse Ruby source into classes/modules and methods."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _RUBY_IMPORT.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    # Classes / modules — use indentation-based end detection
    for m in _RUBY_CLASS.finditer(text):
        name = m.group(1)
        start_idx = text[:m.start()].count("\n")
        indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        # Find matching 'end'
        end_idx = _find_ruby_end(lines, start_idx, indent)
        block_text = "\n".join(lines[start_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="class", name=name, text=block_text,
            start_line=start_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(start_idx, end_idx + 1):
            claimed.add(i)

    # Methods outside classes
    for m in _RUBY_DEF.finditer(text):
        func_name = m.group(1)
        func_idx = text[:m.start()].count("\n")
        if func_idx in claimed:
            continue
        indent = len(lines[func_idx]) - len(lines[func_idx].lstrip())
        end_idx = _find_ruby_end(lines, func_idx, indent)
        block_text = "\n".join(lines[func_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="function", name=func_name, text=block_text,
            start_line=func_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(func_idx, end_idx + 1):
            claimed.add(i)

    # Module-level
    unclaimed = [i for i in range(len(lines)) if i not in claimed and lines[i].strip()]
    if unclaimed:
        units.append(CodeUnit(
            kind="module", name="<module>", text="\n".join(lines),
            start_line=1, end_line=len(lines),
            language=language, imports=file_imports,
        ))

    return units


def _find_ruby_end(lines: List[str], start_idx: int, base_indent: int) -> int:
    """Find the matching 'end' for a Ruby class/def/module block."""
    depth = 1
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Nested blocks increase depth
        if stripped.startswith(("def ", "class ", "module ", "do", "if ", "unless ", "begin", "case ")):
            depth += 1
        elif stripped == "end":
            depth -= 1
            if depth == 0:
                return i
    return len(lines) - 1


# ── PHP parser ────────────────────────────────────────────────────────────────

_PHP_CLASS = re.compile(
    r"^(?:(?:abstract|final)\s+)?class\s+(\w+)", re.MULTILINE
)
_PHP_FUNC = re.compile(
    r"^(?:[ \t]*(?:public|private|protected|static)\s+)*function\s+(\w+)\s*\(",
    re.MULTILINE,
)
_PHP_IMPORT = re.compile(
    r"^(?:use\s+.+;|require(?:_once)?\s+.+;|include(?:_once)?\s+.+;)",
    re.MULTILINE,
)


def parse_php(text: str, language: str = "php") -> List[CodeUnit]:
    """Parse PHP source into classes and functions."""
    lines = text.splitlines()
    if not lines:
        return []

    file_imports = [m.group().strip() for m in _PHP_IMPORT.finditer(text)]
    units: List[CodeUnit] = []
    claimed: set[int] = set()

    for m in _PHP_CLASS.finditer(text):
        name = m.group(1)
        start_idx = text[:m.start()].count("\n")
        end_idx = _find_brace_block_end(lines, start_idx)
        block_text = "\n".join(lines[start_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="class", name=name, text=block_text,
            start_line=start_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(start_idx, end_idx + 1):
            claimed.add(i)

    for m in _PHP_FUNC.finditer(text):
        func_name = m.group(1)
        func_idx = text[:m.start()].count("\n")
        if func_idx in claimed:
            continue
        end_idx = _find_brace_block_end(lines, func_idx)
        block_text = "\n".join(lines[func_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="function", name=func_name, text=block_text,
            start_line=func_idx + 1, end_line=end_idx + 1,
            language=language, imports=file_imports,
        ))
        for i in range(func_idx, end_idx + 1):
            claimed.add(i)

    unclaimed = [i for i in range(len(lines)) if i not in claimed and lines[i].strip()]
    if unclaimed:
        units.append(CodeUnit(
            kind="module", name="<module>", text="\n".join(lines),
            start_line=1, end_line=len(lines),
            language=language, imports=file_imports,
        ))

    return units


# ── Shell script parser ───────────────────────────────────────────────────────

_SH_FUNC = re.compile(
    r"^(?:function\s+)?(\w+)\s*\(\s*\)\s*\{",
    re.MULTILINE,
)


def parse_shell(text: str, language: str = "shell") -> List[CodeUnit]:
    """Parse shell scripts into function blocks."""
    lines = text.splitlines()
    if not lines:
        return []

    units: List[CodeUnit] = []
    claimed: set[int] = set()

    for m in _SH_FUNC.finditer(text):
        func_name = m.group(1)
        func_idx = text[:m.start()].count("\n")
        end_idx = _find_brace_block_end(lines, func_idx)
        block_text = "\n".join(lines[func_idx : end_idx + 1])
        units.append(CodeUnit(
            kind="function", name=func_name, text=block_text,
            start_line=func_idx + 1, end_line=end_idx + 1,
            language=language,
        ))
        for i in range(func_idx, end_idx + 1):
            claimed.add(i)

    unclaimed = [i for i in range(len(lines)) if i not in claimed and lines[i].strip()]
    if unclaimed:
        units.append(CodeUnit(
            kind="module", name="<module>", text="\n".join(lines),
            start_line=1, end_line=len(lines),
            language=language,
        ))

    return units


# ── Dispatch: parse any file ──────────────────────────────────────────────────

_PARSERS = {
    "python":      parse_python,
    "javascript":  parse_javascript,
    "typescript":  lambda t: parse_javascript(t, "typescript"),
    "jsx":         lambda t: parse_javascript(t, "jsx"),
    "tsx":         lambda t: parse_javascript(t, "tsx"),
    "go":          parse_go,
    "java":        parse_java,
    "json":        parse_json,
    "yaml":        lambda t: parse_config(t, "yaml"),
    "toml":        lambda t: parse_config(t, "toml"),
    "config":      lambda t: parse_config(t, "config"),
    # New parsers
    "c":           lambda t: parse_c_family(t, "c"),
    "cpp":         lambda t: parse_c_family(t, "cpp"),
    "csharp":      lambda t: parse_c_family(t, "csharp"),
    "rust":        lambda t: parse_c_family(t, "rust"),
    "kotlin":      lambda t: parse_c_family(t, "kotlin"),
    "swift":       lambda t: parse_c_family(t, "swift"),
    "scala":       lambda t: parse_c_family(t, "scala"),
    "ruby":        parse_ruby,
    "php":         parse_php,
    "shell":       parse_shell,
}


def parse_file(text: str, language: str) -> List[CodeUnit]:
    """
    Parse source text into logical code units using the appropriate
    language-specific parser.

    Falls back to a single "module" unit for unsupported languages.
    """
    parser = _PARSERS.get(language)
    if parser:
        units = parser(text)
        if units:
            return units

    # Fallback: treat entire file as one module unit
    line_count = text.count("\n") + 1
    if len(text.strip()) < 30:
        return []
    return [CodeUnit(
        kind="module",
        name="<module>",
        text=text,
        start_line=1,
        end_line=line_count,
        language=language,
    )]
