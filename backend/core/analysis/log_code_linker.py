"""
core/analysis/log_code_linker.py
─────────────────────────────────────────────────────────────────────────────
Log-to-code cross-analysis engine.

When log errors contain file paths, class names, function names, or stack
traces, this module links them to indexed code chunks. This enables
root-cause analysis by correlating runtime errors with source code.

Capabilities:
  - Extract file references from log text
  - Extract class/function names from stack traces
  - Match extracted references to indexed code chunks
  - Build combined log + code context for LLM reasoning

Example:
  User asks: "Why is payment failing?"
  1. Retrieve relevant log errors mentioning payment
  2. Extract class/function references from those logs
  3. Fetch the corresponding code chunks
  4. Bundle log + code context for the LLM
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.retrieval.ranker import RankedResult

logger = logging.getLogger("bakup.log_code_linker")


# ── Reference extraction patterns ────────────────────────────────────────────

# Python traceback: File "path/to/file.py", line 42, in function_name
_PY_TRACEBACK = re.compile(
    r'File\s+"([^"]+\.py)",\s+line\s+(\d+),\s+in\s+(\w+)',
)

# Generic file:line references:  session.py:42  or  src/auth/session.py:42
_FILE_LINE = re.compile(
    r'([\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php))\s*[:\s]+(?:line\s+)?(\d+)',
    re.IGNORECASE,
)

# Java stack trace: at com.example.Service.method(Service.java:42)
_JAVA_STACK = re.compile(
    r'at\s+([\w.]+)\.([\w]+)\(([\w.]+):(\d+)\)',
)

# JavaScript/Node stack trace: at functionName (path/file.js:42:10)
_JS_STACK = re.compile(
    r'at\s+(\w+)\s+\(([^)]+\.(?:js|ts|tsx|jsx)):(\d+):\d+\)',
)

# Go panic: goroutine N [running]: package.function(args) \n path/file.go:42
_GO_STACK = re.compile(
    r'([\w./\\-]+\.go):(\d+)',
)

# Generic class.method pattern:  SessionStore.create_session
_CLASS_METHOD = re.compile(
    r'\b([A-Z][a-zA-Z0-9]+)\.([\w]+)\b',
)

# Function name in error context: "in function_name" or "function 'name'"
_FUNC_REF = re.compile(
    r"(?:in|function|method|def)\s+['\"]?(\w{2,})['\"]?",
    re.IGNORECASE,
)

# Exception class names:  NullPointerException, ConnectionRefusedError
_EXCEPTION_CLASS = re.compile(
    r'\b([A-Z][a-zA-Z]+(?:Error|Exception|Fault|Failure|Panic))\b',
)


@dataclass
class CodeReference:
    """A reference to source code extracted from a log entry."""
    file_path: str = ""
    line_number: int = 0
    function_name: str = ""
    class_name: str = ""
    confidence: float = 0.0      # How likely this is a real reference
    source_text: str = ""        # The log text it was extracted from


@dataclass
class LogCodeLink:
    """A link between a log error and related code."""
    log_chunk: RankedResult
    code_chunks: List[RankedResult]
    references: List[CodeReference]


def extract_code_references(text: str) -> List[CodeReference]:
    """
    Extract code references (file paths, function names, class names)
    from a log entry or error message.
    """
    refs: List[CodeReference] = []
    seen: Set[str] = set()

    # Python tracebacks — highest confidence
    for m in _PY_TRACEBACK.finditer(text):
        key = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
        if key not in seen:
            seen.add(key)
            refs.append(CodeReference(
                file_path=m.group(1),
                line_number=int(m.group(2)),
                function_name=m.group(3),
                confidence=0.95,
                source_text=m.group(0),
            ))

    # Java stack traces
    for m in _JAVA_STACK.finditer(text):
        key = f"{m.group(3)}:{m.group(4)}:{m.group(2)}"
        if key not in seen:
            seen.add(key)
            full_class = m.group(1)
            class_name = full_class.split(".")[-1] if "." in full_class else full_class
            refs.append(CodeReference(
                file_path=m.group(3),
                line_number=int(m.group(4)),
                function_name=m.group(2),
                class_name=class_name,
                confidence=0.90,
                source_text=m.group(0),
            ))

    # JS/Node stack traces
    for m in _JS_STACK.finditer(text):
        key = f"{m.group(2)}:{m.group(3)}:{m.group(1)}"
        if key not in seen:
            seen.add(key)
            refs.append(CodeReference(
                file_path=m.group(2),
                line_number=int(m.group(3)),
                function_name=m.group(1),
                confidence=0.90,
                source_text=m.group(0),
            ))

    # File:line references (generic)
    for m in _FILE_LINE.finditer(text):
        key = f"{m.group(1)}:{m.group(2)}"
        if key not in seen:
            seen.add(key)
            refs.append(CodeReference(
                file_path=m.group(1),
                line_number=int(m.group(2)),
                confidence=0.80,
                source_text=m.group(0),
            ))

    # Class.method patterns
    for m in _CLASS_METHOD.finditer(text):
        key = f"cm:{m.group(1)}.{m.group(2)}"
        if key not in seen:
            seen.add(key)
            # Skip common false positives
            if m.group(1) not in ("System", "Console", "Object", "Array", "String",
                                   "Integer", "Boolean", "Math", "Date"):
                refs.append(CodeReference(
                    class_name=m.group(1),
                    function_name=m.group(2),
                    confidence=0.70,
                    source_text=m.group(0),
                ))

    # Function name references
    for m in _FUNC_REF.finditer(text):
        key = f"func:{m.group(1)}"
        if key not in seen:
            seen.add(key)
            refs.append(CodeReference(
                function_name=m.group(1),
                confidence=0.50,
                source_text=m.group(0),
            ))

    return refs


def link_logs_to_code(
    log_chunks: List[RankedResult],
    code_chunks: List[RankedResult],
) -> List[LogCodeLink]:
    """
    For each log chunk, find related code chunks by matching extracted
    file paths, function names, and class names.

    Returns a list of LogCodeLink objects, one per log chunk that has
    code references. Log chunks with no code matches are excluded.
    """
    if not log_chunks or not code_chunks:
        return []

    links: List[LogCodeLink] = []

    # Build lookup indices for code chunks
    code_by_file: Dict[str, List[RankedResult]] = {}
    code_by_func: Dict[str, List[RankedResult]] = {}
    code_by_class: Dict[str, List[RankedResult]] = {}

    for c in code_chunks:
        # Index by file path (basename and full path)
        from pathlib import Path as P
        basename = P(c.source_file).name
        code_by_file.setdefault(basename, []).append(c)
        code_by_file.setdefault(c.source_file, []).append(c)

        # Index by function name
        if c.function_name:
            code_by_func.setdefault(c.function_name, []).append(c)

        # Index by class name
        if c.class_name:
            code_by_class.setdefault(c.class_name, []).append(c)

    for log in log_chunks:
        if log.source_type != "log":
            continue

        refs = extract_code_references(log.text)
        if not refs:
            continue

        matched_code: List[RankedResult] = []
        matched_keys: Set[str] = set()

        for ref in refs:
            # Match by file path
            if ref.file_path:
                from pathlib import Path as P
                basename = P(ref.file_path).name
                for c in code_by_file.get(basename, []):
                    key = f"{c.source_file}:{c.line_start}"
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matched_code.append(c)
                for c in code_by_file.get(ref.file_path, []):
                    key = f"{c.source_file}:{c.line_start}"
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matched_code.append(c)

            # Match by function name
            if ref.function_name:
                for c in code_by_func.get(ref.function_name, []):
                    key = f"{c.source_file}:{c.line_start}"
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matched_code.append(c)

            # Match by class name
            if ref.class_name:
                for c in code_by_class.get(ref.class_name, []):
                    key = f"{c.source_file}:{c.line_start}"
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matched_code.append(c)

        if matched_code:
            links.append(LogCodeLink(
                log_chunk=log,
                code_chunks=matched_code[:5],   # Limit per log entry
                references=refs,
            ))

    logger.debug(
        "Log-code linking: %d log chunk(s), %d code chunk(s) → %d link(s)",
        len(log_chunks), len(code_chunks), len(links),
    )

    return links


def build_cross_analysis_context(
    links: List[LogCodeLink],
    max_chars: int = 800,
) -> str:
    """
    Build a combined log + code context block for the LLM.

    Groups each log error with its related code so the LLM can reason
    about root causes.
    """
    if not links:
        return ""

    parts: List[str] = []
    idx = 0

    for link in links[:5]:
        idx += 1
        # Log entry
        log = link.log_chunk
        log_text = log.text[:max_chars]
        parts.append(
            f"## Error #{idx}\n"
            f"**Log entry** ({log.source_file}, lines {log.line_start}–{log.line_end}):\n"
            f"```\n{log_text}\n```\n"
        )

        # Code references
        if link.references:
            ref_lines = []
            for ref in link.references:
                ref_parts = []
                if ref.file_path:
                    ref_parts.append(f"file: {ref.file_path}")
                if ref.line_number:
                    ref_parts.append(f"line: {ref.line_number}")
                if ref.function_name:
                    ref_parts.append(f"function: {ref.function_name}")
                if ref.class_name:
                    ref_parts.append(f"class: {ref.class_name}")
                if ref_parts:
                    ref_lines.append("  - " + ", ".join(ref_parts))
            if ref_lines:
                parts.append("**Extracted references:**\n" + "\n".join(ref_lines) + "\n")

        # Related code
        if link.code_chunks:
            parts.append("**Related source code:**\n")
            for c in link.code_chunks:
                code_text = c.text[:max_chars]
                label = c.source_file
                if c.function_name:
                    label += f" → {c.function_name}()"
                if c.class_name:
                    label += f" [class: {c.class_name}]"
                parts.append(
                    f"*{label}* (lines {c.line_start}–{c.line_end}):\n"
                    f"```\n{code_text}\n```\n"
                )

    return "\n".join(parts)
