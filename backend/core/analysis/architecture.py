"""
core/analysis/architecture.py
─────────────────────────────────────────────────────────────────────────────
Project architecture summarizer.

After indexing, generates a high-level architecture overview by analysing:
  - Directory structure → detect services, modules, layers
  - Entry points → main files, server files, route definitions
  - Core dependencies → most-imported modules
  - Languages used → breakdown by file count
  - Key classes and functions → most-connected symbols

The summary is stored per-namespace and can be used by the LLM
to answer "Explain the architecture of this project."
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("bakup.architecture")


# ── Patterns that indicate entry points ───────────────────────────────────────

_ENTRY_POINT_PATTERNS = [
    re.compile(r"^main\.(py|js|ts|go|java)$", re.IGNORECASE),
    re.compile(r"^app\.(py|js|ts)$", re.IGNORECASE),
    re.compile(r"^server\.(py|js|ts|go)$", re.IGNORECASE),
    re.compile(r"^index\.(js|ts|tsx)$", re.IGNORECASE),
    re.compile(r"^manage\.py$", re.IGNORECASE),
    re.compile(r"^wsgi\.py$", re.IGNORECASE),
    re.compile(r"^asgi\.py$", re.IGNORECASE),
    re.compile(r"^cmd[/\\]", re.IGNORECASE),
]

# Patterns that indicate config/infra files
_CONFIG_PATTERNS = [
    re.compile(r"^(docker-compose|Dockerfile|\.dockerignore)", re.IGNORECASE),
    re.compile(r"^(Makefile|CMakeLists\.txt|build\.gradle)$", re.IGNORECASE),
    re.compile(r"^(package\.json|requirements\.txt|go\.mod|pom\.xml|Cargo\.toml)$", re.IGNORECASE),
    re.compile(r"\.(yaml|yml|toml|ini|cfg|conf|env)$", re.IGNORECASE),
]

# Common directory names that indicate project modules/services
_MODULE_DIR_NAMES = {
    "api", "routes", "handlers", "controllers", "views",
    "models", "schemas", "entities",
    "services", "core", "lib", "utils", "helpers", "common",
    "middleware", "auth", "authentication", "authorization",
    "db", "database", "storage", "cache", "queue",
    "config", "settings", "constants",
    "tests", "test", "spec", "__tests__",
    "cmd", "pkg", "internal",
    "components", "pages", "hooks", "context",
}


@dataclass
class ModuleInfo:
    """Information about a detected module/service."""
    name: str
    path: str
    file_count: int = 0
    languages: Set[str] = field(default_factory=set)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ArchitectureSummary:
    """High-level project architecture overview."""
    project_name: str = ""
    languages: Dict[str, int] = field(default_factory=dict)         # lang → file count
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    entry_points: List[str] = field(default_factory=list)
    modules: List[ModuleInfo] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    core_dependencies: List[str] = field(default_factory=list)      # most-imported modules
    directory_tree: str = ""                                         # simplified ASCII tree

    def summary_text(self) -> str:
        """Generate a human-readable architecture summary."""
        parts: List[str] = []

        parts.append(f"## Project: {self.project_name}")
        parts.append("")

        # Languages
        if self.languages:
            lang_lines = [f"  - {lang}: {count} file(s)" for lang, count in
                          sorted(self.languages.items(), key=lambda x: -x[1])]
            parts.append("### Languages")
            parts.extend(lang_lines)
            parts.append("")

        # Overview stats
        parts.append(f"### Overview")
        parts.append(f"  - Total files: {self.total_files}")
        parts.append(f"  - Total functions: {self.total_functions}")
        parts.append(f"  - Total classes: {self.total_classes}")
        parts.append("")

        # Entry points
        if self.entry_points:
            parts.append("### Entry Points")
            for ep in self.entry_points:
                parts.append(f"  - {ep}")
            parts.append("")

        # Modules
        if self.modules:
            parts.append("### Modules / Services")
            for mod in self.modules:
                desc = f" — {mod.description}" if mod.description else ""
                parts.append(f"  - **{mod.name}/** ({mod.file_count} files, "
                             f"{', '.join(sorted(mod.languages)) or 'mixed'}){desc}")
                if mod.classes:
                    parts.append(f"    Classes: {', '.join(mod.classes[:8])}")
                if mod.functions:
                    parts.append(f"    Functions: {', '.join(mod.functions[:8])}")
            parts.append("")

        # Core dependencies
        if self.core_dependencies:
            parts.append("### Core Dependencies (most imported)")
            for dep in self.core_dependencies[:10]:
                parts.append(f"  - {dep}")
            parts.append("")

        # Config files
        if self.config_files:
            parts.append("### Configuration Files")
            for cf in self.config_files[:15]:
                parts.append(f"  - {cf}")
            parts.append("")

        # Directory tree
        if self.directory_tree:
            parts.append("### Directory Structure")
            parts.append("```")
            parts.append(self.directory_tree)
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> Dict:
        """Serialise to JSON-safe dict."""
        return {
            "project_name": self.project_name,
            "languages": self.languages,
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "entry_points": self.entry_points,
            "modules": [
                {"name": m.name, "path": m.path, "file_count": m.file_count,
                 "languages": sorted(m.languages), "classes": m.classes[:10],
                 "functions": m.functions[:10]}
                for m in self.modules
            ],
            "config_files": self.config_files,
            "core_dependencies": self.core_dependencies,
        }


# ── Singleton storage ─────────────────────────────────────────────────────────

_summaries: Dict[str, ArchitectureSummary] = {}


def get_architecture(namespace: str) -> Optional[ArchitectureSummary]:
    """Get the cached architecture summary for a namespace."""
    return _summaries.get(namespace)


def clear_architecture(namespace: str) -> None:
    """Remove cached architecture summary."""
    _summaries.pop(namespace, None)


# ── Builder ───────────────────────────────────────────────────────────────────

def build_architecture_summary(
    file_paths: List[str],
    units_by_file: Dict[str, list],
    import_counter: Counter,
    project_name: str = "",
    namespace: str = "",
) -> ArchitectureSummary:
    """
    Build an architecture summary from indexed data.

    Args:
        file_paths:      List of relative file paths indexed
        units_by_file:   Dict mapping file_path → List[CodeUnit]
        import_counter:  Counter of imported module names across all files
        project_name:    Human-readable project name
        namespace:       Namespace to cache the summary under

    Returns:
        ArchitectureSummary with detected structure
    """
    summary = ArchitectureSummary(project_name=project_name or "Unknown Project")
    summary.total_files = len(file_paths)

    # ── Language breakdown ────────────────────────────────────────────────
    lang_counter: Counter = Counter()
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        lang = _ext_to_language(ext)
        if lang:
            lang_counter[lang] += 1
    summary.languages = dict(lang_counter.most_common())

    # ── Entry points ─────────────────────────────────────────────────────
    for fp in file_paths:
        name = Path(fp).name
        for pat in _ENTRY_POINT_PATTERNS:
            if pat.search(name) or pat.search(fp):
                summary.entry_points.append(fp)
                break
    summary.entry_points = sorted(set(summary.entry_points))

    # ── Config files ─────────────────────────────────────────────────────
    for fp in file_paths:
        name = Path(fp).name
        for pat in _CONFIG_PATTERNS:
            if pat.search(name):
                summary.config_files.append(fp)
                break
    summary.config_files = sorted(set(summary.config_files))

    # ── Functions and classes ────────────────────────────────────────────
    all_functions: List[str] = []
    all_classes: List[str] = []

    for file_path, units in units_by_file.items():
        for unit in units:
            if unit.kind == "function":
                all_functions.append(unit.name)
            elif unit.kind == "class":
                all_classes.append(unit.name)
            elif unit.kind == "method":
                all_functions.append(unit.name)

    summary.total_functions = len(all_functions)
    summary.total_classes = len(all_classes)

    # ── Module detection ─────────────────────────────────────────────────
    dir_files: Dict[str, List[str]] = defaultdict(list)
    for fp in file_paths:
        parts = Path(fp).parts
        if len(parts) >= 2:
            top_dir = parts[0]
            dir_files[top_dir].append(fp)

    for dir_name, files_in_dir in sorted(dir_files.items(), key=lambda x: -len(x[1])):
        if dir_name.startswith("."):
            continue

        # Collect metadata for this module
        mod = ModuleInfo(
            name=dir_name,
            path=dir_name,
            file_count=len(files_in_dir),
        )

        for fp in files_in_dir:
            ext = Path(fp).suffix.lower()
            lang = _ext_to_language(ext)
            if lang:
                mod.languages.add(lang)
            # Collect symbols from this module
            if fp in units_by_file:
                for unit in units_by_file[fp]:
                    if unit.kind == "class":
                        mod.classes.append(unit.name)
                    elif unit.kind == "function":
                        mod.functions.append(unit.name)

        # Detect module purpose from directory name
        dir_lower = dir_name.lower()
        if dir_lower in ("api", "routes", "handlers", "controllers"):
            mod.description = "API / route handlers"
        elif dir_lower in ("models", "schemas", "entities"):
            mod.description = "Data models / schemas"
        elif dir_lower in ("services", "core", "lib"):
            mod.description = "Core business logic"
        elif dir_lower in ("tests", "test", "spec", "__tests__"):
            mod.description = "Test suite"
        elif dir_lower in ("utils", "helpers", "common"):
            mod.description = "Shared utilities"
        elif dir_lower in ("db", "database", "storage"):
            mod.description = "Data storage layer"
        elif dir_lower in ("auth", "authentication"):
            mod.description = "Authentication / authorization"
        elif dir_lower in ("config", "settings"):
            mod.description = "Configuration"
        elif dir_lower in ("middleware"):
            mod.description = "Request middleware"
        elif dir_lower in ("components", "pages", "views"):
            mod.description = "UI components / views"

        summary.modules.append(mod)

    # Limit modules shown
    summary.modules = summary.modules[:15]

    # ── Core dependencies ────────────────────────────────────────────────
    summary.core_dependencies = [
        name for name, _ in import_counter.most_common(15)
    ]

    # ── Directory tree ───────────────────────────────────────────────────
    summary.directory_tree = _build_dir_tree(file_paths, max_depth=3)

    # ── Cache ────────────────────────────────────────────────────────────
    if namespace:
        _summaries[namespace] = summary

    logger.info(
        "Architecture summary: %s — %d files, %d functions, %d classes, "
        "%d modules, %d entry points",
        summary.project_name,
        summary.total_files, summary.total_functions, summary.total_classes,
        len(summary.modules), len(summary.entry_points),
    )

    return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext_to_language(ext: str) -> str:
    """Map file extension to language name."""
    mapping = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "JSX", ".tsx": "TSX", ".go": "Go", ".java": "Java",
        ".rs": "Rust", ".rb": "Ruby", ".php": "PHP",
        ".c": "C", ".cpp": "C++", ".h": "C/C++ Header",
        ".cs": "C#", ".swift": "Swift", ".kt": "Kotlin",
        ".scala": "Scala", ".sh": "Shell", ".sql": "SQL",
        ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
        ".toml": "TOML", ".xml": "XML", ".html": "HTML",
        ".css": "CSS", ".scss": "SCSS", ".md": "Markdown",
    }
    return mapping.get(ext, "")


def _build_dir_tree(file_paths: List[str], max_depth: int = 3) -> str:
    """Build a simplified ASCII directory tree."""
    dirs: Set[str] = set()
    for fp in file_paths:
        parts = Path(fp).parts
        for i in range(1, min(len(parts), max_depth + 1)):
            dirs.add("/".join(parts[:i]))

    if not dirs:
        return ""

    sorted_dirs = sorted(dirs)
    lines: List[str] = []
    for d in sorted_dirs[:40]:  # Limit output
        depth = d.count("/")
        prefix = "  " * depth
        name = d.split("/")[-1]
        # Check if it's a file (has extension) or directory
        if "." in name and any(fp == d or fp.startswith(d + "/") for fp in file_paths):
            lines.append(f"{prefix}{name}")
        else:
            lines.append(f"{prefix}{name}/")

    return "\n".join(lines)
