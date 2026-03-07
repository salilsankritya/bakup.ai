"""
core/ingestion/symbol_graph.py
─────────────────────────────────────────────────────────────────────────────
Lightweight in-memory symbol graph for project intelligence.

Tracks relationships between code symbols:
  - function → calls (which functions it calls)
  - class → methods (its own methods)
  - module → imports (what it depends on)
  - file → defines (functions/classes defined in it)
  - file → imports (what files/modules it imports)

The graph is built during ingestion by scanning parsed CodeUnits.
No external dependencies — uses stdlib collections only.

Used to answer structural questions like:
  "Which files use the payment service?"
  "What depends on the auth module?"
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.ingestion.code_parser import CodeUnit

logger = logging.getLogger("bakup.symbol_graph")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SymbolNode:
    """A node in the symbol graph representing a code entity."""
    name: str
    kind: str                         # "function" | "class" | "method" | "module" | "file"
    file_path: str                    # Relative path to source file
    line_start: int = 0
    line_end: int = 0
    language: str = ""
    docstring: str = ""
    parent: str = ""                  # e.g. class name for methods


@dataclass
class SymbolEdge:
    """A directed edge in the symbol graph."""
    source: str                       # Symbol key (e.g. "file:src/auth.py")
    target: str                       # Symbol key
    relation: str                     # "imports" | "calls" | "defines" | "contains"


@dataclass
class SymbolGraph:
    """In-memory symbol graph for a project."""
    nodes: Dict[str, SymbolNode] = field(default_factory=dict)
    edges: List[SymbolEdge] = field(default_factory=list)

    # Reverse indices for fast lookup
    _defined_in: Dict[str, str] = field(default_factory=dict)       # symbol_name → file_path
    _file_defines: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _file_imports: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _callers: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _callees: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    _class_methods: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, key: str, node: SymbolNode) -> None:
        self.nodes[key] = node

    def add_edge(self, edge: SymbolEdge) -> None:
        self.edges.append(edge)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def files_that_import(self, module_name: str) -> List[str]:
        """Find files that import a given module or symbol."""
        results: List[str] = []
        for file_path, imports in self._file_imports.items():
            if module_name in imports:
                results.append(file_path)
        return sorted(results)

    def symbols_in_file(self, file_path: str) -> List[SymbolNode]:
        """Get all symbols defined in a file."""
        names = self._file_defines.get(file_path, set())
        return [self.nodes[f"symbol:{n}@{file_path}"]
                for n in names
                if f"symbol:{n}@{file_path}" in self.nodes]

    def dependents_of(self, symbol_name: str) -> List[str]:
        """Find files/symbols that depend on a given symbol."""
        results: Set[str] = set()
        # Check which files import this symbol
        for file_path, imports in self._file_imports.items():
            for imp in imports:
                if symbol_name in imp:
                    results.add(file_path)
        return sorted(results)

    def methods_of_class(self, class_name: str) -> List[str]:
        """Get method names of a class."""
        return sorted(self._class_methods.get(class_name, set()))

    def summary(self) -> Dict:
        """Return a summary of the graph for debug/diagnostic purposes."""
        kind_counts: Dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            kind_counts[node.kind] += 1

        relation_counts: Dict[str, int] = defaultdict(int)
        for edge in self.edges:
            relation_counts[edge.relation] += 1

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "node_kinds": dict(kind_counts),
            "edge_relations": dict(relation_counts),
            "files_tracked": len(self._file_defines),
        }

    def to_dict(self) -> Dict:
        """Serialise the graph to a JSON-safe dict."""
        return {
            "nodes": {k: {
                "name": n.name,
                "kind": n.kind,
                "file_path": n.file_path,
                "line_start": n.line_start,
                "line_end": n.line_end,
                "language": n.language,
                "parent": n.parent,
            } for k, n in self.nodes.items()},
            "edges": [{"source": e.source, "target": e.target, "relation": e.relation}
                      for e in self.edges],
            "summary": self.summary(),
        }


# ── Graph singleton ───────────────────────────────────────────────────────────

_graphs: Dict[str, SymbolGraph] = {}


def get_graph(namespace: str) -> SymbolGraph:
    """Get or create the symbol graph for a namespace."""
    if namespace not in _graphs:
        _graphs[namespace] = SymbolGraph()
    return _graphs[namespace]


def clear_graph(namespace: str) -> None:
    """Remove the symbol graph for a namespace."""
    _graphs.pop(namespace, None)


# ── Call extraction (regex-based) ─────────────────────────────────────────────

# Pattern for function/method calls: identifier followed by (
_CALL_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")

# Names to ignore as "calls" (language keywords, builtins)
_CALL_IGNORE = frozenset({
    "if", "for", "while", "return", "print", "len", "range", "str", "int",
    "float", "list", "dict", "set", "tuple", "type", "isinstance", "super",
    "hasattr", "getattr", "setattr", "delattr", "open", "close",
    "True", "False", "None", "self", "cls", "import", "from",
    "def", "class", "async", "await", "yield",
    "function", "const", "let", "var", "new", "this", "constructor",
    "require", "export", "module", "console",
    "fmt", "log", "err", "nil", "make", "append",
    "System", "String", "Integer", "Boolean", "Object", "void",
})


def _extract_calls(text: str, own_name: str = "") -> Set[str]:
    """Extract function/method call names from code text."""
    calls: Set[str] = set()
    for m in _CALL_PATTERN.finditer(text):
        name = m.group(1)
        if name not in _CALL_IGNORE and name != own_name:
            calls.add(name)
    return calls


# ── Import extraction ─────────────────────────────────────────────────────────

def _extract_import_names(imports: List[str]) -> Set[str]:
    """Extract module and symbol names from import statements."""
    names: Set[str] = set()
    for line in imports:
        line = line.strip()
        if not line:
            continue
        # Python: from foo.bar import baz
        if line.startswith("from "):
            parts = line.split()
            if len(parts) >= 2:
                names.add(parts[1].split(".")[0])  # module root
            if "import" in line:
                after_import = line.split("import", 1)[1].strip()
                for sym in after_import.split(","):
                    sym = sym.strip().split(" as ")[0].strip()
                    if sym and sym != "*":
                        names.add(sym)
        # Python: import foo
        elif line.startswith("import "):
            after = line[7:].strip()
            for mod in after.split(","):
                mod = mod.strip().split(" as ")[0].strip()
                if mod:
                    names.add(mod.split(".")[0])
        # JS/TS: import ... from '...'
        elif "require(" in line or "from " in line:
            # Extract the module path
            for q in ('"', "'", '`'):
                if q in line:
                    start = line.index(q) + 1
                    end = line.index(q, start) if q in line[start:] else len(line)
                    mod_path = line[start:end]
                    # Get the base module name
                    names.add(mod_path.split("/")[-1].split(".")[0])
                    break
    return names


# ── Graph building ────────────────────────────────────────────────────────────

def build_graph_from_units(
    units: List[CodeUnit],
    file_path: str,
    namespace: str,
) -> None:
    """
    Process parsed code units and add them to the symbol graph.
    Called during ingestion for each file.
    """
    graph = get_graph(namespace)

    # Add file node
    file_key = f"file:{file_path}"
    graph.add_node(file_key, SymbolNode(
        name=Path(file_path).name,
        kind="file",
        file_path=file_path,
        language=units[0].language if units else "",
    ))

    # Track imports for this file
    file_imports: Set[str] = set()

    for unit in units:
        if unit.kind == "module":
            # Module-level code — extract imports
            if unit.imports:
                import_names = _extract_import_names(unit.imports)
                file_imports.update(import_names)
                graph._file_imports[file_path] = file_imports
                for imp_name in import_names:
                    graph.add_edge(SymbolEdge(
                        source=file_key,
                        target=f"module:{imp_name}",
                        relation="imports",
                    ))
            continue

        # Create symbol node
        symbol_key = f"symbol:{unit.name}@{file_path}"
        graph.add_node(symbol_key, SymbolNode(
            name=unit.name,
            kind=unit.kind,
            file_path=file_path,
            line_start=unit.start_line,
            line_end=unit.end_line,
            language=unit.language,
            docstring=unit.docstring[:200] if unit.docstring else "",
            parent=unit.class_name,
        ))

        # File → defines
        graph._file_defines[file_path].add(unit.name)
        graph._defined_in[unit.name] = file_path
        graph.add_edge(SymbolEdge(
            source=file_key,
            target=symbol_key,
            relation="defines",
        ))

        # Class → contains method
        if unit.kind == "method" and unit.class_name:
            class_key = f"symbol:{unit.class_name}@{file_path}"
            graph._class_methods[unit.class_name].add(unit.name)
            graph.add_edge(SymbolEdge(
                source=class_key,
                target=symbol_key,
                relation="contains",
            ))

        # Extract function calls
        calls = _extract_calls(unit.text, own_name=unit.name)
        for call_name in calls:
            graph._callees[unit.name].add(call_name)
            graph._callers[call_name].add(unit.name)
            graph.add_edge(SymbolEdge(
                source=symbol_key,
                target=f"symbol:{call_name}",
                relation="calls",
            ))

    logger.debug(
        "Symbol graph: %s → %d node(s), %d edge(s) (total: %d/%d)",
        file_path,
        len([n for n in graph.nodes if file_path in n]),
        len([e for e in graph.edges if file_path in e.source]),
        graph.node_count,
        graph.edge_count,
    )


# ── Query helpers ─────────────────────────────────────────────────────────────

def query_symbol_graph(
    namespace: str,
    question: str,
) -> Optional[str]:
    """
    Check if a question can be answered directly from the symbol graph.
    Returns a text answer or None if the graph cannot answer it.
    """
    graph = get_graph(namespace)
    if graph.node_count == 0:
        return None

    q = question.lower().strip()

    # "which files use/import X?"
    use_match = re.search(r"which\s+files?\s+(?:use|import|depend on|reference)\s+(?:the\s+)?(\w+)", q)
    if use_match:
        target = use_match.group(1)
        files = graph.files_that_import(target)
        if not files:
            # Also check callers
            callers = graph._callers.get(target, set())
            if callers:
                caller_files = set()
                for caller_name in callers:
                    if caller_name in graph._defined_in:
                        caller_files.add(graph._defined_in[caller_name])
                if caller_files:
                    files = sorted(caller_files)
        if files:
            file_list = "\n".join(f"  - {f}" for f in files)
            return f"Files that use/import `{target}`:\n{file_list}"
        return f"No files found that import or reference `{target}`."

    # "what depends on X?"
    dep_match = re.search(r"what\s+depends?\s+on\s+(?:the\s+)?(\w+)", q)
    if dep_match:
        target = dep_match.group(1)
        dependents = graph.dependents_of(target)
        if dependents:
            dep_list = "\n".join(f"  - {f}" for f in dependents)
            return f"Files/modules that depend on `{target}`:\n{dep_list}"
        return f"No dependents found for `{target}`."

    # "what methods does X have?"
    method_match = re.search(r"(?:what|which)\s+methods?\s+(?:does|has)\s+(\w+)", q)
    if method_match:
        query_name = method_match.group(1)
        # Case-insensitive class name lookup
        class_name = query_name
        for stored_name in graph._class_methods:
            if stored_name.lower() == query_name.lower():
                class_name = stored_name
                break
        methods = graph.methods_of_class(class_name)
        if methods:
            method_list = "\n".join(f"  - {m}()" for m in methods)
            return f"Methods of class `{class_name}`:\n{method_list}"

    # "what functions are in X file?"
    file_match = re.search(r"(?:what|which)\s+(?:functions?|classes?|symbols?)\s+(?:are\s+)?in\s+(.+?)(?:\?|$)", q)
    if file_match:
        target_file = file_match.group(1).strip().strip("'\"")
        for file_path in graph._file_defines:
            if target_file in file_path or Path(file_path).name == target_file:
                symbols = graph.symbols_in_file(file_path)
                if symbols:
                    sym_list = "\n".join(
                        f"  - {s.kind}: {s.name} (lines {s.line_start}–{s.line_end})"
                        for s in sorted(symbols, key=lambda s: s.line_start)
                    )
                    return f"Symbols in `{file_path}`:\n{sym_list}"

    return None
