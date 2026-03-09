"""
core/brain/tools.py
─────────────────────────────────────────────────────────────────────────────
Tool interface layer for the LLM Brain.

Each tool wraps an existing bakup.ai capability (search, retrieval,
analysis) behind a uniform callable interface with a JSON schema that
the LLM can use for function-calling.

Design:
  - Every tool has a name, description, parameter schema, and an execute()
    function that returns a plain dict (JSON-serialisable).
  - The registry exposes all tools as OpenAI-compatible function schemas
    so they can be passed directly to the LLM.
  - Tool execution is synchronous (runs on the server thread pool).
  - Results are intentionally compact — they are injected back into the
    LLM context window for reasoning, so verbosity must be controlled.

Safety:
  - Tools are *read-only* — they never mutate project data.
  - Execution is bounded by a per-query call budget (see brain.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ── Tool definition ────────────────────────────────────────────────────────────

@dataclass
class ToolParam:
    """A single parameter of a tool."""
    name: str
    type: str                  # "string" | "integer" | "boolean"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """
    A callable tool that the LLM brain can invoke.

    Attributes:
        name:        Machine-readable tool name (e.g. "search_logs").
        description: One-line description for the LLM system prompt.
        params:      Ordered list of parameters.
        execute:     The actual function to call.  Signature:
                     ``execute(**kwargs) -> dict``
    """
    name: str
    description: str
    params: List[ToolParam]
    execute: Callable[..., Dict[str, Any]]

    def to_openai_schema(self) -> dict:
        """
        Return an OpenAI-compatible function schema for this tool.

        Compatible with OpenAI, Azure OpenAI, and Anthropic tool formats
        (the outer wrapper differs per provider — this method returns the
        inner ``function`` definition).
        """
        properties = {}
        required = []
        for p in self.params:
            prop: dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


# ── Tool execution result ─────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    ms: float = 0.0
    error: Optional[str] = None


# ── Tool implementations ──────────────────────────────────────────────────────

def _search_logs(query: str, namespace: str, top_k: int = 8) -> dict:
    """
    Semantic + keyword search across indexed log files.

    Returns log entries ranked by relevance with trend/cluster analysis.
    """
    from core.retrieval.agent import _execute_search_logs, StructuredEvidence

    evidence = StructuredEvidence()
    _execute_search_logs(query, namespace, top_k, evidence)

    return {
        "logs_found": len(evidence.logs),
        "code_found": len(evidence.code),
        "log_entries": [
            {
                "file": c.source_file,
                "lines": f"{c.line_start}-{c.line_end}",
                "confidence": round(c.confidence, 3),
                "text": c.text[:600],
            }
            for c in evidence.logs[:8]
        ],
        "trend_summary": evidence.trend_summary or "",
        "cluster_summary": evidence.cluster_summary or "",
        "confidence_summary": evidence.confidence_summary or "",
        "file_distribution": evidence.file_aggregation_summary or "",
        "error_cluster_report": evidence.error_cluster_report,
        "causal_confidence": evidence.causal_confidence,
    }


def _search_code(query: str, namespace: str, top_k: int = 8) -> dict:
    """
    Semantic search across indexed source code.

    Returns code chunks ranked by relevance with function/class metadata.
    """
    from core.retrieval.agent import _execute_search_code, StructuredEvidence

    evidence = StructuredEvidence()
    _execute_search_code(query, namespace, top_k, evidence)

    return {
        "code_found": len(evidence.code),
        "code_chunks": [
            {
                "file": c.source_file,
                "lines": f"{c.line_start}-{c.line_end}",
                "confidence": round(c.confidence, 3),
                "function": getattr(c, "function_name", "") or "",
                "class": getattr(c, "class_name", "") or "",
                "text": c.text[:800],
            }
            for c in evidence.code[:8]
        ],
    }


def _retrieve_dependencies(file_path: str, namespace: str) -> dict:
    """
    Look up which files depend on or are depended upon by a given file.

    Uses the symbol graph built during ingestion.
    """
    from core.ingestion.symbol_graph import get_graph

    graph = get_graph(namespace)
    if graph.node_count == 0:
        return {"dependents": [], "imports": [], "graph_available": False}

    dependents = list(graph.dependents_of(file_path))
    # Also check what this file imports
    imports: List[str] = []
    try:
        imports = list(graph.imports_of(file_path))
    except (AttributeError, KeyError):
        pass

    return {
        "file": file_path,
        "dependents": dependents[:20],
        "imports": imports[:20],
        "graph_available": True,
    }


def _get_architecture_summary(namespace: str) -> dict:
    """
    Return the cached architecture overview for the indexed project.

    Includes module breakdown, entry points, and file counts.
    """
    from core.analysis.architecture import get_architecture

    arch = get_architecture(namespace)
    if not arch:
        return {"available": False, "summary": "No architecture data cached."}
    return {
        "available": True,
        "total_files": arch.total_files,
        "modules": [
            {"name": m.name, "files": m.file_count, "description": m.description}
            for m in arch.modules[:20]
        ] if hasattr(arch, "modules") else [],
        "summary": arch.summary_text()[:3000],
    }


def _get_error_clusters(namespace: str) -> dict:
    """
    Cluster log error patterns across all indexed log entries.

    Returns error pattern clusters with occurrence counts, trends,
    and causal confidence scoring.
    """
    from core.embeddings.embedder import embed_query
    from core.retrieval.vector_store import severity_search
    from core.retrieval.ranker import rank_results

    # Pull error-severity log chunks
    sev_chunks = severity_search(namespace, severity="error", top_k=30)
    if not sev_chunks:
        return {"clusters": [], "total_errors": 0}

    ranked = rank_results(sev_chunks)

    try:
        from core.analysis.error_clustering import cluster_error_patterns
        ecr = cluster_error_patterns(ranked)
        return ecr.to_dict()
    except Exception as exc:
        return {"clusters": [], "total_errors": len(ranked), "error": str(exc)}


def _get_file_context(file_path: str, namespace: str) -> dict:
    """
    Retrieve all indexed chunks for a specific file path.

    Useful when the LLM wants to dive deeper into a particular file
    after seeing it mentioned in search results.
    """
    from core.retrieval.vector_store import keyword_search
    from core.retrieval.ranker import rank_results

    chunks = keyword_search(namespace, [file_path], top_k=15)
    if not chunks:
        return {"file": file_path, "chunks_found": 0, "chunks": []}

    ranked = rank_results(chunks)
    return {
        "file": file_path,
        "chunks_found": len(ranked),
        "chunks": [
            {
                "lines": f"{c.line_start}-{c.line_end}",
                "type": c.source_type,
                "confidence": round(c.confidence, 3),
                "text": c.text[:800],
            }
            for c in ranked[:10]
        ],
    }


def _query_symbol_graph(query: str, namespace: str) -> dict:
    """
    Answer structural questions from the symbol graph.

    Handles questions like "which files import X", "what depends on Y",
    "what methods does Z have".
    """
    from core.ingestion.symbol_graph import query_symbol_graph

    answer = query_symbol_graph(namespace, query)
    return {
        "answered": bool(answer),
        "answer": answer or "No structural information available for this query.",
    }


def _cross_analyse_logs_code(namespace: str, query: str, top_k: int = 8) -> dict:
    """
    Run the full cross-analysis pipeline: find log errors, extract code
    references, and link them to source code.

    This is the most powerful tool — it correlates logs with code to
    identify root causes.
    """
    from core.retrieval.planner import create_plan, QuestionType
    from core.retrieval.agent import execute_plan, build_evidence_context

    plan = create_plan(query, QuestionType.ROOT_CAUSE)
    evidence = execute_plan(plan, query, namespace, top_k=top_k)

    context = build_evidence_context(evidence)

    return {
        "logs_found": len(evidence.logs),
        "code_found": len(evidence.code),
        "references": len(evidence.references_found),
        "dependencies": evidence.dependencies[:15],
        "has_cross_analysis": evidence.has_cross_analysis,
        "architecture_available": bool(evidence.architecture_summary),
        "causal_confidence": evidence.causal_confidence,
        "error_cluster_report": evidence.error_cluster_report,
        "trend_detection_report": evidence.trend_detection_report,
        "evidence_context": context[:4000],
        "step_trace": [
            {
                "step": sr.step_type,
                "description": sr.description,
                "chunks": sr.chunks_found,
                "ms": sr.ms,
            }
            for sr in evidence.step_results
        ],
        "total_ms": evidence.total_ms,
    }


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS: List[Tool] = [
    Tool(
        name="search_logs",
        description="Search indexed log files for errors, exceptions, warnings, or any log entries matching a query. Returns ranked results with trend and cluster analysis.",
        params=[
            ToolParam("query", "string", "Search query describing what to look for in logs"),
            ToolParam("namespace", "string", "Project namespace"),
            ToolParam("top_k", "integer", "Number of results to return (default 8)", required=False),
        ],
        execute=_search_logs,
    ),
    Tool(
        name="search_code",
        description="Search indexed source code for functions, classes, modules, or code patterns matching a query. Returns code chunks with file/line/function metadata.",
        params=[
            ToolParam("query", "string", "Search query describing what code to find"),
            ToolParam("namespace", "string", "Project namespace"),
            ToolParam("top_k", "integer", "Number of results to return (default 8)", required=False),
        ],
        execute=_search_code,
    ),
    Tool(
        name="retrieve_dependencies",
        description="Look up which files depend on or import a specific file. Uses the project's symbol graph to trace dependency chains.",
        params=[
            ToolParam("file_path", "string", "Path of the file to check dependencies for"),
            ToolParam("namespace", "string", "Project namespace"),
        ],
        execute=_retrieve_dependencies,
    ),
    Tool(
        name="get_architecture_summary",
        description="Get the project architecture overview including module breakdown, entry points, file counts, and structural summary.",
        params=[
            ToolParam("namespace", "string", "Project namespace"),
        ],
        execute=_get_architecture_summary,
    ),
    Tool(
        name="get_error_clusters",
        description="Cluster all detected error patterns across indexed log files. Returns error pattern groups with occurrence counts, severity, timestamps, and causal confidence.",
        params=[
            ToolParam("namespace", "string", "Project namespace"),
        ],
        execute=_get_error_clusters,
    ),
    Tool(
        name="get_file_context",
        description="Retrieve all indexed content (code and logs) for a specific file path. Use this to dive deeper into a particular file after seeing it in search results.",
        params=[
            ToolParam("file_path", "string", "File path to retrieve content for"),
            ToolParam("namespace", "string", "Project namespace"),
        ],
        execute=_get_file_context,
    ),
    Tool(
        name="query_symbol_graph",
        description="Answer structural questions about the codebase: which files import a module, what depends on a class, what methods a file has. Uses the symbol graph built during indexing.",
        params=[
            ToolParam("query", "string", "Structural question about the codebase"),
            ToolParam("namespace", "string", "Project namespace"),
        ],
        execute=_query_symbol_graph,
    ),
    Tool(
        name="cross_analyse",
        description="Run full root-cause analysis: find log errors, extract code references from stack traces, link errors to source code, analyse dependencies, and compute causal confidence. This is the most comprehensive analysis tool.",
        params=[
            ToolParam("query", "string", "Describe the issue or error to investigate"),
            ToolParam("namespace", "string", "Project namespace"),
            ToolParam("top_k", "integer", "Number of results per step (default 8)", required=False),
        ],
        execute=_cross_analyse_logs_code,
    ),
]

# Lookup by name
TOOL_MAP: Dict[str, Tool] = {t.name: t for t in TOOLS}


def get_tool_schemas_openai() -> List[dict]:
    """Return all tool definitions in OpenAI function-calling format."""
    return [
        {"type": "function", "function": t.to_openai_schema()}
        for t in TOOLS
    ]


def get_tool_schemas_anthropic() -> List[dict]:
    """Return all tool definitions in Anthropic tool-use format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.to_openai_schema()["parameters"],
        }
        for t in TOOLS
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> ToolResult:
    """
    Execute a tool by name with the given arguments.

    Returns a ToolResult with the execution output and timing.
    """
    tool = TOOL_MAP.get(name)
    if not tool:
        return ToolResult(
            tool_name=name,
            arguments=arguments,
            result={},
            error=f"Unknown tool: {name}",
        )

    t0 = time.perf_counter()
    try:
        result = tool.execute(**arguments)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return ToolResult(
            tool_name=name,
            arguments=arguments,
            result=result,
            ms=ms,
        )
    except Exception as exc:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return ToolResult(
            tool_name=name,
            arguments=arguments,
            result={},
            ms=ms,
            error=str(exc),
        )
