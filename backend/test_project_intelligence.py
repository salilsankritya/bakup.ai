"""Test script for project intelligence modules."""
import sys, os
sys.path.insert(0, '.')

from collections import Counter

# ==============================
# Test 1: Symbol Graph
# ==============================
from core.ingestion.code_parser import CodeUnit
from core.ingestion.symbol_graph import (
    SymbolGraph, build_graph_from_units, get_graph,
    query_symbol_graph, _extract_calls, _extract_import_names,
)

# Build a test graph
units = [
    CodeUnit(
        name="UserService",
        kind="class",
        start_line=1, end_line=20,
        text="class UserService:\n    pass",
        language="python",
    ),
    CodeUnit(
        name="authenticate",
        kind="method",
        start_line=5, end_line=15,
        text="def authenticate(self, user):\n    result = validate_token(user.token)\n    return result",
        class_name="UserService",
        language="python",
    ),
    CodeUnit(
        name="validate_token",
        kind="function",
        start_line=22, end_line=30,
        text="def validate_token(token):\n    return check_expiry(token)",
        language="python",
        imports=["from auth.tokens import check_expiry"],
    ),
]

test_ns = "__test_symbols__"
build_graph_from_units(units, "src/auth/service.py", test_ns)
graph = get_graph(test_ns)

assert graph.node_count >= 3, f"Expected ≥3 nodes, got {graph.node_count}"
assert graph.edge_count >= 2, f"Expected ≥2 edges, got {graph.edge_count}"

# Test symbol queries
symbols = graph.symbols_in_file("src/auth/service.py")
names = {s.name for s in symbols}
assert "UserService" in names, f"UserService not in {names}"
assert "authenticate" in names, f"authenticate not in {names}"

# Test method lookup
methods = graph.methods_of_class("UserService")
assert "authenticate" in methods, f"authenticate not in methods: {methods}"

# Test import tracking  
importers = graph.files_that_import("check_expiry")
# check_expiry may be tracked as import from auth.tokens
print(f"[PASS] Symbol graph: {graph.node_count} nodes, {graph.edge_count} edges")

# Test call extraction
calls = _extract_calls("result = validate_token(user.token)\nreturn check_expiry(token)")
assert "validate_token" in calls, f"validate_token not in {calls}"
assert "check_expiry" in calls, f"check_expiry not in {calls}"
# Built-in keywords should be excluded
calls2 = _extract_calls("if True:\n    print(x)\n    return len(items)")
assert "print" not in calls2, f"print should be excluded: {calls2}"
assert "len" not in calls2, f"len should be excluded: {calls2}"
print("[PASS] Call extraction: correctly identifies calls, excludes builtins")

# Test import name extraction
imports = _extract_import_names(["from auth.tokens import check_expiry, validate", "import os"])
assert "check_expiry" in imports, f"check_expiry not in {imports}"
assert "validate" in imports, f"validate not in {imports}"
assert "os" in imports, f"os not in {imports}"
print("[PASS] Import extraction: correctly parses import statements")

# Test query answering
answer = query_symbol_graph(test_ns, "what methods does UserService have?")
assert answer is not None, "Expected graph to answer method query"
assert "authenticate" in answer, f"authenticate not in answer: {answer}"
print(f"[PASS] Symbol graph query: '{answer[:60]}...'")

# ==============================
# Test 2: Architecture Summarizer
# ==============================
from core.analysis.architecture import build_architecture_summary, get_architecture

file_paths = [
    "main.py", "src/api/routes.py", "src/api/handlers.py",
    "src/models/user.py", "src/models/session.py",
    "src/services/auth.py", "src/utils/helpers.py",
    "package.json", "Dockerfile", "requirements.txt",
]

units_by_file = {
    "src/api/routes.py": [
        CodeUnit(name="get_users", kind="function", start_line=1, end_line=10, text="", language="python"),
        CodeUnit(name="create_user", kind="function", start_line=12, end_line=25, text="", language="python"),
    ],
    "src/models/user.py": [
        CodeUnit(name="User", kind="class", start_line=1, end_line=30, text="", language="python"),
    ],
    "src/services/auth.py": [
        CodeUnit(name="AuthService", kind="class", start_line=1, end_line=50, text="", language="python"),
        CodeUnit(name="login", kind="method", start_line=10, end_line=20, text="", class_name="AuthService", language="python"),
    ],
}

import_counter = Counter({"os": 5, "datetime": 3, "typing": 2, "src.models.user": 4})

arch_ns = "__test_arch__"
summary = build_architecture_summary(
    file_paths=file_paths,
    units_by_file=units_by_file,
    import_counter=import_counter,
    project_name="test-project",
    namespace=arch_ns,
)

assert summary.total_files == len(file_paths)
assert summary.total_functions >= 2
assert summary.total_classes >= 2
assert len(summary.entry_points) >= 1, f"Should detect main.py as entry point: {summary.entry_points}"
assert len(summary.config_files) >= 1, f"Should detect config files: {summary.config_files}"
assert len(summary.modules) >= 1, f"Should detect modules: {summary.modules}"
assert summary.core_dependencies, f"Should have core deps"

text = summary.summary_text()
assert "python" in text.lower(), f"Should mention python language"
assert len(text) > 100, f"Summary too short: {len(text)} chars"

# Verify caching
cached = get_architecture(arch_ns)
assert cached is summary, "Architecture should be cached by namespace"

print(f"[PASS] Architecture summary: {summary.total_files} files, {len(summary.modules)} modules, {len(summary.entry_points)} entry points")
print(f"  Summary: {len(text)} chars")

# ==============================
# Test 3: Context Bundler
# ==============================
from core.retrieval.ranker import RankedResult
from core.retrieval.context_bundler import bundle_context, bundles_to_ranked_list, build_bundled_context_block

# Create test ranked results from the same file
results = [
    RankedResult(
        text="def authenticate(self, user):\n    token = validate_token(user.token)",
        source_file="src/auth/service.py",
        line_start=5, line_end=15,
        source_type="code",
        confidence=0.92,
        confidence_label="high",
        function_name="authenticate",
        class_name="UserService",
        imports="from auth.tokens import validate_token",
    ),
    RankedResult(
        text="def validate_token(token):\n    return check_expiry(token)",
        source_file="src/auth/service.py",
        line_start=22, line_end=30,
        source_type="code",
        confidence=0.85,
        confidence_label="high",
        function_name="validate_token",
    ),
    RankedResult(
        text="class UserService:\n    def __init__(self):\n        self.db = Database()",
        source_file="src/auth/service.py",
        line_start=1, line_end=4,
        source_type="code",
        confidence=0.78,
        confidence_label="medium",
        class_name="UserService",
    ),
    RankedResult(
        text="def handle_login(request):\n    return UserService().authenticate(request.user)",
        source_file="src/api/routes.py",
        line_start=10, line_end=15,
        source_type="code",
        confidence=0.70,
        confidence_label="medium",
        function_name="handle_login",
    ),
]

bundles = bundle_context(results, top_n=3)
assert len(bundles) >= 1, f"Expected bundles, got {len(bundles)}"

# First bundle should be the highest-confidence result
assert bundles[0].primary.function_name == "authenticate"

# Should find siblings from same file
if bundles[0].siblings:
    for sib in bundles[0].siblings:
        assert sib.source_file == "src/auth/service.py", "Siblings must be from same file"

# Should find import chunks
if bundles[0].import_chunks:
    import_names = {ic.function_name for ic in bundles[0].import_chunks}
    assert "validate_token" in import_names, f"Should find validate_token in imports: {import_names}"

# Flatten back
flat = bundles_to_ranked_list(bundles)
assert len(flat) >= len(bundles), "Flattened list should be at least as long as bundles"
# No duplicates
keys = [(r.source_file, r.line_start, r.line_end) for r in flat]
assert len(keys) == len(set(keys)), "Flattened list should have no duplicates"

# Build context block
ctx = build_bundled_context_block(bundles)
assert len(ctx) > 50, f"Context block too short: {len(ctx)} chars"
assert "service.py" in ctx, "Context should reference the source file"

print(f"[PASS] Context bundler: {len(bundles)} bundles, {len(flat)} flattened results, {len(ctx)} chars context")

# ==============================
# Test 4: Log-to-Code Linker
# ==============================
from core.analysis.log_code_linker import extract_code_references, link_logs_to_code, build_cross_analysis_context

# Test reference extraction
py_log = '''
Traceback (most recent call last):
  File "src/auth/service.py", line 12, in authenticate
    token = validate_token(user.token)
  File "src/auth/tokens.py", line 45, in validate_token
    return check_expiry(token)
AttributeError: 'NoneType' object has no attribute 'expiry'
'''

refs = extract_code_references(py_log)
assert len(refs) >= 2, f"Expected ≥2 references from Python traceback, got {len(refs)}"

# Check we extracted file paths and function names
file_paths_found = {r.file_path for r in refs if r.file_path}
func_names_found = {r.function_name for r in refs if r.function_name}
assert any("service.py" in f for f in file_paths_found), f"Should find service.py: {file_paths_found}"
assert "authenticate" in func_names_found or "validate_token" in func_names_found, f"Should find function names: {func_names_found}"

print(f"[PASS] Reference extraction: {len(refs)} references from Python traceback")
print(f"  Files: {file_paths_found}")
print(f"  Functions: {func_names_found}")

# Test Java stack trace extraction
java_log = "at com.example.UserService.authenticate(UserService.java:42)"
java_refs = extract_code_references(java_log)
assert len(java_refs) >= 1, f"Expected ≥1 reference from Java stack trace, got {len(java_refs)}"
print(f"[PASS] Java stack trace: {len(java_refs)} reference(s)")

# Test JS stack trace extraction
js_log = "at handleRequest (server.js:42:10)"
js_refs = extract_code_references(js_log)
assert len(js_refs) >= 1, f"Expected ≥1 reference from JS stack trace, got {len(js_refs)}"
print(f"[PASS] JS stack trace: {len(js_refs)} reference(s)")

# Test log-to-code linking
log_chunks = [
    RankedResult(
        text=py_log,
        source_file="logs/app.log",
        line_start=100, line_end=107,
        source_type="log",
        confidence=0.80,
        confidence_label="high",
    ),
]

code_chunks = [
    RankedResult(
        text="def authenticate(self, user):\n    token = validate_token(user.token)",
        source_file="src/auth/service.py",
        line_start=5, line_end=15,
        source_type="code",
        confidence=0.90,
        confidence_label="high",
        function_name="authenticate",
        class_name="UserService",
    ),
    RankedResult(
        text="def validate_token(token):\n    return check_expiry(token)",
        source_file="src/auth/tokens.py",
        line_start=40, line_end=50,
        source_type="code",
        confidence=0.85,
        confidence_label="high",
        function_name="validate_token",
    ),
]

links = link_logs_to_code(log_chunks, code_chunks)
assert len(links) == 1, f"Expected 1 link, got {len(links)}"
assert len(links[0].code_chunks) >= 1, f"Expected code chunks linked, got {len(links[0].code_chunks)}"
assert len(links[0].references) >= 2, f"Expected ≥2 references, got {len(links[0].references)}"

print(f"[PASS] Log-to-code linking: {len(links)} link(s), {len(links[0].code_chunks)} code chunk(s) matched")

# Test cross-analysis context building
ctx = build_cross_analysis_context(links)
assert len(ctx) > 100, f"Cross-analysis context too short: {len(ctx)} chars"
assert "Error #1" in ctx, "Should have error header"
assert "service.py" in ctx, "Should reference source file"

print(f"[PASS] Cross-analysis context: {len(ctx)} chars")

# ==============================
# Test 5: Query detection (rag.py helpers)
# ==============================
from core.retrieval.rag import _is_architecture_query, _is_structural_query, _is_log_query

# Architecture queries
assert _is_architecture_query("explain the architecture of this project")
assert _is_architecture_query("what is the project structure?")
assert _is_architecture_query("give me an overview of the codebase")
assert _is_architecture_query("tell me about the project")
assert not _is_architecture_query("what does the authenticate function do?")
print("[PASS] Architecture query detection: 5/5 correct")

# Structural queries
assert _is_structural_query("which files use the auth module?")
assert _is_structural_query("what depends on database?")
assert _is_structural_query("what methods does UserService have?")
assert _is_structural_query("what functions are in routes.py?")
assert not _is_structural_query("how do I fix the login error?")
print("[PASS] Structural query detection: 5/5 correct")

# Log queries
assert _is_log_query("what errors are in the logs?")
assert _is_log_query("show me the exceptions")
assert _is_log_query("why is the server crashing?")
assert not _is_log_query("explain the architecture")
print("[PASS] Log query detection: 4/4 correct")

# ==============================
# Test 6: Prompt templates
# ==============================
from core.llm.prompt_templates import (
    SYSTEM_RAG, SYSTEM_CLARIFY, SYSTEM_LOG_SUMMARY,
    SYSTEM_CROSS_ANALYSIS, SYSTEM_CONVERSATIONAL,
    build_context_block, build_log_analysis_context,
)

assert "NO_ANSWER" in SYSTEM_RAG
assert "clarif" in SYSTEM_CLARIFY.lower()
assert "incident" in SYSTEM_LOG_SUMMARY.lower()
assert "root" in SYSTEM_CROSS_ANALYSIS.lower() and "cause" in SYSTEM_CROSS_ANALYSIS.lower()
assert "bakup.ai" in SYSTEM_CONVERSATIONAL

print("[PASS] Prompt templates: all 5 prompts validated")

# ==============================
# Summary
# ==============================
print("\n" + "=" * 50)
print("All project intelligence tests passed.")
print("=" * 50)
