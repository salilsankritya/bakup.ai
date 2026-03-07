"""Quick validation of the hybrid router + ingestion filtering."""
import sys, os
sys.path.insert(0, ".")

from core.router.router import route_query, CONFIDENCE_THRESHOLD, get_scope_message

tests = [
    ("hello",                          "conversational"),
    ("are you gay",                    "conversational"),
    ("tell me a joke",                 "conversational"),
    ("you are stupid",                 "conversational"),
    ("any errors in the logs?",        "project_query"),
    ("is the code optimized properly", "project_query"),
    ("what is the capital of France",  "unrelated"),
    ("explain quantum physics",        "unrelated"),
    ("what function handles auth",     "project_query"),
    ("show me the traceback",          "project_query"),
]

print(f"Router confidence threshold: {CONFIDENCE_THRESHOLD}")
print()

all_pass = True
for query, expected in tests:
    r = route_query(query)
    status = "PASS" if r.intent == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  [{status}] {query!r:45s} -> {r.intent:20s} ({r.confidence:.0%}, {r.source})")

print()

# Test ingestion filtering
from core.ingestion.file_walker import SKIP_DIRS, SKIP_EXTENSIONS
assert "model-weights" in SKIP_DIRS, "model-weights not in SKIP_DIRS"
assert "vectordb" in SKIP_DIRS, "vectordb not in SKIP_DIRS"
assert ".bin" in SKIP_EXTENSIONS, ".bin not in SKIP_EXTENSIONS"
assert ".vocab" in SKIP_EXTENSIONS, ".vocab not in SKIP_EXTENSIONS"
assert ".model" in SKIP_EXTENSIONS, ".model not in SKIP_EXTENSIONS"
assert ".gguf" in SKIP_EXTENSIONS, ".gguf not in SKIP_EXTENSIONS"
print("[PASS] Ingestion filtering: SKIP_DIRS and SKIP_EXTENSIONS correct")

# Test scope message
msg = get_scope_message()
assert "bakup.ai" in msg
print(f"[PASS] Scope message: {msg[:60]}...")

print()
if all_pass:
    print("All router tests passed!")
else:
    print("Some tests FAILED — check output above")
    sys.exit(1)
