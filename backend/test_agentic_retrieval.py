"""Test script for the agentic retrieval system."""
import sys, os
sys.path.insert(0, '.')

# ==============================
# Test 1: Planner — Question Classification
# ==============================
from core.retrieval.planner import (
    QuestionType, StepType, classify_question, create_plan,
)

# Root-cause questions
assert classify_question("Why is payment failing?") == QuestionType.ROOT_CAUSE
assert classify_question("What caused the crash?") == QuestionType.ROOT_CAUSE
assert classify_question("Diagnose the authentication error") == QuestionType.ROOT_CAUSE
assert classify_question("Investigate the timeout issue") == QuestionType.ROOT_CAUSE
assert classify_question("Why are users getting 500 errors?") == QuestionType.ROOT_CAUSE
print("✓ Test 1 passed: Root-cause classification")

# ==============================
# Test 2: Planner — Log Analysis Classification
# ==============================
assert classify_question("Show me the errors") == QuestionType.LOG_ANALYSIS
assert classify_question("Are there any exceptions in the logs?") == QuestionType.LOG_ANALYSIS
assert classify_question("What traceback errors occurred?") == QuestionType.LOG_ANALYSIS
assert classify_question("Show recent failures") == QuestionType.LOG_ANALYSIS
print("✓ Test 2 passed: Log analysis classification")

# ==============================
# Test 3: Planner — Code Analysis Classification
# ==============================
assert classify_question("How does the auth module work?") == QuestionType.CODE_ANALYSIS
assert classify_question("Explain the payment processing code") == QuestionType.CODE_ANALYSIS
assert classify_question("Show me the implementation of login") == QuestionType.CODE_ANALYSIS
print("✓ Test 3 passed: Code analysis classification")

# ==============================
# Test 4: Planner — Architecture Classification
# ==============================
assert classify_question("Explain the project architecture") == QuestionType.ARCHITECTURE
assert classify_question("What is the project structure?") == QuestionType.ARCHITECTURE
assert classify_question("Give me an overview of the codebase") == QuestionType.ARCHITECTURE
print("✓ Test 4 passed: Architecture classification")

# ==============================
# Test 5: Planner — Structural Classification
# ==============================
assert classify_question("Which files import auth?") == QuestionType.STRUCTURAL
assert classify_question("What depends on database?") == QuestionType.STRUCTURAL
assert classify_question("What methods does UserService have?") == QuestionType.STRUCTURAL
print("✓ Test 5 passed: Structural classification")

# ==============================
# Test 6: Planner — General Fallback
# ==============================
assert classify_question("Tell me about the database") == QuestionType.GENERAL
assert classify_question("How is data stored?") == QuestionType.GENERAL
print("✓ Test 6 passed: General fallback classification")

# ==============================
# Test 7: Planner — Plan Creation (Root Cause)
# ==============================
plan = create_plan("Why is payment failing?")
assert plan.question_type == QuestionType.ROOT_CAUSE
assert plan.fast_path is False
assert len(plan.steps) == 6
step_types = [s.step_type for s in plan.steps]
assert StepType.SEARCH_LOGS in step_types
assert StepType.EXTRACT_REFS in step_types
assert StepType.RETRIEVE_CODE in step_types
assert StepType.GET_DEPS in step_types
assert StepType.GET_ARCH in step_types
assert StepType.CROSS_ANALYSIS in step_types
assert plan.reasoning  # Not empty
print("✓ Test 7 passed: Root-cause plan has 6 steps")

# ==============================
# Test 8: Planner — Plan Creation (Architecture Fast-Path)
# ==============================
plan = create_plan("Explain the project architecture")
assert plan.question_type == QuestionType.ARCHITECTURE
assert plan.fast_path is True
assert len(plan.steps) == 1
assert plan.steps[0].step_type == StepType.GET_ARCH
print("✓ Test 8 passed: Architecture fast-path plan")

# ==============================
# Test 9: Planner — Plan Creation (Log Analysis)
# ==============================
plan = create_plan("Show me the errors in the logs")
assert plan.question_type == QuestionType.LOG_ANALYSIS
assert plan.fast_path is False
assert len(plan.steps) == 3
step_types = [s.step_type for s in plan.steps]
assert StepType.SEARCH_LOGS in step_types
assert StepType.EXTRACT_REFS in step_types
assert StepType.CROSS_ANALYSIS in step_types
print("✓ Test 9 passed: Log analysis plan has 3 steps")

# ==============================
# Test 10: Planner — Plan Creation (Code Analysis)
# ==============================
plan = create_plan("How does authentication work?")
assert plan.question_type == QuestionType.CODE_ANALYSIS
assert plan.fast_path is False
step_types = [s.step_type for s in plan.steps]
assert StepType.SEARCH_CODE in step_types
assert StepType.BUNDLE_CONTEXT in step_types
assert StepType.GET_DEPS in step_types
print("✓ Test 10 passed: Code analysis plan")

# ==============================
# Test 11: Planner — Plan Step Dependencies
# ==============================
plan = create_plan("Why is the server crashing?")
# extract_refs should depend on search_logs
refs_step = [s for s in plan.steps if s.step_type == StepType.EXTRACT_REFS]
assert len(refs_step) == 1
assert refs_step[0].depends_on == "search_logs"

# cross_analysis should depend on retrieve_code
cross_step = [s for s in plan.steps if s.step_type == StepType.CROSS_ANALYSIS]
assert len(cross_step) == 1
assert cross_step[0].depends_on == "retrieve_code"
print("✓ Test 11 passed: Plan step dependencies")

# ==============================
# Test 12: Session Memory — Basic Operations
# ==============================
from core.retrieval.session import (
    get_session, add_turn, get_session_info,
    clear_session, clear_all_sessions,
    Session, ConversationTurn,
)

test_ns = "__test_agentic__"
clear_session(test_ns)

session = get_session(test_ns)
assert session.turn_count == 0
assert session.recent_files == []
assert session.format_context() == ""
assert not session.is_follow_up("What errors occurred?")
print("✓ Test 12 passed: Empty session operations")

# ==============================
# Test 13: Session Memory — Add Turns
# ==============================
add_turn(
    test_ns,
    question="What errors occurred?",
    answer="Found 3 errors in payment.py: NullPointerException at line 42...",
    source_files=["payment.py", "auth.py"],
    question_type="log_analysis",
    evidence_summary="5 logs, 3 code chunks",
)

session = get_session(test_ns)
assert session.turn_count == 1
assert "payment.py" in session.recent_files
assert session.recent_question_types == ["log_analysis"]
print("✓ Test 13 passed: Add turn and retrieve")

# ==============================
# Test 14: Session Memory — Follow-Up Detection
# ==============================
session = get_session(test_ns)
assert session.is_follow_up("why?") is True            # Short question
assert session.is_follow_up("tell me more") is True     # Continuation marker
assert session.is_follow_up("which file has the most?") is True   # "which file"
assert session.is_follow_up("what about that error?") is True     # "that"
assert session.is_follow_up("also show warnings") is True         # "also"
print("✓ Test 14 passed: Follow-up detection")

# ==============================
# Test 15: Session Memory — Context Formatting
# ==============================
context = session.format_context()
assert "Prior Conversation Context" in context
assert "What errors occurred?" in context
assert "payment.py" in context
assert "log_analysis" in context
print("✓ Test 15 passed: Context formatting")

# ==============================
# Test 16: Session Memory — Prior Context for Follow-Up
# ==============================
prior = session.get_prior_context_for_follow_up()
assert prior["prior_question"] == "What errors occurred?"
assert prior["prior_type"] == "log_analysis"
assert "payment.py" in prior["prior_files"]
assert prior["prior_evidence"] == "5 logs, 3 code chunks"
print("✓ Test 16 passed: Prior context retrieval")

# ==============================
# Test 17: Session Memory — Turn Eviction
# ==============================
# Add 12 turns to a session with max 10
test_ns2 = "__test_evict__"
clear_session(test_ns2)
for i in range(12):
    add_turn(test_ns2, f"Question {i}", f"Answer {i}")

session2 = get_session(test_ns2)
assert session2.turn_count == 10  # Oldest 2 evicted
assert session2.turns[0].question == "Question 2"  # First 2 gone
assert session2.turns[-1].question == "Question 11"
print("✓ Test 17 passed: Turn eviction at capacity")

# ==============================
# Test 18: Session Memory — Info Endpoint
# ==============================
info = get_session_info(test_ns)
assert info["namespace"] == test_ns
assert info["turn_count"] == 1
assert info["is_empty"] is False
assert len(info["turns"]) == 1
assert info["turns"][0]["question"] == "What errors occurred?"
print("✓ Test 18 passed: Session info endpoint")

# ==============================
# Test 19: Session Memory — Clear
# ==============================
clear_session(test_ns)
session = get_session(test_ns)
assert session.turn_count == 0
print("✓ Test 19 passed: Session clear")

# ==============================
# Test 20: Agent — StructuredEvidence
# ==============================
from core.retrieval.agent import StructuredEvidence, StepResult, build_evidence_context
from core.retrieval.ranker import RankedResult

evidence = StructuredEvidence()
assert evidence.total_chunks == 0
assert evidence.has_logs is False
assert evidence.has_code is False
assert evidence.has_cross_analysis is False

# Add mock log chunk
log_chunk = RankedResult(
    text="ERROR 2024-01-15 10:30:00 Payment failed: NullPointerException",
    source_file="logs/app.log",
    line_start=100, line_end=105,
    source_type="log",
    confidence=0.75, confidence_label="high",
    severity="error",
)
evidence.logs.append(log_chunk)

# Add mock code chunk
code_chunk = RankedResult(
    text="def process_payment(amount):\n    result = gateway.charge(amount)\n    return result",
    source_file="src/payment.py",
    line_start=10, line_end=15,
    source_type="code",
    confidence=0.65, confidence_label="medium",
    function_name="process_payment",
    language="python",
)
evidence.code.append(code_chunk)

assert evidence.total_chunks == 2
assert evidence.has_logs is True
assert evidence.has_code is True
print("✓ Test 20 passed: StructuredEvidence properties")

# ==============================
# Test 21: Agent — Evidence Context Rendering
# ==============================
evidence.dependencies = ["src/gateway.py", "src/config.py"]
evidence.architecture_summary = "Payment processing module"
evidence.cross_analysis_context = "Log error at app.log:100 → process_payment() in payment.py:10"
evidence.trend_summary = "3 errors in last hour"
evidence.references_found = [
    {"file_path": "src/payment.py", "function_name": "process_payment", "line_number": 10}
]

context = build_evidence_context(evidence)
assert "Log Evidence" in context
assert "Code Evidence" in context
assert "Dependencies" in context
assert "Architecture Context" in context
assert "Log-to-Code Cross Analysis" in context
assert "Automated Analysis" in context
assert "Error Trends" in context
assert "Extracted Code References" in context
assert "payment.py" in context
assert "process_payment" in context
print("✓ Test 21 passed: Evidence context rendering")

# ==============================
# Test 22: Agent — StepResult Tracking
# ==============================
sr = StepResult(
    step_type="search_logs",
    description="Semantic + keyword search for log entries",
    chunks_found=5,
    ms=42.3,
    details={"logs": 3, "code": 2},
)
evidence.step_results.append(sr)
assert len(evidence.step_results) == 1
assert evidence.step_results[0].step_type == "search_logs"
assert evidence.step_results[0].ms == 42.3
print("✓ Test 22 passed: Step result tracking")

# ==============================
# Test 23: Planner — General Plan Has Both Search Types
# ==============================
plan = create_plan("Tell me about the database")
assert plan.question_type == QuestionType.GENERAL
step_types = [s.step_type for s in plan.steps]
assert StepType.SEARCH_LOGS in step_types
assert StepType.SEARCH_CODE in step_types
assert StepType.BUNDLE_CONTEXT in step_types
print("✓ Test 23 passed: General plan searches both logs and code")

# ==============================
# Test 24: Session Memory — Clear All
# ==============================
add_turn("__ns1__", "q1", "a1")
add_turn("__ns2__", "q2", "a2")
clear_all_sessions()
assert get_session("__ns1__").turn_count == 0
assert get_session("__ns2__").turn_count == 0
print("✓ Test 24 passed: Clear all sessions")

# ==============================
# Test 25: Planner — Structural Fast-Path
# ==============================
plan = create_plan("Which files import auth?")
assert plan.question_type == QuestionType.STRUCTURAL
assert plan.fast_path is True
assert len(plan.steps) == 1
assert plan.steps[0].step_type == StepType.QUERY_GRAPH
print("✓ Test 25 passed: Structural fast-path")

# ==============================
# Cleanup
# ==============================
clear_all_sessions()
print("\n" + "=" * 50)
print(f"All 25 tests passed!")
print("=" * 50)
