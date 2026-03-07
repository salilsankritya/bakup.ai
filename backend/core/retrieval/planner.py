"""
core/retrieval/planner.py
─────────────────────────────────────────────────────────────────────────────
Agentic retrieval planner — classifies questions and generates multi-step
retrieval plans.

Instead of a single embed→retrieve→answer pass, the planner analyses the
question and creates an ordered list of retrieval steps that build on each
other's evidence.

Example plan for "Why is payment failing?":
    1. search_logs    → find log entries mentioning payment errors
    2. extract_refs   → pull file/function/class references from log text
    3. retrieve_code  → fetch the actual source code referenced
    4. get_deps       → look up dependencies of the failing code
    5. get_arch       → fetch architecture context for the module
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ── Question types ─────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    """Categorisation of the incoming question for plan selection."""
    LOG_ANALYSIS    = "log_analysis"       # errors, exceptions, crash reports
    CODE_ANALYSIS   = "code_analysis"      # how does X work, explain Y
    CODE_REVIEW     = "code_review"        # is it optimized, review quality
    ROOT_CAUSE      = "root_cause"         # why did X fail, correlate log + code
    ARCHITECTURE    = "architecture"       # project structure, overview
    STRUCTURAL      = "structural"         # which files import X, what depends on Y
    GENERAL         = "general"            # broad or ambiguous project question


# ── Plan step types ────────────────────────────────────────────────────────────

class StepType(str, Enum):
    """Individual retrieval operation types."""
    SEARCH_LOGS     = "search_logs"        # semantic + keyword search in log chunks
    SEARCH_CODE     = "search_code"        # semantic search in code chunks
    EXTRACT_REFS    = "extract_refs"       # extract file/fn/class refs from evidence
    RETRIEVE_CODE   = "retrieve_code"      # fetch code chunks by identifiers
    GET_DEPS        = "get_deps"           # look up dependencies via symbol graph
    GET_ARCH        = "get_arch"           # fetch architecture summary
    QUERY_GRAPH     = "query_graph"        # answer from symbol graph directly
    CROSS_ANALYSIS  = "cross_analysis"     # link logs to code via log_code_linker
    BUNDLE_CONTEXT  = "bundle_context"     # group related code chunks with siblings


@dataclass
class PlanStep:
    """A single step in the retrieval plan."""
    step_type: StepType
    description: str
    depends_on: Optional[str] = None  # Label of a prior step whose output feeds in


@dataclass
class RetrievalPlan:
    """
    An ordered retrieval plan created by the planner.

    The agent executor will run these steps sequentially, feeding evidence
    from earlier steps into later ones.
    """
    question_type: QuestionType
    steps: List[PlanStep]
    reasoning: str            # Why this plan was chosen
    fast_path: bool = False   # True when no multi-step retrieval needed


# ── Detection helpers ──────────────────────────────────────────────────────────

_LOG_KEYWORDS = {
    "error", "exception", "traceback", "failed", "failure", "crash",
    "warning", "warn", "fatal", "critical", "timeout", "panic",
    "null", "none", "undefined", "oom", "stacktrace", "stack trace",
    "segfault", "deadlock", "bug", "issue",
}

_ROOT_CAUSE_PATTERNS = [
    r"\bwhy\b.*\b(?:fail|error|crash|broken|wrong|exception)",
    r"\broot\s*cause\b",
    r"\bwhat\s+(?:caused|causes)\b",
    r"\bwhy\s+(?:is|are|did|does|was)\b.*\b(?:failing|crashing|broken|erroring)",
    r"\bdiagnos[ei]\b",
    r"\binvestigat[ei]\b",
    r"\bcorrelat[ei]\b.*\b(?:log|error|code)",
    r"\btrace\b.*\b(?:back|through|issue|error)",
    r"\bfix\b.*\b(?:error|bug|issue|crash|failure)",
]

_ARCHITECTURE_PATTERNS = [
    r"\barchitecture\b", r"\bproject\s+structure\b", r"\boverview\b",
    r"\bhow\s+is.*\borganized\b", r"\bexplain\s+the\s+project\b",
    r"\bdirectory\s+structure\b", r"\bmodules?\b.*\bproject\b",
    r"\bproject\s+layout\b", r"\bentry\s+points?\b",
    r"\btell\s+me\s+about\s+(?:the\s+)?(?:project|codebase)\b",
    r"\bwhat\s+does\s+(?:the\s+)?(?:project|codebase)\s+do\b",
]

_STRUCTURAL_PATTERNS = [
    r"which\s+files?\s+(?:use|import|depend on|reference)",
    r"what\s+depends?\s+on",
    r"(?:what|which)\s+methods?\s+(?:does|has)",
    r"(?:what|which)\s+(?:functions?|classes?|symbols?)\s+(?:are\s+)?in",
]

_CODE_PATTERNS = [
    r"\bhow\s+does\b",
    r"\bexplain\b.*\b(?:code|function|class|method|implementation)\b",
    r"\bwhat\s+does\s+\w+\s+do\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:code|implementation|function|class)\b",
    r"\bwhere\s+is\s+\w+\s+(?:defined|implemented|declared)\b",
    r"\bhow\s+is\s+\w+\s+(?:implemented|defined|used)\b",
]

_CODE_REVIEW_PATTERNS = [
    r"\boptimiz",
    r"\brefactor",
    r"\b(code|codebase)\s*(quality|review|smell|health)",
    r"\breview\s+(the|this|my)\s+(code|codebase|project)",
    r"\b(is|are)\s+(the|this|it)\s+(code|codebase|project)\s+(good|bad|clean|messy|optimized|well)",
    r"\b(best\s+practice|anti.?pattern|technical\s+debt)",
    r"\b(performance|security|scalab|maintainab|readab)\w*\s+(issue|problem|concern|improve)",
    r"\bimprov\w*\s+(the|this|my|any)\s+(code|codebase|project)",
    r"\b(any|suggest|give)\s+improve",
    r"\bcode\s+(is|looks?)\s+(good|bad|ok|fine|terrible)",
    r"\bhow\s+can\s+i\s+(improve|optimize|make.*better)",
    r"\bwhat.*wrong\s+with\s+(the|this|my)\s+(code|codebase)",
]


def _matches(text: str, patterns: list) -> bool:
    return any(re.search(p, text) for p in patterns)


# ── Planner ────────────────────────────────────────────────────────────────────

def classify_question(question: str) -> QuestionType:
    """
    Determine the question type for plan selection.

    Priority order matters — root-cause trumps log/code because it needs
    both pipelines.
    """
    q = question.lower()

    # 1. Root-cause questions (highest priority — they subsume log + code)
    if _matches(q, _ROOT_CAUSE_PATTERNS):
        return QuestionType.ROOT_CAUSE

    # 2. Architecture
    if _matches(q, _ARCHITECTURE_PATTERNS):
        return QuestionType.ARCHITECTURE

    # 3. Structural (symbol graph)
    if _matches(q, _STRUCTURAL_PATTERNS):
        return QuestionType.STRUCTURAL

    # 4. Log analysis
    if any(kw in q for kw in _LOG_KEYWORDS) or re.search(r"\blog(s|file)?\b", q):
        return QuestionType.LOG_ANALYSIS

    # 5. Code review / quality / optimization
    if _matches(q, _CODE_REVIEW_PATTERNS):
        return QuestionType.CODE_REVIEW

    # 6. Code analysis
    if _matches(q, _CODE_PATTERNS):
        return QuestionType.CODE_ANALYSIS

    # 6. General (fallback)
    return QuestionType.GENERAL


def create_plan(question: str, question_type: Optional[QuestionType] = None) -> RetrievalPlan:
    """
    Generate a multi-step retrieval plan based on the question type.

    Each plan describes the steps the agent executor should run, in order.
    Steps can declare dependencies on prior steps to form an evidence chain.
    """
    if question_type is None:
        question_type = classify_question(question)

    if question_type == QuestionType.ARCHITECTURE:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.GET_ARCH, "Fetch cached architecture summary"),
            ],
            reasoning="Architecture query → serve from cached summary",
            fast_path=True,
        )

    if question_type == QuestionType.STRUCTURAL:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.QUERY_GRAPH, "Query symbol graph for structural answer"),
            ],
            reasoning="Structural question → answer from symbol graph without LLM",
            fast_path=True,
        )

    if question_type == QuestionType.LOG_ANALYSIS:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.SEARCH_LOGS, "Semantic + keyword search for log entries"),
                PlanStep(StepType.EXTRACT_REFS, "Extract code references from log text",
                         depends_on="search_logs"),
                PlanStep(StepType.CROSS_ANALYSIS, "Link log errors to source code",
                         depends_on="extract_refs"),
            ],
            reasoning="Log query → search logs, extract references, cross-analyse with code",
        )

    if question_type == QuestionType.ROOT_CAUSE:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.SEARCH_LOGS, "Search for error/failure log entries"),
                PlanStep(StepType.EXTRACT_REFS, "Extract file/function/class refs from errors",
                         depends_on="search_logs"),
                PlanStep(StepType.RETRIEVE_CODE, "Fetch source code for referenced identifiers",
                         depends_on="extract_refs"),
                PlanStep(StepType.GET_DEPS, "Look up dependencies of failing code",
                         depends_on="retrieve_code"),
                PlanStep(StepType.GET_ARCH, "Fetch architecture context for the module"),
                PlanStep(StepType.CROSS_ANALYSIS, "Correlate log errors with code + deps",
                         depends_on="retrieve_code"),
            ],
            reasoning="Root-cause question → full pipeline: logs → refs → code → deps → architecture → cross-analysis",
        )

    if question_type == QuestionType.CODE_ANALYSIS:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.SEARCH_CODE, "Semantic search for relevant code"),
                PlanStep(StepType.BUNDLE_CONTEXT, "Bundle with siblings and imports",
                         depends_on="search_code"),
                PlanStep(StepType.GET_DEPS, "Fetch dependency chain for key results",
                         depends_on="search_code"),
            ],
            reasoning="Code question → search code, bundle context, fetch dependencies",
        )

    if question_type == QuestionType.CODE_REVIEW:
        return RetrievalPlan(
            question_type=question_type,
            steps=[
                PlanStep(StepType.SEARCH_CODE, "Semantic search for representative code"),
                PlanStep(StepType.BUNDLE_CONTEXT, "Bundle with siblings and imports",
                         depends_on="search_code"),
                PlanStep(StepType.GET_ARCH, "Fetch architecture context for review"),
                PlanStep(StepType.GET_DEPS, "Check dependency patterns",
                         depends_on="search_code"),
            ],
            reasoning="Code review question → gather broad code samples + arch context for quality analysis",
        )

    # GENERAL (fallback — search everything, bundle if code)
    return RetrievalPlan(
        question_type=question_type,
        steps=[
            PlanStep(StepType.SEARCH_LOGS, "Semantic + keyword search for log entries"),
            PlanStep(StepType.SEARCH_CODE, "Semantic search for relevant code"),
            PlanStep(StepType.BUNDLE_CONTEXT, "Bundle code context if available",
                     depends_on="search_code"),
        ],
        reasoning="General question → search both logs and code, bundle code context",
    )
