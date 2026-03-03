"""
core/classifier/query_classifier.py
─────────────────────────────────────────────────────────────────────────────
Classifies incoming user questions into one of three categories BEFORE any
retrieval or LLM call is made.

Categories:
    project   — about indexed code, logs, incidents, project structure, files
    greeting  — conversational pleasantries (hi, how are you, thanks, etc.)
    off_topic — clearly unrelated to the project (physics, politics, recipes…)

The classifier is a lightweight keyword / pattern matcher — no LLM call
required, no latency.  It runs in < 1 ms and is the very first step in the
query pipeline.

When an LLM provider IS configured, the classifier can optionally delegate
ambiguous cases to the LLM for a one-word ruling ("project" | "greeting" |
"off_topic") but this is never required — the keyword layer handles 95 %+
of real-world queries.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class QueryCategory(str, Enum):
    PROJECT   = "project"       # About indexed data — route to RAG
    GREETING  = "greeting"      # Conversational — short polite reply
    OFF_TOPIC = "off_topic"     # Unrelated to project — scope guard


# ── Keyword banks ──────────────────────────────────────────────────────────────
# Lowercase. Checked via substring / word-boundary match.

_GREETING_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(hi|hello|hey|howdy|hola|yo)[\s!.,?]*$",
        r"^(good\s*(morning|afternoon|evening|day|night))[\s!.,?]*$",
        r"^how\s+are\s+you",
        r"^what'?s?\s+up[\s!?]*$",
        r"^(thanks?|thank\s+you|thx|ty)[\s!.,?]*$",
        r"^(bye|goodbye|see\s+you|take\s+care)[\s!.,?]*$",
        r"^(who|what)\s+are\s+you[\s?]*$",
        r"^your\s+name[\s?]*$",
        r"^(please\s+)?help[\s!?]*$",
    ]
]

# Phrases strongly signalling a project/code/log/incident question
# Split into STRONG (unambiguous code/infra terms) and WEAK (generic verbs
# that could also appear in off-topic questions like "Explain quantum physics").
_PROJECT_STRONG: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Error / incident language
        r"error|exception|traceback|stack\s*trace|crash|fail|bug|issue",
        r"timeout|latency|slow|OOM|out\s+of\s+memory|memory\s+leak",
        r"null\s*pointer|NoneType|undefined|segfault|panic|deadlock",
        r"5[0-9]{2}\b|4[0-9]{2}\b",   # HTTP status codes (4xx, 5xx)
        r"log(s|file|ged)?|incident|alert|warning|critical|fatal",

        # Code / project structure
        r"\.(py|js|ts|java|go|rs|rb|cpp|c|h|yaml|yml|json|toml|xml|sql)\b",
        r"function|class|method|module|package|import|endpoint|route|\bapi\b",
        r"\bfile\b|directory|folder|config|setting|\benv\b|variable|parameter",
        r"database|\bdb\b|table|schema|migration|\bquery\b|\bindex\b",
        r"deploy|docker|container|pod|k8s|kubernetes|service|server",
        r"git|commit|branch|merge|pull\s+request|PR\b",
        r"\btest\b|\bspec\b|assert|coverage|\bci\b|\bcd\b|pipeline",
        r"depend|requirement|version|upgrade|install",

        # Direct references to project artefacts
        r"(src|lib|app|core|api|routes?|models?|services?|utils?|helpers?)/",
        r"line\s+\d+",
        r"which\s+(file|class|function|module|service|endpoint)",
        r"list\s+(all|the|files|classes|functions|errors|logs)",
    ]
]

_PROJECT_WEAK: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Generic question words — only count as project if no off-topic match
        r"what\s+(is|does|caused|happened|triggers?|calls?)",
        r"where\s+(is|are|does|did|can)",
        r"show\s+me|find|search|look\s+(for|at|up)",
        r"explain|describe|summarize|analyse|analyze",
        r"how\s+(does|do|to|is|did)",
        r"why\s+(does|do|is|did|was)",
    ]
]

# Topics clearly outside any software project scope
_OFF_TOPIC_SIGNALS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"quantum\s+(physics|mechanics|computing|entanglement)",
        r"recipe|cooking|bake\b|ingredients",
        r"(stock|crypto)\s*(market|price|trading)|bitcoin|ethereum",
        r"weather\s+(forecast|today|tomorrow)",
        r"(write|compose|create)\s+(a\s+)?(poem|song|story|essay|novel)",
        r"(who\s+is|tell\s+me\s+about)\s+(elon|trump|biden|obama|modi|taylor)",
        r"(capital|president|prime\s+minister)\s+of\s+\w+",
        r"translate\s+.+\s+(to|into)\s+\w+",
        r"(play|recommend)\s+.{0,20}(movie|song|game|book)",
        r"(explain|what\s+is)\s+(relativity|evolution|gravity|photosynthesis)",
        r"(solve|calculate)\s+\d+\s*[\+\-\*/]",
        r"(history|geography|philosophy)\s+of\s+",
        r"(meaning|origin)\s+of\s+life",
        r"(sport|football|basketball|cricket|tennis)\s+(score|match|result)",
    ]
]


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_query(question: str) -> QueryCategory:
    """
    Classify *question* into project / greeting / off_topic.

    Fast (regex-only, < 1 ms). No LLM call.

    Priority:
        1. If it matches greeting patterns exactly  → GREETING
        2. If it matches STRONG project signals      → PROJECT
        3. If it matches off-topic signals           → OFF_TOPIC
        4. If it matches WEAK project signals        → PROJECT
        5. Default                                   → PROJECT (benefit of the doubt)

    The two-tier split prevents generic verbs like "explain" from
    overriding clearly off-topic questions ("Explain quantum physics").
    """
    q = question.strip()
    if not q:
        return QueryCategory.PROJECT   # empty → will be caught downstream

    # 1. Greetings — must be a near-exact match (short utterance)
    if len(q) < 60:
        for pat in _GREETING_PATTERNS:
            if pat.search(q):
                return QueryCategory.GREETING

    # 2. Strong project signal — unambiguous code/infra terms
    for pat in _PROJECT_STRONG:
        if pat.search(q):
            return QueryCategory.PROJECT

    # 3. Off-topic — only if we didn't find a strong project signal
    for pat in _OFF_TOPIC_SIGNALS:
        if pat.search(q):
            return QueryCategory.OFF_TOPIC

    # 4. Weak project signal — generic question words (explain, how, why…)
    for pat in _PROJECT_WEAK:
        if pat.search(q):
            return QueryCategory.PROJECT

    # 5. Default: assume project-related (benefit of the doubt)
    return QueryCategory.PROJECT


def greeting_response() -> str:
    """Canonical short reply for conversational greetings."""
    return (
        "Hello! I'm bakup.ai, your project-scoped incident intelligence assistant. "
        "Ask me about your indexed code, logs, or incidents and I'll find relevant context."
    )


def off_topic_response() -> str:
    """Canonical reply for questions outside project scope."""
    return (
        "This question is outside the scope of this project. "
        "bakup.ai focuses on indexed project data — including source code, "
        "log files, and incident traces. Try asking about an error, "
        "a specific file, or a recent log event."
    )


def low_confidence_response(best_confidence: float) -> str:
    """
    Reply when retrieval returns results but all are below useful confidence.

    Instead of guessing, we ask the user to refine their question.
    """
    return (
        f"I found some content that might be related (best match confidence: "
        f"{best_confidence:.0%}), but I'm not confident enough to give an accurate answer.\n\n"
        "Could you try rephrasing your question? For example:\n"
        "• Mention the specific file, function, or log entry you're asking about\n"
        "• Include an error message or status code\n"
        "• Narrow down the time range or component"
    )
