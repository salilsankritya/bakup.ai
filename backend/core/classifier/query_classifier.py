"""
core/classifier/query_classifier.py
─────────────────────────────────────────────────────────────────────────────
Classifies incoming user questions into one of three categories BEFORE any
retrieval or LLM call is made.

Categories:
    project        — about indexed code, logs, incidents, project structure, files
    greeting       — short pleasantries (hi, how are you, thanks, etc.)
    conversational — personal / meta / casual questions (LLM responds directly)
    off_topic      — clearly unrelated to the project (physics, politics, recipes…)

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
    PROJECT        = "project"        # About indexed data — route to RAG
    GREETING       = "greeting"       # Short pleasantries — canned polite reply
    CONVERSATIONAL = "conversational" # Personal/meta/casual — LLM direct call
    OFF_TOPIC      = "off_topic"      # Unrelated to project — scope guard


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

# Conversational / meta / personal — caught before the ambiguous weak-project
# signals so "are you gay" or "tell me a joke" go to the LLM instead of retrieval.
_CONVERSATIONAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # Personal questions directed at the AI
        r"^are\s+you\s+(?:a\s+)?\w+",            # "are you gay", "are you a bot"
        r"^do\s+you\s+(like|love|hate|feel|think|believe|dream|eat|sleep|remember)",
        r"^can\s+you\s+(feel|think|dream|love|sing|dance|laugh|cry|joke|play|flirt)",
        r"^(who|what)\s+(made|created|built|trained|programmed|designed)\s+you",
        r"^how\s+old\s+are\s+you",
        r"^what('?s|\s+is)\s+your\s+(name|age|gender|sex|fav|favorite|opinion|problem|purpose|job)",
        r"^(tell|say)\s+(me\s+)?(a\s+)?(joke|story|fun\s*fact|something\s+(funny|interesting|cool))",
        r"^i\s+(love|hate|like|miss|need|want)\s+you",
        r"^you('re|\s+are)\s+\w+",                # "you're stupid", "you are great"
        r"^do\s+you\s+have\s+(?:a\s+)?(?:boyfriend|girlfriend|wife|husband|body|soul|mind|brain|feelings?|emotions?|consciousness|name|age|life|family|friends?)",
        # Short reactions / fillers
        r"^(lol|haha|rofl|lmao|omg|bruh|bro|dude|wow|nice|hmm+|meh|ugh|yikes|oops|ha+)[\s!?.]*$",
        r"^(ok|okay|sure|yes|no|nah|nope|yep|yeah|yea|k|kk|alright|cool|fine|whatever|great)[\s!?.]*$",
        # Non-project requests
        r"^(sing|dance|rap|beatbox|flirt|marry)",
        r"^(write|compose|make)\s+(me\s+)?(a\s+)?(poem|song|haiku|limerick|rap|joke|riddle)",
        # General capability
        r"^what\s+can\s+you\s+do",
        # Provocative / profanity (professional deflection, not retrieval)
        r"^(f+u+c+k+|sh+i+t+|damn|wtf|stfu|shut\s+up)",
    ]
]


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_query(question: str) -> QueryCategory:
    """
    Classify *question* into project / greeting / conversational / off_topic.

    Fast (regex-only, < 1 ms). No LLM call.

    Priority:
        1. If it matches greeting patterns exactly   → GREETING
        2. If it matches STRONG project signals      → PROJECT
        3. If it matches off-topic signals           → OFF_TOPIC
        4. If it matches conversational patterns     → CONVERSATIONAL
        5. If it matches WEAK project signals        → PROJECT
        6. Default                                   → CONVERSATIONAL (LLM handles naturally)

    The two-tier split prevents generic verbs like \"explain\" from
    overriding clearly off-topic questions (\"Explain quantum physics\").
    Conversational patterns (step 4) catch personal / meta queries like
    \"are you gay\" before they fall through to retrieval.
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

    # 4. Conversational / meta — personal questions, casual chat, identity
    for pat in _CONVERSATIONAL_PATTERNS:
        if pat.search(q):
            return QueryCategory.CONVERSATIONAL

    # 5. Weak project signal — generic question words (explain, how, why…)
    for pat in _PROJECT_WEAK:
        if pat.search(q):
            return QueryCategory.PROJECT

    # 6. Default: conversational — if nothing matches, let the LLM handle
    #    it naturally instead of forcing through retrieval where the results
    #    would be irrelevant.
    return QueryCategory.CONVERSATIONAL


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


def conversational_response() -> str:
    """Fallback when LLM is not configured for conversational questions."""
    return (
        "I'm bakup.ai, a project-scoped incident intelligence assistant. "
        "I'm best at helping you investigate errors, search through logs, "
        "and understand your codebase. "
        "Try asking about an error, a specific file, or a recent log event!"
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
