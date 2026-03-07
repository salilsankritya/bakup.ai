"""
core/router/router.py
─────────────────────────────────────────────────────────────────────────────
Hybrid query router for bakup.ai.

Decides how to route every incoming user query:
    • project_query   → planner + agent retrieval pipeline
    • conversational  → conversational response (LLM or canned)
    • unrelated       → polite scope message

Two classification strategies:
    1. **LLM-based** — when a provider is configured, a small classification
       prompt is sent to the LLM for high-accuracy intent detection.
    2. **Rule-based** — keyword + regex heuristics (always available, < 1 ms).

If the LLM classifier returns low confidence (< CONFIDENCE_THRESHOLD) or
fails for any reason, the router seamlessly falls back to rule-based
classification.  This guarantees correct behaviour in offline / local-only
environments.

Public API:
    route_query(query, namespace, session_context) → RoutingDecision
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("bakup.router")

# ── Routing result ─────────────────────────────────────────────────────────────

VALID_INTENTS = frozenset({"project_query", "conversational", "unrelated"})

CONFIDENCE_THRESHOLD = 0.6   # Below this → fall back to project_query pipeline


@dataclass
class RoutingDecision:
    """Result of the hybrid router."""
    intent:     str            # "project_query" | "conversational" | "unrelated"
    confidence: float          # 0.0 – 1.0
    source:     str            # "llm" | "rules"
    latency_ms: float = 0.0   # wall-clock time taken by the router


# ── Keyword / pattern banks (rule-based) ───────────────────────────────────────

_PROJECT_KEYWORDS: list[str] = [
    "error", "exception", "stack", "log", "trace", "function", "module",
    "file", "service", "crash", "dependency", "architecture", "bug",
    "issue", "timeout", "latency", "slow", "oom", "memory", "leak",
    "null", "undefined", "segfault", "panic", "deadlock", "deploy",
    "docker", "container", "kubernetes", "database", "migration",
    "config", "endpoint", "route", "api", "test", "coverage",
    "git", "commit", "branch", "import", "class", "method",
    "variable", "server", "warning", "critical", "fatal",
    "optimize", "refactor", "review", "code", "codebase",
    "performance", "security", "scalab", "incident",
]

_PROJECT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\.(py|js|ts|java|go|rs|rb|cpp|c|h|yaml|yml|json|toml|sql)\b",
        r"line\s+\d+",
        r"5[0-9]{2}\b|4[0-9]{2}\b",               # HTTP 4xx / 5xx
        r"(src|lib|app|core|api|routes?|models?)/",  # path references
        r"which\s+(file|class|function|module|service|endpoint)",
        r"list\s+(all|the|files|classes|functions|errors|logs)",
        r"(what|why|how)\s+.{0,40}(error|bug|crash|fail|timeout)",
    ]
]

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

_CONVERSATIONAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^are\s+you\s+(?:a\s+)?\w+",
        r"^do\s+you\s+(like|love|hate|feel|think|believe|dream|eat|sleep|remember)",
        r"^can\s+you\s+(feel|think|dream|love|sing|dance|laugh|cry|joke|play|flirt)",
        r"^(who|what)\s+(made|created|built|trained|programmed|designed)\s+you",
        r"^how\s+old\s+are\s+you",
        r"^what('?s|\s+is)\s+your\s+(name|age|gender|sex|fav|favorite|opinion|problem|purpose|job)",
        r"^(tell|say)\s+(me\s+)?(a\s+)?(joke|story|fun\s*fact|something\s+(funny|interesting|cool))",
        r"^i\s+(love|hate|like|miss|need|want)\s+you",
        r"^you('re|\s+are)\s+\w+",
        r"^do\s+you\s+have\s+(?:a\s+)?(?:boyfriend|girlfriend|wife|husband|body|soul|mind|brain|feelings?|emotions?|consciousness|name|age|life|family|friends?)",
        r"^(lol|haha|rofl|lmao|omg|bruh|bro|dude|wow|nice|hmm+|meh|ugh|yikes|oops|ha+)[\s!?.]*$",
        r"^(ok|okay|sure|yes|no|nah|nope|yep|yeah|yea|k|kk|alright|cool|fine|whatever|great)[\s!?.]*$",
        r"^(sing|dance|rap|beatbox|flirt|marry)",
        r"^(write|compose|make)\s+(me\s+)?(a\s+)?(poem|song|haiku|limerick|rap|joke|riddle)",
        r"^what\s+can\s+you\s+do",
        r"^(f+u+c+k+|sh+i+t+|damn|wtf|stfu|shut\s+up)",
    ]
]

_UNRELATED_PATTERNS: list[re.Pattern] = [
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


# ── LLM classification prompt ────────────────────────────────────────────────

_LLM_CLASSIFY_PROMPT = """\
You are an intent classifier for an AI project assistant called bakup.ai.

Classify the following user query into exactly one of these categories:

- project_query  — about code, logs, debugging, errors, architecture, files, incidents, deployments, project structure, testing, performance, security
- conversational — greetings, small talk, personal questions aimed at the AI, jokes, casual chat
- unrelated      — clearly not about a software project (recipes, weather, politics, sports, physics, math, etc.)

Return ONLY valid JSON with no markdown formatting, no code fences, no extra text:
{{"intent": "<category>", "confidence": <float 0-1>}}

Query:
"{user_query}"
"""


# ── Rule-based classifier ────────────────────────────────────────────────────

def _classify_rules(query: str) -> RoutingDecision:
    """
    Fast rule-based intent classification (< 1 ms, no network).

    Priority order:
        1. Greeting patterns  → conversational (high confidence)
        2. Conversational patterns → conversational
        3. Unrelated patterns → unrelated
        4. Project keyword count → project_query (confidence from keyword density)
        5. Project regex patterns → project_query
        6. Default → project_query (safe fallback — retrieval can handle edge cases)
    """
    q = query.strip()
    if not q:
        return RoutingDecision(intent="project_query", confidence=0.5, source="rules")

    q_lower = q.lower()

    # 1. Greetings (near-exact short match)
    if len(q) < 60:
        for pat in _GREETING_PATTERNS:
            if pat.search(q):
                return RoutingDecision(intent="conversational", confidence=0.95, source="rules")

    # 2. Conversational / meta / personal
    for pat in _CONVERSATIONAL_PATTERNS:
        if pat.search(q):
            return RoutingDecision(intent="conversational", confidence=0.90, source="rules")

    # 3. Clearly unrelated topics
    for pat in _UNRELATED_PATTERNS:
        if pat.search(q):
            return RoutingDecision(intent="unrelated", confidence=0.85, source="rules")

    # 4. Project keywords — count matches for confidence scaling
    words = set(re.findall(r"[a-zA-Z_]+", q_lower))
    kw_hits = sum(1 for kw in _PROJECT_KEYWORDS if kw in q_lower or kw in words)

    if kw_hits >= 3:
        conf = min(0.95, 0.60 + kw_hits * 0.07)
        return RoutingDecision(intent="project_query", confidence=round(conf, 2), source="rules")

    if kw_hits >= 1:
        conf = min(0.80, 0.50 + kw_hits * 0.10)
        return RoutingDecision(intent="project_query", confidence=round(conf, 2), source="rules")

    # 5. Project regex patterns
    for pat in _PROJECT_PATTERNS:
        if pat.search(q):
            return RoutingDecision(intent="project_query", confidence=0.70, source="rules")

    # 6. Default → project_query (retrieval will handle unknown queries gracefully)
    return RoutingDecision(intent="project_query", confidence=0.50, source="rules")


# ── LLM-based classifier ─────────────────────────────────────────────────────

def _classify_llm(query: str) -> Optional[RoutingDecision]:
    """
    Use the configured LLM to classify intent.

    Returns None if the LLM is not configured, call fails, or response
    cannot be parsed — caller should fall back to rules.
    """
    try:
        from core.llm.config_store import load_config
        from core.llm.llm_service import get_llm_service

        cfg = load_config()
        if not cfg.configured:
            return None

        svc = get_llm_service()
        prompt = _LLM_CLASSIFY_PROMPT.format(user_query=query.replace('"', '\\"'))

        raw = svc._call_provider(cfg, prompt, system_prompt="You are an intent classifier. Reply with JSON only.")
        raw = raw.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        intent = parsed.get("intent", "").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))

        if intent not in VALID_INTENTS:
            logger.warning("LLM returned unknown intent %r — falling back to rules", intent)
            return None

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return RoutingDecision(intent=intent, confidence=confidence, source="llm")

    except Exception as exc:
        logger.warning("LLM classification failed (%s) — falling back to rules", exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def route_query(
    query: str,
    namespace: str = "",
    session_context: str = "",
) -> RoutingDecision:
    """
    Classify a user query into an intent using the hybrid router.

    Strategy:
        1. Try LLM classification (if provider is configured).
        2. If LLM succeeds with confidence ≥ CONFIDENCE_THRESHOLD → use it.
        3. Otherwise fall back to rule-based classification.

    Fallback protection:
        If either classifier returns confidence < CONFIDENCE_THRESHOLD,
        the intent is forced to "project_query" to avoid blocking valid
        project questions.

    Args:
        query:           The user's raw question text.
        namespace:       Project namespace (used for context, not classification).
        session_context: Prior conversation context (reserved for future use).

    Returns:
        RoutingDecision with intent, confidence, source, and latency_ms.
    """
    t0 = time.perf_counter()

    # Step 1: Try LLM classification
    llm_result = _classify_llm(query)

    if llm_result is not None and llm_result.confidence >= CONFIDENCE_THRESHOLD:
        llm_result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "Router: LLM → %s (%.0f%%) in %.1fms",
            llm_result.intent, llm_result.confidence * 100, llm_result.latency_ms,
        )
        return llm_result

    # Step 2: Fall back to rules
    rule_result = _classify_rules(query)
    rule_result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # If LLM returned a result but below threshold, log it
    if llm_result is not None:
        logger.info(
            "Router: LLM confidence too low (%.0f%%) — using rules → %s (%.0f%%)",
            llm_result.confidence * 100, rule_result.intent, rule_result.confidence * 100,
        )

    # Fallback protection: if rule confidence is also below threshold,
    # force project_query to avoid blocking valid queries
    if rule_result.confidence < CONFIDENCE_THRESHOLD and rule_result.intent != "project_query":
        logger.info(
            "Router: rule confidence too low (%.0f%%) — forcing project_query",
            rule_result.confidence * 100,
        )
        rule_result.intent = "project_query"

    logger.info(
        "Router: rules → %s (%.0f%%) in %.1fms",
        rule_result.intent, rule_result.confidence * 100, rule_result.latency_ms,
    )

    return rule_result


# ── Scope message for unrelated queries ──────────────────────────────────────

UNRELATED_SCOPE_MESSAGE = (
    "I'm bakup.ai, a project support assistant focused on analyzing code, "
    "logs, and system issues. That question isn't related to the current project."
)


def get_scope_message() -> str:
    """Return the canonical scope-guard message for unrelated queries."""
    return UNRELATED_SCOPE_MESSAGE
