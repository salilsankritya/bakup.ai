"""
core/retrieval/session.py
─────────────────────────────────────────────────────────────────────────────
Session memory for the agentic retrieval system.

Stores recent conversation context per namespace so follow-up questions
can reuse prior retrieval results and reasoning. This enables:

  - "What errors did you find?" → retrieves from indexed data
  - "Which file has the most?" → reuses prior error analysis context

Design:
  - In-memory storage (no persistence across restarts)
  - Namespace-scoped (each project has its own conversation)
  - Fixed-size window (oldest entries evicted when limit reached)
  - Thread-safe via simple dict operations (GIL-protected)

Usage:
    from core.retrieval.session import get_session, add_turn

    session = get_session("my-project")
    add_turn("my-project", question, answer, sources, evidence_summary)
    context = session.format_context()  # inject into LLM prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Turn data ──────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single Q&A exchange stored in session memory."""
    question: str
    answer_summary: str          # First ~300 chars of the answer
    source_files: List[str]      # Files referenced in the answer
    question_type: str           # From planner classification
    evidence_summary: str        # Brief description of evidence gathered
    timestamp: float = field(default_factory=time.time)


# ── Session store ──────────────────────────────────────────────────────────────

@dataclass
class Session:
    """
    Conversation session for a namespace.

    Stores recent turns and provides formatted context for the LLM.
    """
    namespace: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 10
    created_at: float = field(default_factory=time.time)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn, evicting the oldest if at capacity."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def recent_files(self) -> List[str]:
        """Files mentioned in recent turns (deduplicated, most recent first)."""
        seen = set()
        files = []
        for turn in reversed(self.turns):
            for f in turn.source_files:
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        return files[:20]

    @property
    def recent_question_types(self) -> List[str]:
        """Question types from recent turns."""
        return [t.question_type for t in self.turns[-5:]]

    def format_context(self, max_turns: int = 3) -> str:
        """
        Format recent conversation turns for injection into the LLM prompt.

        Returns a concise summary of the last N turns so the LLM can
        understand follow-up context without replaying the full history.
        """
        if not self.turns:
            return ""

        recent = self.turns[-max_turns:]
        parts = ["## Prior Conversation Context\n"]

        for i, turn in enumerate(recent, 1):
            parts.append(
                f"**Q{i}**: {turn.question}\n"
                f"**A{i}** (summary): {turn.answer_summary}\n"
                f"Files: {', '.join(turn.source_files[:5]) if turn.source_files else 'none'}\n"
                f"Type: {turn.question_type}\n"
            )

        return "\n".join(parts)

    def is_follow_up(self, question: str) -> bool:
        """
        Heuristically detect if the question is a follow-up that needs
        prior conversation context.

        Follow-up indicators:
          - Pronouns without antecedents: "it", "that", "those", "them"
          - Relative references: "the same", "the other", "which one"
          - Short questions: fewer than 5 words (e.g., "why?", "show more")
          - Continuation words: "also", "and what about", "more details"
        """
        if not self.turns:
            return False

        q = question.lower().strip()
        words = q.split()

        # Very short question → likely follow-up
        if len(words) <= 4:
            return True

        # Pronoun-heavy without specific nouns
        follow_up_markers = [
            "it", "that", "those", "them", "this", "these",
            "the same", "the other", "which one", "which file",
            "also", "and what about", "more details", "more about",
            "tell me more", "elaborate", "explain further",
            "what else", "anything else", "any other",
            "the error", "the issue", "the bug",  # vague references
        ]
        for marker in follow_up_markers:
            if marker in q:
                return True

        return False

    def get_prior_context_for_follow_up(self) -> dict:
        """
        Get context from the most recent turn for follow-up enrichment.

        Returns a dict with source_files, question_type, and evidence_summary
        from the last turn, which can be used to scope the next retrieval.
        """
        if not self.turns:
            return {}

        last = self.turns[-1]
        return {
            "prior_question": last.question,
            "prior_type": last.question_type,
            "prior_files": last.source_files,
            "prior_evidence": last.evidence_summary,
        }

    def clear(self) -> None:
        """Clear all turns."""
        self.turns.clear()


# ── Global session store ──────────────────────────────────────────────────────

_sessions: Dict[str, Session] = {}


def get_session(namespace: str) -> Session:
    """Get or create a session for a namespace."""
    if namespace not in _sessions:
        _sessions[namespace] = Session(namespace=namespace)
    return _sessions[namespace]


def add_turn(
    namespace: str,
    question: str,
    answer: str,
    source_files: Optional[List[str]] = None,
    question_type: str = "general",
    evidence_summary: str = "",
) -> None:
    """
    Record a conversation turn in the session for a namespace.

    Args:
        namespace:        Project namespace.
        question:         The user's question.
        answer:           The generated answer (truncated to 300 chars for storage).
        source_files:     Files referenced in the answer.
        question_type:    From planner classification.
        evidence_summary: Brief summary of evidence gathered.
    """
    session = get_session(namespace)
    turn = ConversationTurn(
        question=question,
        answer_summary=answer[:300],
        source_files=source_files or [],
        question_type=question_type,
        evidence_summary=evidence_summary,
    )
    session.add_turn(turn)


def get_session_info(namespace: str) -> dict:
    """Return session metadata for debug/diagnostic endpoints."""
    session = get_session(namespace)
    return {
        "namespace": namespace,
        "turn_count": session.turn_count,
        "max_turns": session.max_turns,
        "recent_files": session.recent_files,
        "recent_types": session.recent_question_types,
        "created_at": session.created_at,
        "is_empty": session.turn_count == 0,
        "turns": [
            {
                "question": t.question,
                "answer_summary": t.answer_summary,
                "source_files": t.source_files,
                "question_type": t.question_type,
                "timestamp": t.timestamp,
            }
            for t in session.turns
        ],
    }


def clear_session(namespace: str) -> None:
    """Clear session memory for a namespace."""
    if namespace in _sessions:
        _sessions[namespace].clear()


def clear_all_sessions() -> None:
    """Clear all session memory."""
    _sessions.clear()
