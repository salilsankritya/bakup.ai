"""
core/brain — LLM-orchestrated project intelligence.

The brain module turns bakup.ai's retrieval, analysis, and planning
subsystems into *tools* that a configured LLM can invoke on demand.
When no LLM is configured the system falls back to the existing
deterministic pipeline in core/retrieval/rag.py.
"""
