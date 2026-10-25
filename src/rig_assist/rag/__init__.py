"""Single-agent RAG baseline: one retrieval, one LLM call, checked citations."""

from rig_assist.rag.citations import extract_citations
from rig_assist.rag.pipeline import (
    MAX_ANSWER_TOKENS,
    SNIPPET_CHARS,
    AskResult,
    Citation,
    RagPipeline,
    RefusalReason,
    RelevanceGate,
)
from rig_assist.rag.prompt import NOT_FOUND, SYSTEM_PROMPT, build_messages, citation_label

__all__ = [
    "MAX_ANSWER_TOKENS",
    "NOT_FOUND",
    "SNIPPET_CHARS",
    "SYSTEM_PROMPT",
    "AskResult",
    "Citation",
    "RagPipeline",
    "RefusalReason",
    "RelevanceGate",
    "build_messages",
    "citation_label",
    "extract_citations",
]
