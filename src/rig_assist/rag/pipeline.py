"""Single-agent RAG: retrieve, answer from the chunks with one LLM call, check citations.

The baseline the Phase 6 agent graph is compared against. Every way an answer can fail
ends in the same NOT_FOUND reply, with a `refusal_reason` saying why:

- no_results: retrieval returned nothing (the LLM isn't called);
- not_relevant: the relevance gate rejected the chunks (the LLM isn't called);
- llm_not_found: the LLM said the context doesn't answer the question;
- no_citations: the answer cites nothing, so nothing in it can be checked;
- invalid_citation: the answer cites a page that wasn't retrieved (an invented source).

Rejecting the whole answer on one bad citation is strict on purpose (as in v1): in a
maintenance manual, an uncheckable number is worse than no answer.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from rig_assist.llm import LlmClient, Prices, Usage, cost_usd
from rig_assist.rag.citations import extract_citations
from rig_assist.rag.prompt import NOT_FOUND, build_messages
from rig_assist.retrieval.models import Retriever, ScoredChunk

SNIPPET_CHARS = 300
# Output cap for one answer, including a model's hidden thinking: Gemini 3.5 Flash thought
# 600-900 tokens before a ~200-token answer, so 1,024 cut answers off (finish_reason=length).
MAX_ANSWER_TOKENS = 4096

RefusalReason = Literal[
    "no_results", "not_relevant", "llm_not_found", "no_citations", "invalid_citation"
]
RelevanceGate = Callable[[list[ScoredChunk]], bool]


@dataclass(frozen=True)
class Citation:
    doc_id: str
    doc_title: str
    page: int
    snippet: str  # start of the best-ranked retrieved chunk on that page


@dataclass(frozen=True)
class AskResult:
    answer: str
    found: bool
    citations: list[Citation]
    refusal_reason: RefusalReason | None
    llm_answer: str | None  # the LLM's raw text, kept even when it was rejected
    usage: Usage
    cost_usd: float | None  # None = prices not configured
    llm_calls: int
    model: str | None  # the model that answered; None if the LLM wasn't called
    retrieval_s: float
    llm_s: float
    total_s: float
    retrieved: list[ScoredChunk] = field(default_factory=list)  # the context, for the eval


def make_snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1] + "…"


def _is_not_found(answer: str) -> bool:
    # Only a reply that *starts* with it: an answer may cite one fact and say another
    # detail isn't in the manuals, and that's still an answer.
    return answer.strip().lower().startswith(NOT_FOUND.lower().rstrip("."))


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LlmClient,
        top_k: int = 5,
        is_relevant: RelevanceGate | None = None,  # None = let the LLM decide
        prices: Prices | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.is_relevant = is_relevant
        self.prices = prices
        self.clock = clock

    def ask(self, question: str) -> AskResult:
        start = self.clock()
        hits = self.retriever.search(question, self.top_k)
        retrieved_at = self.clock()
        retrieval_s = retrieved_at - start

        def refuse(reason: RefusalReason) -> AskResult:
            return AskResult(
                answer=NOT_FOUND,
                found=False,
                citations=[],
                refusal_reason=reason,
                llm_answer=None,
                usage=Usage(),
                cost_usd=cost_usd(Usage(), self.prices),
                llm_calls=0,
                model=None,
                retrieval_s=retrieval_s,
                llm_s=0.0,
                total_s=retrieval_s,
                retrieved=hits,
            )

        if not hits:
            return refuse("no_results")
        if self.is_relevant is not None and not self.is_relevant(hits):
            return refuse("not_relevant")

        response = self.llm.complete(build_messages(question, hits), MAX_ANSWER_TOKENS)
        answered_at = self.clock()

        reason: RefusalReason | None = None
        cited = extract_citations(response.text)
        # The best-ranked chunk per page: hits are in rank order, so keep the first seen.
        by_page: dict[tuple[str, int], ScoredChunk] = {}
        for hit in hits:
            by_page.setdefault((hit.chunk.doc_id, hit.chunk.page), hit)
        if _is_not_found(response.text):
            reason = "llm_not_found"
        elif not cited:
            reason = "no_citations"
        elif any(pair not in by_page for pair in cited):
            reason = "invalid_citation"

        citations = [
            Citation(
                doc_id=doc_id,
                doc_title=by_page[(doc_id, page)].chunk.doc_title,
                page=page,
                snippet=make_snippet(by_page[(doc_id, page)].chunk.text),
            )
            for doc_id, page in (cited if reason is None else [])
        ]
        return AskResult(
            answer=NOT_FOUND if reason else response.text,
            found=reason is None,
            citations=citations,
            refusal_reason=reason,
            llm_answer=response.text,
            usage=response.usage,
            cost_usd=cost_usd(response.usage, self.prices),
            llm_calls=1,
            model=response.model,
            retrieval_s=retrieval_s,
            llm_s=answered_at - retrieved_at,
            total_s=answered_at - start,
            retrieved=hits,
        )
