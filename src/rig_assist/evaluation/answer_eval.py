"""Answer eval, part 1: run the eval set through the RAG pipeline and compare "not found" rules.

`generate_answers` asks every question once and appends each outcome to a JSON Lines cache
as soon as it arrives, so a run stopped by a rate limit resumes where it stopped (records
made with another config are ignored, not mixed in). Each record keeps the top retrieval
score, so refusal rules that act *before* the LLM (a reranker-score gate) can be compared
offline with the LLM's own refusals, without asking the LLM again:

- llm: refuse when the pipeline answered "not found" (any refusal reason);
- reranker: refuse when the best reranker score is below a threshold (LLM never called);
- llm_or_reranker: refuse if either does.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.metrics import RefusalResult
from rig_assist.rag import AskResult, extract_citations


@dataclass(frozen=True)
class AnswerRecord:
    id: str
    answerable: bool
    found: bool
    refusal_reason: str | None
    answer: str
    llm_answer: str | None
    cited: list[tuple[str, int]]  # (doc_id, page) of accepted citations
    cited_expected_page: bool  # an accepted citation is on an expected page (cheap proxy)
    top_score: float | None  # best retrieval score in the context; None = nothing retrieved
    context: list[dict[str, Any]]  # chunk_id, doc_id, page, text: what the LLM saw
    input_tokens: int
    output_tokens: int
    model: str | None
    llm_s: float
    retrieval_s: float
    config: Mapping[str, str]  # model, mode, top_k...: records only count for this config


def record_from_result(
    question: EvalQuestion, result: AskResult, config: Mapping[str, str]
) -> AnswerRecord:
    cited = extract_citations(result.answer) if result.found else []
    expected = {(question.expected_doc_id, page) for page in question.expected_pages}
    return AnswerRecord(
        id=question.id,
        answerable=question.answerable,
        found=result.found,
        refusal_reason=result.refusal_reason,
        answer=result.answer,
        llm_answer=result.llm_answer,
        cited=cited,
        cited_expected_page=any(pair in expected for pair in cited),
        top_score=max((hit.score for hit in result.retrieved), default=None),
        context=[
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "page": hit.chunk.page,
                "text": hit.chunk.text,
            }
            for hit in result.retrieved
        ],
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        model=result.model,
        llm_s=result.llm_s,
        retrieval_s=result.retrieval_s,
        config=dict(config),
    )


class AnswerCache:
    """Records in a JSON Lines file, one per question, appended as they're made."""

    def __init__(self, path: Path, config: Mapping[str, str]) -> None:
        self.path = path
        self.config = dict(config)

    def load(self) -> dict[str, AnswerRecord]:
        """Records made with this config, by question id (the last one wins)."""
        if not self.path.exists():
            return {}
        records = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            if data["config"] != self.config:
                continue
            data["cited"] = [tuple(pair) for pair in data["cited"]]
            records[data["id"]] = AnswerRecord(**data)
        return records

    def append(self, record: AnswerRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def generate_answers(
    ask: Callable[[str], AskResult],
    questions: Sequence[EvalQuestion],
    cache: AnswerCache,
    progress: Callable[[str], None] = lambda line: None,
) -> list[AnswerRecord]:
    """One record per question, in question order; asks only the ones not cached yet.

    Any error (a daily quota, an outage) propagates: the answers made so far are already
    saved, and running again continues from the first unanswered question.
    """
    done = cache.load()
    todo = [q for q in questions if q.id not in done]
    new = 0
    for q in questions:
        if q.id in done:
            progress(f"{q.id} cached")
            continue
        record = record_from_result(q, ask(q.question), cache.config)
        cache.append(record)
        done[q.id] = record
        new += 1
        progress(f"{q.id} answered ({new}/{len(todo)} new)")
    return [done[q.id] for q in questions]


def _confusion(decisions: Sequence[tuple[bool, bool]]) -> RefusalResult:
    """(refused, answerable) pairs → counts with "refused" as the positive class."""
    return RefusalResult(
        true_pos=sum(refused and not answerable for refused, answerable in decisions),
        false_pos=sum(refused and answerable for refused, answerable in decisions),
        false_neg=sum(not refused and not answerable for refused, answerable in decisions),
        true_neg=sum(not refused and answerable for refused, answerable in decisions),
    )


def _below(record: AnswerRecord, threshold: float) -> bool:
    # Same rule as `refusal_metrics`: nothing retrieved is refused, equal is answered.
    return record.top_score is None or record.top_score < threshold


def refusal_strategies(
    records: Sequence[AnswerRecord], threshold: float
) -> dict[str, RefusalResult]:
    return {
        "llm": _confusion([(not r.found, r.answerable) for r in records]),
        "reranker": _confusion([(_below(r, threshold), r.answerable) for r in records]),
        "llm_or_reranker": _confusion(
            [(not r.found or _below(r, threshold), r.answerable) for r in records]
        ),
    }


def safe_threshold(records: Sequence[AnswerRecord]) -> float:
    """The highest reranker threshold that refuses no answerable question in `records`."""
    scores = [r.top_score for r in records if r.answerable and r.top_score is not None]
    if not scores:
        raise ValueError("no answerable question has a score")
    return min(scores)


@dataclass(frozen=True)
class AnswerStats:
    answerable: int
    answered: int  # answerable questions that got an answer
    cited_expected_page: int  # ... whose citations include an expected page
    unanswerable: int
    unanswerable_answered: int  # should have been "not found"
    refusal_reasons: dict[str, int]
    input_tokens: int
    output_tokens: int


def answer_stats(records: Sequence[AnswerRecord]) -> AnswerStats:
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]
    reasons = Counter(r.refusal_reason for r in records if r.refusal_reason)
    return AnswerStats(
        answerable=len(answerable),
        answered=sum(r.found for r in answerable),
        cited_expected_page=sum(r.cited_expected_page for r in answerable),
        unanswerable=len(unanswerable),
        unanswerable_answered=sum(r.found for r in unanswerable),
        refusal_reasons=dict(sorted(reasons.items())),
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
    )
