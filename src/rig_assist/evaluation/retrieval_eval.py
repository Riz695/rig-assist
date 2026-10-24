"""Run the eval set through a retriever and summarise the results.

`evaluate` asks the retriever every question and records, per question, the rank of the
first chunk on an expected page, the latency, and the best dense (cosine) score among the
top 5, the chunks an LLM would see. `summarize` turns those into Hit@1, Hit@5, MRR@10 and
latency percentiles, overall and per subset. `choose_threshold` picks the "not found"
threshold that best separates unanswerable questions, on this same set (so it's optimistic).
"""

import math
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.metrics import (
    PageRef,
    RefusalCase,
    first_relevant_rank,
    refusal_metrics,
)
from rig_assist.retrieval.models import Retriever

RESULTS_K = 10  # MRR@10 needs the top 10
CONTEXT_K = 5  # chunks handed to the LLM; Hit@5 and the "not found" decision use these


@dataclass(frozen=True)
class QuestionResult:
    id: str
    rank: int | None  # 1-based rank of the first relevant result; None = not in the top 10
    latency_s: float
    top_dense_score: float | None  # best cosine score in the top 5; None = no dense scores


def evaluate(
    retriever: Retriever,
    questions: Sequence[EvalQuestion],
    k: int = RESULTS_K,
    clock: Callable[[], float] = time.perf_counter,
) -> list[QuestionResult]:
    results = []
    for question in questions:
        start = clock()
        hits = retriever.search(question.question, k)
        latency = clock() - start
        rank = None
        if question.answerable:
            relevant = {PageRef(question.expected_doc_id or "", p) for p in question.expected_pages}
            rank = first_relevant_rank(
                [PageRef(h.chunk.doc_id, h.chunk.page) for h in hits], relevant
            )
        scores = [h.dense_score for h in hits[:CONTEXT_K] if h.dense_score is not None]
        results.append(QuestionResult(question.id, rank, latency, max(scores, default=None)))
    return results


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile with linear interpolation between ranks (NumPy's default)."""
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class Summary:
    n: int  # answerable questions in the subset
    hit_at_1: float
    hit_at_5: float
    mrr_at_10: float
    p50_latency_s: float
    p95_latency_s: float


def _summary(results: list[QuestionResult], answerable: set[str]) -> Summary:
    ranks = [r.rank for r in results if r.id in answerable]
    latencies = [r.latency_s for r in results]

    def share(hits: list[bool]) -> float:
        return sum(hits) / len(hits)

    return Summary(
        n=len(ranks),
        hit_at_1=share([rank is not None and rank <= 1 for rank in ranks]),
        hit_at_5=share([rank is not None and rank <= CONTEXT_K for rank in ranks]),
        mrr_at_10=sum(1 / r for r in ranks if r is not None and r <= RESULTS_K) / len(ranks),
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
    )


def summarize(
    results: Sequence[QuestionResult],
    questions: Sequence[EvalQuestion],
    scanned_docs: Collection[str] = (),
) -> dict[str, Summary]:
    """Metrics for "all" answerable questions and each non-empty subset.

    Subsets: "table", "scanned" (questions on a manual in `scanned_docs`, whose text only
    exists through OCR), "paraphrased" and "manual_wording". Retrieval metrics use the
    answerable questions of a subset; latency uses every question in it.
    """
    by_id = {r.id: r for r in results}
    missing = [q.id for q in questions if q.id not in by_id]
    if missing:
        raise ValueError(f"no result for: {', '.join(missing)}")
    subsets: dict[str, Callable[[EvalQuestion], bool]] = {
        "all": lambda q: True,
        "table": lambda q: q.type == "table",
        "scanned": lambda q: q.expected_doc_id in scanned_docs,
        "paraphrased": lambda q: q.answerable and q.wording == "paraphrased",
        "manual_wording": lambda q: q.answerable and q.wording == "manual",
    }
    summaries = {}
    for name, keep in subsets.items():
        chosen = [q for q in questions if keep(q)]
        answerable = {q.id for q in chosen if q.answerable}
        if answerable:
            summaries[name] = _summary([by_id[q.id] for q in chosen], answerable)
    return summaries


def refusal_cases(
    results: Sequence[QuestionResult], questions: Sequence[EvalQuestion]
) -> list[RefusalCase]:
    """One (top dense score, answerable) pair per question, in question order."""
    by_id = {r.id: r for r in results}
    return [RefusalCase(by_id[q.id].top_dense_score, q.answerable) for q in questions]


def choose_threshold(cases: Sequence[RefusalCase]) -> float:
    """The threshold with the best F1 for "not found"; on ties, the lowest one.

    Candidates sit just above each observed score (refusal means score < threshold), so
    each candidate refuses that score and everything below it. Lowest-on-ties keeps
    needless refusals of answerable questions to a minimum.
    """
    scores = sorted({c.top_score for c in cases if c.top_score is not None})
    if not scores:
        raise ValueError("no case has a score to threshold")
    if all(c.answerable for c in cases) or not any(c.answerable for c in cases):
        raise ValueError("need both answerable and unanswerable cases")

    def f1(threshold: float) -> float:
        result = refusal_metrics(cases, threshold)
        p, r = result.precision, result.recall
        return 0.0 if not p or not r else 2 * p * r / (p + r)

    candidates = [math.nextafter(score, math.inf) for score in scores]
    return max(candidates, key=lambda t: (f1(t), -t))
