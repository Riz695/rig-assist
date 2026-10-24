"""Retrieval metrics (Hit@k, reciprocal rank) and refusal ("not found") metrics.

Ground truth is a set of (doc_id, page): a retrieved chunk is relevant when it sits on
an expected page. That makes the metrics independent of the parser and the chunker, so
pipelines that cut the same manual into different chunks are compared fairly.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple


class PageRef(NamedTuple):
    """One page of one manual. Page numbers are 1-based PDF positions."""

    doc_id: str
    page: int


def first_relevant_rank(retrieved: Sequence[PageRef], relevant: set[PageRef]) -> int | None:
    """1-based rank of the first retrieved page that is relevant; None if there is none.

    Several chunks can come from one page; only the first counts, which is what a reader
    scanning the results top-down would see.
    """
    if not relevant:
        raise ValueError("relevant must not be empty (unanswerable questions use refusal_metrics)")
    for rank, page in enumerate(retrieved, start=1):
        if page in relevant:
            return rank
    return None


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def hit_at_k(retrieved: Sequence[PageRef], relevant: set[PageRef], k: int) -> float:
    """1.0 if a relevant page is in the top k, else 0.0. Its mean over questions is Hit@k."""
    _check_k(k)
    rank = first_relevant_rank(retrieved, relevant)
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank_at_k(retrieved: Sequence[PageRef], relevant: set[PageRef], k: int) -> float:
    """1 / rank of the first relevant page, or 0.0 if it isn't in the top k.

    Its mean over questions is MRR@k. Unlike Hit@k, it rewards rank 1 over rank 5.
    """
    _check_k(k)
    rank = first_relevant_rank(retrieved, relevant)
    return 1 / rank if rank is not None and rank <= k else 0.0


@dataclass(frozen=True)
class RefusalCase:
    """One question's best retrieval score (None = nothing retrieved) and its label."""

    top_score: float | None
    answerable: bool


@dataclass(frozen=True)
class RefusalResult:
    """Confusion counts with "refused" as the positive class."""

    true_pos: int  # refused an unanswerable question
    false_pos: int  # refused an answerable question
    false_neg: int  # answered an unanswerable question
    true_neg: int  # answered an answerable question

    @property
    def precision(self) -> float | None:
        """Share of refusals that were right; None when nothing was refused (0/0)."""
        refused = self.true_pos + self.false_pos
        return self.true_pos / refused if refused else None

    @property
    def recall(self) -> float | None:
        """Share of unanswerable questions refused; None when there are none (0/0)."""
        unanswerable = self.true_pos + self.false_neg
        return self.true_pos / unanswerable if unanswerable else None


def refusal_metrics(cases: Sequence[RefusalCase], threshold: float) -> RefusalResult:
    """Refuse when the top score is below `threshold` (or nothing was retrieved).

    A score equal to the threshold is answered, matching v1's `dense_score >= 0.6` filter.
    """
    if not cases:
        raise ValueError("need at least one case")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for case in cases:
        refused = case.top_score is None or case.top_score < threshold
        if refused:
            counts["fp" if case.answerable else "tp"] += 1
        else:
            counts["tn" if case.answerable else "fn"] += 1
    return RefusalResult(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
