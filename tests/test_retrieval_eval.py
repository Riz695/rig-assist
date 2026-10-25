"""Retrieval eval runner: per-question results, metric summaries per subset, threshold choice.

Retrievers are fakes that return scripted pages, and the clock is fake, so every number
here is hand-computable.
"""

from collections.abc import Iterator

import pytest

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.metrics import RefusalCase
from rig_assist.evaluation.retrieval_eval import (
    QuestionResult,
    choose_threshold,
    evaluate,
    percentile,
    refusal_cases,
    summarize,
)
from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.models import ScoredChunk


def question(
    qid: str,
    doc: str | None = "manual-a",
    pages: list[int] | None = None,
    qtype: str = "factual",
    wording: str = "manual",
) -> EvalQuestion:
    answerable = qtype != "unanswerable"
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "question": f"question {qid}",
            "expected_doc_id": doc if answerable else None,
            "expected_pages": (pages or [1]) if answerable else [],
            "reference_answer": "answer",
            "evidence_quote": "quote" if answerable else None,
            "type": qtype,
            "difficulty": "easy",
            "wording": wording,
        }
    )


def hit(doc: str, page: int, dense: float | None = None) -> ScoredChunk:
    chunk = Chunk(f"{doc}::p{page}::c0", doc, doc.title(), page, None, "pdfplumber", "text")
    return ScoredChunk(chunk=chunk, score=1.0, dense_score=dense)


class ScriptedRetriever:
    """Returns a fixed result list per question text; records the k it was asked for."""

    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.ks: list[int] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.ks.append(k)
        return self.results.get(query, [])[:k]


def fake_clock(*times: float) -> Iterator[float]:
    return iter(times)


# --- evaluate ------------------------------------------------------------


def test_evaluate_records_rank_latency_and_top_dense_score() -> None:
    q1 = question("q1", pages=[7])
    retriever = ScriptedRetriever(
        {"question q1": [hit("manual-a", 3, 0.5), hit("manual-a", 7, 0.8)]}
    )
    clock = fake_clock(10.0, 10.25)

    [result] = evaluate(retriever, [q1], clock=lambda: next(clock))

    assert result == QuestionResult(id="q1", rank=2, latency_s=0.25, top_dense_score=0.8)


def test_evaluate_asks_for_ten_results_by_default() -> None:
    retriever = ScriptedRetriever({})
    evaluate(retriever, [question("q1")])
    assert retriever.ks == [10]


def test_top_dense_score_only_looks_at_the_first_five_results() -> None:
    # The LLM sees 5 chunks, so the "not found" decision uses those 5 only.
    results = [hit("manual-a", p, 0.1) for p in range(2, 7)] + [hit("manual-a", 9, 0.99)]
    retriever = ScriptedRetriever({"question q1": results})
    [result] = evaluate(retriever, [question("q1")])
    assert result.top_dense_score == 0.1


def test_top_dense_score_is_none_without_dense_scores() -> None:
    # BM25-only results carry no cosine score: refusal can't be judged for that mode.
    retriever = ScriptedRetriever({"question q1": [hit("manual-a", 1)]})
    [result] = evaluate(retriever, [question("q1")])
    assert result.top_dense_score is None


def test_unanswerable_question_has_no_rank_but_keeps_its_score() -> None:
    retriever = ScriptedRetriever({"question u1": [hit("manual-a", 1, 0.42)]})
    [result] = evaluate(retriever, [question("u1", qtype="unanswerable")])
    assert result.rank is None
    assert result.top_dense_score == 0.42


# --- percentile ----------------------------------------------------------


def test_percentile_interpolates_linearly() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8)


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([0.3], 95) == 0.3


def test_percentile_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


# --- summarize -----------------------------------------------------------


def results_for(ranks: dict[str, int | None]) -> list[QuestionResult]:
    return [QuestionResult(qid, rank, 0.1, None) for qid, rank in ranks.items()]


def test_summary_hand_computed_over_answerable_questions() -> None:
    questions = [
        question("q1"),
        question("q2"),
        question("q3"),
        question("u1", qtype="unanswerable"),
    ]
    results = results_for({"q1": 1, "q2": 4, "q3": None, "u1": None})

    overall = summarize(results, questions)["all"]

    assert overall.n == 3  # the unanswerable question doesn't count for retrieval metrics
    assert overall.hit_at_1 == pytest.approx(1 / 3)
    assert overall.hit_at_5 == pytest.approx(2 / 3)
    assert overall.mrr_at_10 == pytest.approx((1 + 1 / 4 + 0) / 3)


def test_rank_beyond_ten_counts_as_a_miss_for_mrr() -> None:
    summary = summarize(results_for({"q1": 11}), [question("q1")])["all"]
    assert summary.mrr_at_10 == 0.0


def test_latency_percentiles_use_every_question_including_unanswerable() -> None:
    questions = [question("q1"), question("u1", qtype="unanswerable")]
    results = [QuestionResult("q1", 1, 0.1, None), QuestionResult("u1", None, 0.3, None)]
    overall = summarize(results, questions)["all"]
    assert overall.p50_latency_s == pytest.approx(0.2)
    assert overall.p95_latency_s == pytest.approx(0.29)


def test_subsets_for_tables_scanned_manual_and_paraphrased_questions() -> None:
    questions = [
        question("q1", qtype="table"),
        question("q2", doc="scanned-manual", wording="paraphrased"),
        question("q3"),
    ]
    results = results_for({"q1": 1, "q2": None, "q3": 2})

    summary = summarize(results, questions, scanned_docs={"scanned-manual"})

    assert summary["table"].n == 1 and summary["table"].hit_at_1 == 1.0
    assert summary["scanned"].n == 1 and summary["scanned"].hit_at_5 == 0.0
    assert summary["paraphrased"].n == 1
    assert summary["manual_wording"].n == 2
    assert summary["all"].n == 3


def test_empty_subsets_are_left_out() -> None:
    summary = summarize(results_for({"q1": 1}), [question("q1")])
    assert "table" not in summary
    assert "scanned" not in summary


def test_summary_needs_a_result_for_every_question() -> None:
    with pytest.raises(ValueError, match="q2"):
        summarize(results_for({"q1": 1}), [question("q1"), question("q2")])


# --- refusal cases and threshold choice ----------------------------------


def test_refusal_cases_pair_scores_with_labels() -> None:
    questions = [question("q1"), question("u1", qtype="unanswerable")]
    results = [QuestionResult("q1", 1, 0.1, 0.8), QuestionResult("u1", None, 0.1, 0.4)]
    assert refusal_cases(results, questions) == [
        RefusalCase(top_score=0.8, answerable=True),
        RefusalCase(top_score=0.4, answerable=False),
    ]


def test_choose_threshold_separates_cleanly_separable_scores() -> None:
    cases = [
        RefusalCase(0.30, answerable=False),
        RefusalCase(0.40, answerable=False),
        RefusalCase(0.70, answerable=True),
        RefusalCase(0.90, answerable=True),
    ]
    threshold = choose_threshold(cases)
    # Any threshold in (0.40, 0.70] refuses exactly the two unanswerable questions;
    # the lowest one is chosen (fewest needless refusals): just above 0.40.
    assert 0.40 < threshold <= 0.70
    assert threshold == pytest.approx(0.40, abs=1e-6)


def test_choose_threshold_maximises_refusal_f1() -> None:
    cases = [
        RefusalCase(0.50, answerable=False),
        RefusalCase(0.55, answerable=True),
        RefusalCase(0.60, answerable=False),
        RefusalCase(0.80, answerable=True),
        RefusalCase(0.85, answerable=True),
    ]
    # Refusing below just-above-0.60 catches both unanswerable at the cost of one
    # answerable: P = 2/3, R = 1, F1 = 0.8; refusing only 0.50: P = 1, R = 1/2, F1 = 0.67.
    threshold = choose_threshold(cases)
    assert 0.60 < threshold <= 0.80


def test_choose_threshold_needs_scores_and_both_labels() -> None:
    with pytest.raises(ValueError):
        choose_threshold([RefusalCase(0.5, answerable=True)])
    with pytest.raises(ValueError):
        choose_threshold([RefusalCase(None, answerable=False), RefusalCase(None, answerable=True)])
