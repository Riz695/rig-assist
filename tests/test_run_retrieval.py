"""Eval orchestration: one tracked run per config, warm-up, refusal metrics, Markdown report.

A fake tracker records what would go to MLflow, so these tests need neither MLflow nor
real models.
"""

from typing import Any

import pytest

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.run_retrieval import (
    V1_THRESHOLD,
    ConfigResult,
    EvalConfig,
    render_report,
    run_configs,
)
from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.models import ScoredChunk


def question(qid: str, page: int = 1, answerable: bool = True) -> EvalQuestion:
    return EvalQuestion.model_validate(
        {
            "id": qid,
            "question": f"question {qid}",
            "expected_doc_id": "manual-a" if answerable else None,
            "expected_pages": [page] if answerable else [],
            "reference_answer": "answer",
            "evidence_quote": "quote" if answerable else None,
            "type": "factual" if answerable else "unanswerable",
            "difficulty": "easy",
            "wording": "manual",
        }
    )


QUESTIONS = [question("q1", page=1), question("q2", page=2), question("u1", answerable=False)]


def hit(page: int, dense: float | None) -> ScoredChunk:
    chunk = Chunk(f"manual-a::p{page}::c0", "manual-a", "Manual A", page, None, "pdfplumber", "t")
    return ScoredChunk(chunk, score=1.0, dense_score=dense)


class ScriptedRetriever:
    def __init__(self, results: dict[str, list[ScoredChunk]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.results.get(query, [])[:k]


# q1 found at rank 1, q2 at rank 2; the unanswerable question scores lowest.
DENSE = ScriptedRetriever(
    {
        "question q1": [hit(1, 0.9)],
        "question q2": [hit(5, 0.7), hit(2, 0.65)],
        "question u1": [hit(3, 0.5)],
    }
)
BM25 = ScriptedRetriever(
    {"question q1": [hit(4, None), hit(1, None)], "question q2": [hit(9, None), hit(2, None)]}
)


class FakeTracker:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifacts: dict[str, str],
    ) -> None:
        self.runs.append(
            {"name": name, "params": params, "metrics": metrics, "artifacts": artifacts}
        )


def run(tracker: FakeTracker | None = None) -> list[ConfigResult]:
    configs = [
        EvalConfig("dense", ScriptedRetriever(DENSE.results), {"mode": "dense"}),
        EvalConfig("bm25", ScriptedRetriever(BM25.results), {"mode": "bm25"}),
    ]
    return run_configs(
        configs, QUESTIONS, tracker or FakeTracker(), common_params={"eval_set_sha256": "abc"}
    )


def test_one_tracked_run_per_config_with_common_and_own_params() -> None:
    tracker = FakeTracker()
    run(tracker)
    assert [r["name"] for r in tracker.runs] == ["dense", "bm25"]
    assert tracker.runs[0]["params"] == {"eval_set_sha256": "abc", "mode": "dense"}


def test_metrics_are_logged_per_subset() -> None:
    tracker = FakeTracker()
    run(tracker)
    metrics = tracker.runs[0]["metrics"]
    assert metrics["all/hit_at_1"] == pytest.approx(0.5)
    assert metrics["all/hit_at_5"] == 1.0
    assert metrics["all/mrr_at_10"] == pytest.approx(0.75)
    assert metrics["all/n"] == 2
    assert "all/p95_latency_s" in metrics


def test_per_question_results_are_logged_as_an_artifact() -> None:
    tracker = FakeTracker()
    run(tracker)
    artifact = tracker.runs[0]["artifacts"]["per_question.json"]
    assert '"q2"' in artifact and '"rank": 2' in artifact


def test_a_warm_up_query_runs_first_and_is_not_scored() -> None:
    retriever = ScriptedRetriever(DENSE.results)
    run_configs([EvalConfig("dense", retriever, {})], QUESTIONS, FakeTracker())
    assert len(retriever.queries) == len(QUESTIONS) + 1
    assert retriever.queries[0] not in {q.question for q in QUESTIONS}


def test_refusal_at_v1_threshold_and_at_the_tuned_threshold() -> None:
    [dense, _] = run()
    assert dense.refusal_v1 is not None and dense.refusal_tuned is not None
    # At 0.6 only u1 (0.5) is refused: P = 1, R = 1.
    assert (dense.refusal_v1.precision, dense.refusal_v1.recall) == (1.0, 1.0)
    assert V1_THRESHOLD == 0.6
    assert 0.5 < dense.tuned_threshold <= 0.65  # type: ignore[operator]


def test_refusal_is_skipped_for_a_retriever_without_dense_scores() -> None:
    tracker = FakeTracker()
    [_, bm25] = run(tracker)
    assert bm25.refusal_v1 is None and bm25.tuned_threshold is None
    assert not any(key.startswith("refusal") for key in tracker.runs[1]["metrics"])


def test_report_has_one_row_per_config_with_counts() -> None:
    report = render_report(run(), title="Retrieval eval")
    assert "| dense | 1/2 (0.50) | 2/2 (1.00) | 0.750 |" in report
    assert "| bm25 | 0/2 (0.00) | 2/2 (1.00) | 0.500 |" in report


def test_report_shows_refusal_only_where_it_applies() -> None:
    report = render_report(run(), title="Retrieval eval")
    refusal_section = report.split("## Refusal")[1]
    assert "| dense |" in refusal_section
    assert "| bm25 |" not in refusal_section


def test_report_compares_parsers_mode_by_mode() -> None:
    configs = [
        EvalConfig(
            "dense", ScriptedRetriever(DENSE.results), {"mode": "dense", "parser": "pdfplumber"}
        ),
        EvalConfig(
            "olmocr_all_dense",
            ScriptedRetriever(BM25.results),
            {"mode": "dense", "parser": "olmocr"},
        ),
    ]
    report = render_report(run_configs(configs, QUESTIONS, FakeTracker()), title="Retrieval eval")
    section = report.split("## Parser comparison")[1].split("## Refusal")[0]
    assert "| dense | pdfplumber | 2/2 (1.00) | 0.750 | n/a | n/a | n/a | n/a |" in section
    assert "| dense | olmocr | 2/2 (1.00) | 0.500 | n/a | n/a | n/a | n/a |" in section


def test_no_parser_comparison_without_both_parsers() -> None:
    assert "## Parser comparison" not in render_report(run(), title="Retrieval eval")
