import json
from pathlib import Path

import pytest

from rig_assist.evaluation.answer_eval import (
    AnswerCache,
    AnswerRecord,
    answer_stats,
    generate_answers,
    record_from_result,
    refusal_strategies,
    safe_threshold,
)
from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.metrics import RefusalResult
from rig_assist.ingestion.models import Chunk
from rig_assist.llm import FakeLlm, LlmError
from rig_assist.rag import AskResult, RagPipeline
from rig_assist.retrieval.models import ScoredChunk

CONFIG = {"model": "fake", "mode": "hybrid_rerank", "top_k": "5"}


def question(qid: str, pages: list[int] | None = None, doc: str = "bop-manual") -> EvalQuestion:
    if pages is None:  # unanswerable
        return EvalQuestion(
            id=qid,
            question=f"Question {qid}?",
            expected_doc_id=None,
            expected_pages=[],
            reference_answer="Not in the manuals.",
            evidence_quote=None,
            type="unanswerable",
            difficulty="medium",
            wording="paraphrased",
        )
    return EvalQuestion(
        id=qid,
        question=f"Question {qid}?",
        expected_doc_id=doc,
        expected_pages=pages,
        reference_answer="Every 14 days.",
        evidence_quote="every 14 days",
        type="factual",
        difficulty="easy",
        wording="manual",
    )


def hit(page: int, score: float, doc: str = "bop-manual") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{doc}::p{page}::c0",
        doc_id=doc,
        doc_title="Bop Manual",
        page=page,
        section=None,
        parser="pdfplumber",
        text=f"Text of page {page}.",
    )
    return ScoredChunk(chunk=chunk, score=score)


class FixedRetriever:
    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        return self.hits[:k]


def ask_with(hits: list[ScoredChunk], reply: str) -> AskResult:
    return RagPipeline(FixedRetriever(hits), FakeLlm(replies=[reply])).ask("q")


def record(qid: str, answerable: bool, found: bool, top_score: float | None) -> AnswerRecord:
    return AnswerRecord(
        id=qid,
        answerable=answerable,
        found=found,
        refusal_reason=None if found else "llm_not_found",
        answer="x",
        llm_answer="x",
        cited=[],
        cited_expected_page=found and answerable,
        top_score=top_score,
        context=[],
        input_tokens=0,
        output_tokens=0,
        model="fake",
        llm_s=0.0,
        retrieval_s=0.0,
        config=CONFIG,
    )


# --- record_from_result -------------------------------------------------


def test_record_keeps_the_outcome_citations_and_top_score() -> None:
    result = ask_with([hit(12, 2.5), hit(13, -1.0)], "Every 14 days [bop-manual p.12].")
    rec = record_from_result(question("q001", [12]), result, CONFIG)
    assert (rec.id, rec.answerable, rec.found, rec.refusal_reason) == ("q001", True, True, None)
    assert rec.cited == [("bop-manual", 12)]
    assert rec.cited_expected_page is True
    assert rec.top_score == 2.5
    assert rec.model == "fake"
    assert rec.config == CONFIG
    assert [c["chunk_id"] for c in rec.context] == ["bop-manual::p12::c0", "bop-manual::p13::c0"]
    assert rec.context[0]["text"] == "Text of page 12."


def test_citing_only_other_pages_is_not_citing_the_expected_page() -> None:
    result = ask_with([hit(12, 2.5), hit(13, 1.0)], "Weekly [bop-manual p.13].")
    rec = record_from_result(question("q001", [12]), result, CONFIG)
    assert rec.found is True
    assert rec.cited_expected_page is False


def test_the_expected_page_must_be_in_the_expected_document() -> None:
    result = ask_with([hit(12, 2.5, doc="other-manual")], "Weekly [other-manual p.12].")
    rec = record_from_result(question("q001", [12], doc="bop-manual"), result, CONFIG)
    assert rec.cited_expected_page is False


def test_a_refused_answer_cites_nothing_and_keeps_the_raw_text() -> None:
    result = ask_with([hit(12, 2.5)], "Weekly [bop-manual p.99].")
    rec = record_from_result(question("q001", [12]), result, CONFIG)
    assert (rec.found, rec.refusal_reason) == (False, "invalid_citation")
    assert rec.cited == []
    assert rec.cited_expected_page is False
    assert rec.llm_answer == "Weekly [bop-manual p.99]."


def test_nothing_retrieved_has_no_top_score() -> None:
    result = RagPipeline(FixedRetriever([]), FakeLlm(replies=[])).ask("q")
    assert record_from_result(question("q045"), result, CONFIG).top_score is None


# --- AnswerCache --------------------------------------------------------


def test_cache_round_trips_records(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.jsonl", CONFIG)
    rec = record_from_result(
        question("q001", [12]), ask_with([hit(12, 2.5)], "Yes [bop-manual p.12]."), CONFIG
    )
    cache.append(rec)
    assert AnswerCache(tmp_path / "answers.jsonl", CONFIG).load() == {"q001": rec}


def test_cache_ignores_records_from_another_config(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    AnswerCache(path, CONFIG).append(record("q001", True, True, 1.0))
    other = AnswerCache(path, {**CONFIG, "model": "gemini-x"})
    assert other.load() == {}


def test_missing_cache_file_is_empty(tmp_path: Path) -> None:
    assert AnswerCache(tmp_path / "nope" / "answers.jsonl", CONFIG).load() == {}


def test_cache_appends_one_json_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    cache = AnswerCache(path, CONFIG)
    cache.append(record("q001", True, True, 1.0))
    cache.append(record("q002", False, False, -3.0))
    lines = path.read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["q001", "q002"]


# --- generate_answers ---------------------------------------------------


def test_generate_asks_every_question_and_saves_each(tmp_path: Path) -> None:
    asked: list[str] = []

    def ask(text: str) -> AskResult:
        asked.append(text)
        return ask_with([hit(12, 2.5)], "Yes [bop-manual p.12].")

    cache = AnswerCache(tmp_path / "answers.jsonl", CONFIG)
    records = generate_answers(ask, [question("q001", [12]), question("q043")], cache)
    assert asked == ["Question q001?", "Question q043?"]
    assert [r.id for r in records] == ["q001", "q043"]
    assert set(cache.load()) == {"q001", "q043"}


def test_generate_resumes_without_asking_saved_questions_again(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path / "answers.jsonl", CONFIG)
    saved = record("q001", True, True, 1.0)
    cache.append(saved)
    asked: list[str] = []

    def ask(text: str) -> AskResult:
        asked.append(text)
        return ask_with([hit(12, 2.5)], "Yes [bop-manual p.12].")

    records = generate_answers(ask, [question("q001", [12]), question("q002", [12])], cache)
    assert asked == ["Question q002?"]
    assert records[0] == saved
    assert [r.id for r in records] == ["q001", "q002"]


def test_a_failure_keeps_earlier_answers_and_stops(tmp_path: Path) -> None:
    calls = 0

    def ask(text: str) -> AskResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LlmError("HTTP 429: quota exceeded")
        return ask_with([hit(12, 2.5)], "Yes [bop-manual p.12].")

    cache = AnswerCache(tmp_path / "answers.jsonl", CONFIG)
    questions = [question("q001", [12]), question("q002", [12]), question("q003", [12])]
    with pytest.raises(LlmError, match="429"):
        generate_answers(ask, questions, cache)
    assert set(cache.load()) == {"q001"}
    assert calls == 2  # stopped at the failure, q003 not asked


def test_progress_is_reported_per_question(tmp_path: Path) -> None:
    lines: list[str] = []
    cache = AnswerCache(tmp_path / "answers.jsonl", CONFIG)
    cache.append(record("q001", True, True, 1.0))
    generate_answers(
        lambda _: ask_with([hit(12, 2.5)], "Yes [bop-manual p.12]."),
        [question("q001", [12]), question("q002", [12])],
        cache,
        progress=lines.append,
    )
    assert lines == ["q001 cached", "q002 answered (1/1 new)"]


# --- Refusal strategies -------------------------------------------------

RECORDS = [
    # answerable, LLM answered, strong reranker score
    record("a1", answerable=True, found=True, top_score=3.0),
    # answerable, LLM answered, weak score
    record("a2", answerable=True, found=True, top_score=-2.0),
    # answerable, LLM refused (a needless refusal)
    record("a3", answerable=True, found=False, top_score=1.0),
    # unanswerable, LLM refused correctly, middling score
    record("u1", answerable=False, found=False, top_score=0.5),
    # unanswerable, LLM answered anyway, weak score
    record("u2", answerable=False, found=True, top_score=-4.0),
    # unanswerable, nothing retrieved
    record("u3", answerable=False, found=False, top_score=None),
]


def test_llm_strategy_refuses_what_the_pipeline_refused() -> None:
    result = refusal_strategies(RECORDS, threshold=0.0)["llm"]
    # refused: a3 (FP), u1, u3 (TP); answered: a1, a2 (TN), u2 (FN)
    assert result == RefusalResult(true_pos=2, false_pos=1, false_neg=1, true_neg=2)


def test_reranker_strategy_refuses_scores_below_the_threshold() -> None:
    result = refusal_strategies(RECORDS, threshold=0.0)["reranker"]
    # below 0.0 or none: a2 (FP), u2, u3 (TP); at or above: a1, a3 (TN), u1 (FN)
    assert result == RefusalResult(true_pos=2, false_pos=1, false_neg=1, true_neg=2)


def test_combined_strategy_refuses_if_either_refuses() -> None:
    result = refusal_strategies(RECORDS, threshold=0.0)["llm_or_reranker"]
    # refused: a2, a3 (FP), u1, u2, u3 (TP); answered: a1 (TN)
    assert result == RefusalResult(true_pos=3, false_pos=2, false_neg=0, true_neg=1)


def test_safe_threshold_refuses_no_answerable_question() -> None:
    # The lowest answerable score: refusing below it never refuses an answerable question.
    assert safe_threshold(RECORDS) == -2.0
    result = refusal_strategies(RECORDS, threshold=safe_threshold(RECORDS))["reranker"]
    assert result.false_pos == 0


def test_safe_threshold_needs_scored_answerable_questions() -> None:
    with pytest.raises(ValueError):
        safe_threshold([record("u1", answerable=False, found=False, top_score=0.5)])


# --- Answer stats -------------------------------------------------------


def test_answer_stats_count_answers_expected_pages_and_reasons() -> None:
    stats = answer_stats(RECORDS)
    assert stats.answerable == 3
    assert stats.answered == 2  # a1, a2
    assert stats.cited_expected_page == 2  # record() marks answered answerables as correct
    assert stats.unanswerable == 3
    assert stats.unanswerable_answered == 1  # u2
    assert stats.refusal_reasons == {"llm_not_found": 3}
    assert stats.input_tokens == 0 and stats.output_tokens == 0


# --- Report -------------------------------------------------------------


def test_report_has_the_answer_summary_rules_and_one_row_per_question() -> None:
    from rig_assist.evaluation.run_answers import render_report

    report = render_report(RECORDS, notes=["Generated for a test."])
    assert "Generated for a test." in report
    assert "| Answerable questions answered | 2/3 |" in report
    assert "| Unanswerable questions answered anyway | 1/3 |" in report
    assert "LLM only" in report
    assert report.count("Reranker gate only") == 2  # tuned and safe thresholds
    for rec in RECORDS:
        assert f"| {rec.id} |" in report
    assert "Text of page" not in report  # no manual text in the committed report


def test_report_adds_the_judge_section_and_scores_per_question() -> None:
    from rig_assist.evaluation.judge import JudgeRecord
    from rig_assist.evaluation.run_answers import render_report

    def verdict(qid: str, answerable: bool, faith: int | None, correct: int) -> JudgeRecord:
        return JudgeRecord(
            id=qid,
            answerable=answerable,
            outcome="judged" if faith is not None else "refused",
            faithfulness=faith,
            correctness=correct,
            faithfulness_reason="",
            correctness_reason="",
            raw_reply=None,
            attempts=1 if faith is not None else 0,
            input_tokens=0,
            output_tokens=0,
            config={},
        )

    verdicts = [
        verdict("a1", True, 2, 2),
        verdict("a2", True, 1, 1),
        verdict("a3", True, None, 0),
        verdict("u1", False, None, 2),
        verdict("u2", False, 2, 0),
        verdict("u3", False, None, 2),
    ]
    report = render_report(RECORDS, verdicts=verdicts)
    assert "## LLM judge" in report
    assert "| Correctness, answerable questions (refused = 0) | 1 | 1 | 1 |" in report
    assert "| Faithfulness, answers judged | 2 | 1 | 0 |" in report
    assert "| a1 | yes | yes |  | 3.00 | yes | 2 | 2 |" in report
    assert "| a3 | yes | no | llm_not_found | 1.00 | no | – | 0 |" in report


def test_report_without_verdicts_has_no_judge_section() -> None:
    from rig_assist.evaluation.run_answers import render_report

    assert "## LLM judge" not in render_report(RECORDS)
