import json
from pathlib import Path

import pytest

from rig_assist.evaluation.answer_eval import AnswerRecord
from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.evaluation.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    JudgeCache,
    JudgeRecord,
    VerdictError,
    judge_answer,
    judge_answers,
    judge_messages,
    judge_stats,
    parse_verdict,
)
from rig_assist.llm import FakeLlm, LlmError

CONFIG = {"judge_model": "fake", "judge_prompt_sha256": "abc"}


def question(qid: str, answerable: bool = True) -> EvalQuestion:
    if not answerable:
        return EvalQuestion(
            id=qid,
            question=f"Torque for bolt {qid}?",
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
        question=f"How often is the BOP tested ({qid})?",
        expected_doc_id="bop-manual",
        expected_pages=[12],
        reference_answer="Every 14 days.",
        evidence_quote="every 14 days",
        type="factual",
        difficulty="easy",
        wording="manual",
    )


def answer(qid: str, answerable: bool = True, found: bool = True) -> AnswerRecord:
    text = "Every 14 days [bop-manual p.12]." if found else "Not found in the manuals."
    return AnswerRecord(
        id=qid,
        answerable=answerable,
        found=found,
        refusal_reason=None if found else "llm_not_found",
        answer=text,
        llm_answer=text,
        cited=[("bop-manual", 12)] if found else [],
        cited_expected_page=found and answerable,
        top_score=2.0,
        context=[
            {
                "chunk_id": "bop-manual::p12::c0",
                "doc_id": "bop-manual",
                "page": 12,
                "text": "Test the BOP every 14 days.",
            },
            {
                "chunk_id": "bop-manual::p13::c0",
                "doc_id": "bop-manual",
                "page": 13,
                "text": "Record the test in the log.",
            },
        ],
        input_tokens=100,
        output_tokens=10,
        model="fake",
        llm_s=1.0,
        retrieval_s=1.0,
        config={"llm_model": "fake"},
    )


def verdict_json(faithfulness: object = 2, correctness: object = 2, **extra: object) -> str:
    return json.dumps(
        {
            "faithfulness_reason": "Every claim is on p.12.",
            "faithfulness": faithfulness,
            "correctness_reason": "Matches the reference.",
            "correctness": correctness,
            **extra,
        }
    )


# --- Prompt ---------------------------------------------------------------


def test_prompt_defines_both_scales_and_the_json_reply() -> None:
    for word in ("faithfulness", "correctness", "JSON", "0", "1", "2"):
        assert word in JUDGE_SYSTEM_PROMPT


def test_messages_hold_question_reference_labelled_context_and_answer() -> None:
    messages = judge_messages(question("q001"), answer("q001"))
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == JUDGE_SYSTEM_PROMPT
    user = messages[1].content
    assert "How often is the BOP tested (q001)?" in user
    assert "Every 14 days." in user  # reference answer
    assert "[bop-manual p.12]\nTest the BOP every 14 days." in user
    assert "[bop-manual p.13]\nRecord the test in the log." in user
    assert user.rstrip().endswith("Every 14 days [bop-manual p.12].")  # the answer, last


# --- parse_verdict --------------------------------------------------------


def test_parses_a_valid_verdict() -> None:
    verdict = parse_verdict(verdict_json(faithfulness=1, correctness=0))
    assert (verdict.faithfulness, verdict.correctness) == (1, 0)
    assert verdict.faithfulness_reason == "Every claim is on p.12."
    assert verdict.correctness_reason == "Matches the reference."


def test_accepts_a_json_code_fence_around_the_object() -> None:
    assert parse_verdict(f"```json\n{verdict_json()}\n```").correctness == 2


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is correct.",  # not JSON
        "[1, 2]",  # JSON, not an object
        verdict_json(faithfulness=3),  # out of range
        verdict_json(correctness=-1),
        verdict_json(correctness="2"),  # a string, not a number
        verdict_json(correctness=True),  # bool is not a score
        verdict_json(correctness=1.5),
        verdict_json(confidence=0.9),  # unknown field
        json.dumps({"faithfulness": 2, "correctness": 2}),  # reasons missing
        verdict_json().replace("Matches the reference.", " "),  # blank reason
    ],
)
def test_rejects_invalid_verdicts(reply: str) -> None:
    with pytest.raises(VerdictError):
        parse_verdict(reply)


# --- judge_answer ---------------------------------------------------------


def test_judges_an_answer_with_one_call() -> None:
    llm = FakeLlm(replies=[verdict_json(faithfulness=2, correctness=1)])
    rec = judge_answer(llm, question("q001"), answer("q001"), CONFIG)
    assert (rec.id, rec.answerable, rec.outcome) == ("q001", True, "judged")
    assert (rec.faithfulness, rec.correctness) == (2, 1)
    assert rec.correctness_reason == "Matches the reference."
    assert rec.attempts == 1
    assert rec.config == CONFIG
    assert len(llm.calls) == 1
    assert llm.max_tokens == [JUDGE_MAX_TOKENS]
    assert rec.input_tokens > 0 and rec.output_tokens > 0


def test_retries_once_after_an_invalid_reply_and_says_what_was_wrong() -> None:
    llm = FakeLlm(replies=["Looks right to me.", verdict_json()])
    rec = judge_answer(llm, question("q001"), answer("q001"), CONFIG)
    assert rec.outcome == "judged"
    assert rec.attempts == 2
    retry = llm.calls[1]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == "Looks right to me."
    assert "JSON" in retry[3].content


def test_two_invalid_replies_are_recorded_as_unparseable_not_dropped() -> None:
    llm = FakeLlm(replies=["nope", verdict_json(correctness=7)])
    rec = judge_answer(llm, question("q001"), answer("q001"), CONFIG)
    assert rec.outcome == "unparseable"
    assert rec.faithfulness is None and rec.correctness is None
    assert rec.attempts == 2
    assert rec.raw_reply == verdict_json(correctness=7)  # kept for debugging
    assert len(llm.calls) == 2  # no third try


def test_usage_adds_up_over_the_retry() -> None:
    llm = FakeLlm(replies=["nope", verdict_json()])
    rec = judge_answer(llm, question("q001"), answer("q001"), CONFIG)
    single = judge_answer(
        FakeLlm(replies=[verdict_json()]), question("q001"), answer("q001"), CONFIG
    )
    assert rec.input_tokens > single.input_tokens
    assert rec.output_tokens == 1 + single.output_tokens  # "nope" is one word


def test_answerable_question_refused_scores_wrong_without_calling_the_judge() -> None:
    llm = FakeLlm(replies=[])
    rec = judge_answer(llm, question("q022"), answer("q022", found=False), CONFIG)
    assert rec.outcome == "refused"
    assert rec.correctness == 0
    assert rec.faithfulness is None  # nothing was claimed
    assert rec.attempts == 0 and llm.calls == []


def test_unanswerable_question_refused_scores_right_without_calling_the_judge() -> None:
    llm = FakeLlm(replies=[])
    q, a = question("q044", answerable=False), answer("q044", answerable=False, found=False)
    rec = judge_answer(llm, q, a, CONFIG)
    assert rec.outcome == "correct_refusal"
    assert rec.correctness == 2
    assert rec.faithfulness is None
    assert llm.calls == []


def test_unanswerable_question_answered_is_judged() -> None:
    # e.g. q043: a refusal in other words, which only the judge can recognise.
    llm = FakeLlm(replies=[verdict_json(correctness=2)])
    q, a = question("q043", answerable=False), answer("q043", answerable=False, found=True)
    rec = judge_answer(llm, q, a, CONFIG)
    assert rec.outcome == "judged"
    assert rec.answerable is False
    assert "Not in the manuals." in llm.calls[0][1].content


def test_judge_failures_propagate() -> None:
    with pytest.raises(LlmError):
        judge_answer(FakeLlm(replies=[]), question("q001"), answer("q001"), CONFIG)


# --- JudgeCache and judge_answers ----------------------------------------


def test_cache_round_trips_and_ignores_other_configs(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    rec = judge_answer(FakeLlm(replies=[verdict_json()]), question("q001"), answer("q001"), CONFIG)
    JudgeCache(path, CONFIG).append(rec)
    assert JudgeCache(path, CONFIG).load() == {"q001": rec}
    assert JudgeCache(path, {**CONFIG, "judge_prompt_sha256": "def"}).load() == {}
    assert JudgeCache(tmp_path / "none.jsonl", CONFIG).load() == {}


def test_judges_every_answer_in_question_order_and_resumes(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "judge.jsonl", CONFIG)
    questions = [question("q001"), question("q002"), question("q044", answerable=False)]
    answers = {
        "q001": answer("q001"),
        "q002": answer("q002"),
        "q044": answer("q044", answerable=False, found=False),
    }
    cache.append(
        judge_answer(FakeLlm(replies=[verdict_json()]), questions[0], answers["q001"], CONFIG)
    )
    llm = FakeLlm(replies=[verdict_json(correctness=1)])
    lines: list[str] = []
    records = judge_answers(llm, questions, answers, cache, progress=lines.append)
    assert [r.id for r in records] == ["q001", "q002", "q044"]
    assert len(llm.calls) == 1  # q001 cached, q044 a refusal: only q002 asked
    assert records[1].correctness == 1
    assert set(cache.load()) == {"q001", "q002", "q044"}
    assert lines[0] == "q001 cached"


def test_a_judge_failure_keeps_earlier_verdicts(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "judge.jsonl", CONFIG)
    questions = [question("q001"), question("q002")]
    answers = {"q001": answer("q001"), "q002": answer("q002")}
    with pytest.raises(LlmError):
        judge_answers(FakeLlm(replies=[verdict_json()]), questions, answers, cache)
    assert set(cache.load()) == {"q001"}


def test_every_question_needs_a_saved_answer(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "judge.jsonl", CONFIG)
    with pytest.raises(KeyError, match="q002"):
        judge_answers(FakeLlm(), [question("q002")], {}, cache)


# --- judge_stats ----------------------------------------------------------


def judged(
    qid: str, answerable: bool, outcome: str, faith: int | None, correct: int | None
) -> JudgeRecord:
    return JudgeRecord(
        id=qid,
        answerable=answerable,
        outcome=outcome,  # type: ignore[arg-type]
        faithfulness=faith,
        correctness=correct,
        faithfulness_reason="",
        correctness_reason="",
        raw_reply=None,
        attempts=0 if outcome in ("refused", "correct_refusal") else 1,
        input_tokens=10,
        output_tokens=5,
        config=CONFIG,
    )


def test_stats_count_scores_per_scale() -> None:
    stats = judge_stats(
        [
            judged("a1", True, "judged", 2, 2),
            judged("a2", True, "judged", 1, 1),
            judged("a3", True, "judged", 2, 0),
            judged("a4", True, "refused", None, 0),
            judged("a5", True, "unparseable", None, None),
            judged("u1", False, "correct_refusal", None, 2),
            judged("u2", False, "judged", 2, 2),
        ]
    )
    # Correctness over answerable questions (refusals count as 0, unparseable apart).
    assert stats.answerable_correctness == {2: 1, 1: 1, 0: 2}
    assert stats.unanswerable_correctness == {2: 2, 1: 0, 0: 0}
    # Faithfulness over every judged answer (refusals claim nothing).
    assert stats.faithfulness == {2: 3, 1: 1, 0: 0}
    assert stats.unparseable == 1
    assert stats.judge_calls == 5
    assert (stats.input_tokens, stats.output_tokens) == (70, 35)
