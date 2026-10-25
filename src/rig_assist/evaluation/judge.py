"""Answer eval, part 2: an LLM judge scores each saved answer for faithfulness and correctness.

- Faithfulness: is every claim in the answer supported by the context the RAG LLM saw?
- Correctness: does the answer give the eval set's reference answer?

Each is 0, 1 or 2 with a one-sentence reason, returned as JSON and validated strictly.
An invalid reply gets one retry that says what was wrong; a second invalid reply is
recorded as "unparseable" (counted in the report, not dropped). Refusals are scored by
rule, without calling the judge: an answerable question refused is wrong (0), an
unanswerable one refused is right (2). An unanswerable question that got an answer is
judged, since the answer may be a refusal in other words.

Verdicts go to a JSON Lines cache keyed by the judge config (model, prompt hash), so a
stopped run resumes and a changed prompt never mixes with old verdicts.
"""

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from rig_assist.evaluation.answer_eval import AnswerRecord
from rig_assist.evaluation.eval_set import EvalQuestion, NonEmptyStr
from rig_assist.llm import LlmClient, Message, Usage

# The judge prompt is part of the method: changing it changes `judge_prompt_sha256` in the
# run config, so old verdicts are not reused and MLflow shows which prompt scored a run.
JUDGE_SYSTEM_PROMPT = """\
You grade answers from an assistant that answers questions about offshore drilling \
equipment manuals. You get the question, a reference answer written by an expert, the \
manual excerpts the assistant was given (its context), and the assistant's answer. \
Grade two things separately.

Faithfulness: is every claim in the answer supported by the context? Judge only against \
the context, not against your own knowledge or the reference answer. Citation labels \
such as [doc-id p.12] are not claims.
- 2: every claim is supported by the context.
- 1: mostly supported, but at least one minor detail is not in the context.
- 0: a key claim is not in the context, or contradicts it.

Correctness: does the answer give the information in the reference answer? Judge only \
against the reference answer.
- 2: it contains the key facts of the reference answer and nothing that contradicts it. \
Extra correct detail is fine.
- 1: partly: some key facts are missing or imprecise, but nothing contradicts the reference.
- 0: it misses the key facts, or contradicts the reference.
If the reference answer says the information is not in the manuals, an answer that says \
the manuals do not give it (in any words) scores 2, and an answer that supplies the \
information anyway scores 0.

Reply with only this JSON object, no other text:
{"faithfulness_reason": "<one sentence>", "faithfulness": <0, 1 or 2>, \
"correctness_reason": "<one sentence>", "correctness": <0, 1 or 2>}"""

RETRY_PROMPT = (
    "That reply was not valid: {error}. Reply again with only the JSON object described "
    "above, with integer scores 0, 1 or 2 and a one-sentence reason for each."
)
# Room for hidden reasoning tokens (thinking models count them as output) plus the JSON.
JUDGE_MAX_TOKENS = 4096

Score = Annotated[StrictInt, Field(ge=0, le=2)]  # StrictInt: "2", 2.0 and true are rejected
Outcome = Literal["judged", "refused", "correct_refusal", "unparseable"]
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness_reason: NonEmptyStr
    faithfulness: Score
    correctness_reason: NonEmptyStr
    correctness: Score


class VerdictError(ValueError):
    """The judge's reply is not a valid verdict."""


def judge_messages(question: EvalQuestion, record: AnswerRecord) -> list[Message]:
    context = "\n\n".join(f"[{c['doc_id']} p.{c['page']}]\n{c['text']}" for c in record.context)
    user = (
        f"Question:\n{question.question}\n\n"
        f"Reference answer:\n{question.reference_answer}\n\n"
        f"Context:\n{context or '(nothing retrieved)'}\n\n"
        f"Answer to grade:\n{record.answer}"
    )
    return [Message("system", JUDGE_SYSTEM_PROMPT), Message("user", user)]


def parse_verdict(text: str) -> Verdict:
    """The reply as a Verdict; a ```json fence around the object is tolerated."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    try:
        data = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError as error:
        raise VerdictError(f"not JSON ({error.msg})") from error
    if not isinstance(data, dict):
        raise VerdictError("not a JSON object")
    try:
        return Verdict.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise VerdictError(problems) from error


@dataclass(frozen=True)
class JudgeRecord:
    id: str
    answerable: bool
    outcome: Outcome
    faithfulness: int | None  # None: refused (nothing claimed) or unparseable
    correctness: int | None  # None: unparseable
    faithfulness_reason: str
    correctness_reason: str
    raw_reply: str | None  # the last reply when unparseable, for debugging
    attempts: int  # judge calls made: 0 for refusals, 1, or 2 after a retry
    input_tokens: int
    output_tokens: int
    config: Mapping[str, str]  # judge model, prompt hash...: records only count for this config


def _by_rule(
    record: AnswerRecord, outcome: Outcome, correctness: int, reason: str, config: Mapping[str, str]
) -> JudgeRecord:
    return JudgeRecord(
        id=record.id,
        answerable=record.answerable,
        outcome=outcome,
        faithfulness=None,
        correctness=correctness,
        faithfulness_reason="",
        correctness_reason=reason,
        raw_reply=None,
        attempts=0,
        input_tokens=0,
        output_tokens=0,
        config=dict(config),
    )


def judge_answer(
    llm: LlmClient, question: EvalQuestion, record: AnswerRecord, config: Mapping[str, str]
) -> JudgeRecord:
    """Score one saved answer. LLM errors propagate (the caller's cache keeps earlier ones)."""
    if not record.found:
        if record.answerable:
            return _by_rule(record, "refused", 0, "Refused an answerable question.", config)
        return _by_rule(record, "correct_refusal", 2, "Refused an unanswerable question.", config)

    messages = judge_messages(question, record)
    usage = Usage()
    for attempt in (1, 2):
        response = llm.complete(messages, max_tokens=JUDGE_MAX_TOKENS)
        usage += response.usage
        try:
            verdict = parse_verdict(response.text)
        except VerdictError as error:
            messages = [
                *messages,
                Message("assistant", response.text),
                Message("user", RETRY_PROMPT.format(error=error)),
            ]
            continue
        return JudgeRecord(
            id=record.id,
            answerable=record.answerable,
            outcome="judged",
            faithfulness=verdict.faithfulness,
            correctness=verdict.correctness,
            faithfulness_reason=verdict.faithfulness_reason,
            correctness_reason=verdict.correctness_reason,
            raw_reply=None,
            attempts=attempt,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            config=dict(config),
        )
    return JudgeRecord(
        id=record.id,
        answerable=record.answerable,
        outcome="unparseable",
        faithfulness=None,
        correctness=None,
        faithfulness_reason="",
        correctness_reason="",
        raw_reply=response.text,
        attempts=2,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        config=dict(config),
    )


class JudgeCache:
    """Verdicts in a JSON Lines file, one per question, appended as they're made."""

    def __init__(self, path: Path, config: Mapping[str, str]) -> None:
        self.path = path
        self.config = dict(config)

    def load(self) -> dict[str, JudgeRecord]:
        """Verdicts made with this config, by question id (the last one wins)."""
        if not self.path.exists():
            return {}
        records = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            if data["config"] == self.config:
                records[data["id"]] = JudgeRecord(**data)
        return records

    def append(self, record: JudgeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def judge_answers(
    llm: LlmClient,
    questions: Sequence[EvalQuestion],
    answers: Mapping[str, AnswerRecord],
    cache: JudgeCache,
    progress: Callable[[str], None] = lambda line: None,
) -> list[JudgeRecord]:
    """One verdict per question, in question order; judges only the ones not cached yet."""
    missing = [q.id for q in questions if q.id not in answers]
    if missing:
        raise KeyError(f"no saved answer for {', '.join(missing)}")
    done = cache.load()
    for q in questions:
        if q.id in done:
            progress(f"{q.id} cached")
            continue
        record = judge_answer(llm, q, answers[q.id], cache.config)
        cache.append(record)
        done[q.id] = record
        progress(f"{q.id} {record.outcome}")
    return [done[q.id] for q in questions]


@dataclass(frozen=True)
class JudgeStats:
    answerable_correctness: dict[int, int]  # score -> answerable questions (refused = 0)
    unanswerable_correctness: dict[int, int]  # score -> unanswerable questions
    faithfulness: dict[int, int]  # score -> judged answers (refusals claim nothing)
    unparseable: int
    judge_calls: int
    input_tokens: int
    output_tokens: int


def _histogram(scores: Sequence[int | None]) -> dict[int, int]:
    counts = Counter(s for s in scores if s is not None)
    return {score: counts[score] for score in (2, 1, 0)}


def judge_stats(records: Sequence[JudgeRecord]) -> JudgeStats:
    return JudgeStats(
        answerable_correctness=_histogram([r.correctness for r in records if r.answerable]),
        unanswerable_correctness=_histogram([r.correctness for r in records if not r.answerable]),
        faithfulness=_histogram([r.faithfulness for r in records if r.outcome == "judged"]),
        unparseable=sum(r.outcome == "unparseable" for r in records),
        judge_calls=sum(r.attempts for r in records),
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
    )
