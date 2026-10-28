"""The work order schema and validated structured output from the LLM (one retry).

All LLM replies are scripted with FakeLlm.
"""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rig_assist.agents.schemas import ActionVerdict, Cause, Citation, WorkOrder
from rig_assist.agents.structured import StructuredOutputError, complete_structured
from rig_assist.llm.fake import FakeLlm
from rig_assist.llm.models import Message, Usage


def citation(page: int = 12) -> dict[str, Any]:
    return {"doc_id": "mud-pump-manual", "page": page, "snippet": "Replace worn liners."}


def work_order(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "equipment": "Triplex mud pump MP-2",
        "symptoms": ["pressure dropping", "knocking"],
        "probable_causes": [
            {"cause": "Worn liner", "likelihood": "high", "evidence": [citation()]},
        ],
        "recommended_actions": ["Inspect the liners"],
        "safety_warnings": ["Isolate and bleed down the pump before opening it"],
        "blocked_actions": [],
        "escalate_to_engineer": False,
        "open_questions": [],
        "citations": [citation()],
    }
    return data | overrides


# --- schema ------------------------------------------------------------------


def test_a_complete_work_order_validates() -> None:
    order = WorkOrder.model_validate(work_order())
    assert order.probable_causes[0] == Cause(
        cause="Worn liner", likelihood="high", evidence=[Citation(**citation())]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"equipment": ""},  # blank
        {"symptoms": ["ok", " "]},  # a blank item
        {"probable_causes": [{"cause": "x", "likelihood": "certain", "evidence": []}]},
        {"citations": [citation(page=0)]},  # pages are 1-based
        {"citations": [citation() | {"snippet": ""}]},
        {"escalate_to_engineer": "maybe"},
        {"severity": "high"},  # unknown field
    ],
)
def test_invalid_work_orders_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        WorkOrder.model_validate(work_order(**overrides))


def test_a_missing_field_is_rejected() -> None:
    data = work_order()
    del data["blocked_actions"]
    with pytest.raises(ValidationError):
        WorkOrder.model_validate(data)


def test_an_action_cannot_be_both_recommended_and_blocked() -> None:
    with pytest.raises(ValidationError, match="both recommended and blocked"):
        WorkOrder.model_validate(
            work_order(
                recommended_actions=["Open the pump under pressure"],
                blocked_actions=["open the pump  under pressure"],  # case/space-insensitive
            )
        )


# --- the safety review's verdict per proposed action ---------------------------


def test_an_unsafe_verdict_needs_a_hazard() -> None:
    with pytest.raises(ValidationError, match="hazard"):
        ActionVerdict(number=1, action="Open the fluid end", verdict="unsafe", hazard=None)


def test_a_safe_verdict_has_no_hazard() -> None:
    with pytest.raises(ValidationError, match="hazard"):
        ActionVerdict(number=1, action="Inspect the liners", verdict="safe", hazard="x")
    ok = ActionVerdict(number=1, action="Inspect the liners", verdict="safe", hazard=None)
    assert ok.verdict == "safe"


def test_a_verdict_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ActionVerdict.model_validate(
            {"number": 1, "action": "A", "verdict": "safe", "hazard": None, "why": "x"}
        )
    with pytest.raises(ValidationError):  # only "safe" or "unsafe"
        ActionVerdict.model_validate(
            {"number": 1, "action": "A", "verdict": "maybe", "hazard": None}
        )


def test_a_cause_may_have_no_evidence() -> None:
    # Allowed by the schema; the escalation rule then flags it.
    cause = Cause.model_validate({"cause": "Unknown", "likelihood": "low", "evidence": []})
    assert cause.evidence == []


# --- structured output with one retry ----------------------------------------


class Reply(BaseModel):
    answer: str
    count: int


MESSAGES = [Message("system", "Reply in JSON."), Message("user", "Go.")]


def test_a_valid_reply_is_parsed_on_the_first_call() -> None:
    llm = FakeLlm([json.dumps({"answer": "yes", "count": 2})])
    result = complete_structured(llm, MESSAGES, Reply)
    assert result.value == Reply(answer="yes", count=2)
    assert result.attempts == 1
    assert len(llm.calls) == 1


def test_a_fenced_reply_is_accepted() -> None:
    llm = FakeLlm(['```json\n{"answer": "yes", "count": 2}\n```'])
    assert complete_structured(llm, MESSAGES, Reply).value.count == 2


def test_an_invalid_reply_is_retried_once_with_the_error() -> None:
    llm = FakeLlm(['{"answer": "yes"}', '{"answer": "yes", "count": 3}'])
    result = complete_structured(llm, MESSAGES, Reply)
    assert result.value.count == 3
    assert result.attempts == 2
    retry = llm.calls[1]
    assert retry[:2] == MESSAGES
    assert retry[2] == Message("assistant", '{"answer": "yes"}')
    assert retry[3].role == "user" and "count" in retry[3].content  # names what was wrong


def test_usage_adds_up_over_both_attempts() -> None:
    llm = FakeLlm(["not json", '{"answer": "a", "count": 1}'])
    result = complete_structured(llm, MESSAGES, Reply)
    first = Usage(input_tokens=4, output_tokens=2)  # word counts (FakeLlm)
    assert result.usage.output_tokens == first.output_tokens + 4
    assert result.usage.input_tokens > 2 * first.input_tokens  # the retry resends the context


def test_two_invalid_replies_raise_with_the_last_reply() -> None:
    llm = FakeLlm(["not json", '{"answer": 1}'])
    with pytest.raises(StructuredOutputError) as caught:
        complete_structured(llm, MESSAGES, Reply)
    assert caught.value.reply == '{"answer": 1}'
    assert caught.value.usage.output_tokens == 4
    assert len(llm.calls) == 2


def test_the_max_tokens_limit_is_passed_on() -> None:
    llm = FakeLlm(['{"answer": "a", "count": 1}'])
    complete_structured(llm, MESSAGES, Reply, max_tokens=321)
    assert llm.max_tokens == [321]


def test_a_work_order_reply_is_validated_and_retried() -> None:
    bad = work_order(recommended_actions=["X"], blocked_actions=["x"])
    llm = FakeLlm([json.dumps(bad), json.dumps(work_order())])
    result = complete_structured(llm, MESSAGES, WorkOrder)
    assert result.attempts == 2
    assert "both recommended and blocked" in llm.calls[1][3].content
