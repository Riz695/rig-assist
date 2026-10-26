"""The work order the multi-agent graph returns (PLAN.md, Phase 6).

Strict on purpose: unknown fields, blank strings, pages below 1 and non-boolean flags are
rejected, so an LLM reply that doesn't fit is caught and retried instead of passed on.
An action can't be both recommended and blocked: safety precedence is checked here as
well as enforced by the merge that builds the work order.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Likelihood = Literal["high", "medium", "low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Citation(_Strict):
    doc_id: Text
    page: Annotated[int, Field(ge=1)]  # 1-based PDF page
    snippet: Text


class Cause(_Strict):
    cause: Text
    likelihood: Likelihood
    evidence: list[Citation]  # may be empty: the escalation rule flags unsupported causes


def normalise_action(action: str) -> str:
    """Case and spacing don't make two actions different."""
    return " ".join(action.split()).casefold()


class WorkOrder(_Strict):
    equipment: Text
    symptoms: list[Text]
    probable_causes: list[Cause]
    recommended_actions: list[Text]
    safety_warnings: list[Text]
    blocked_actions: list[Text]
    escalate_to_engineer: StrictBool
    open_questions: list[Text]
    citations: list[Citation]

    @model_validator(mode="after")
    def _blocked_actions_are_not_recommended(self) -> "WorkOrder":
        blocked = {normalise_action(a) for a in self.blocked_actions}
        both = [a for a in self.recommended_actions if normalise_action(a) in blocked]
        if both:
            raise ValueError(f"actions both recommended and blocked: {both}")
        return self
