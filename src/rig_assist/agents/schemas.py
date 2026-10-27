"""The work order the multi-agent graph returns (PLAN.md, Phase 6), and what each LLM
node returns on the way.

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


# --- What the LLM nodes return -------------------------------------------------
# Nodes cite evidence as (doc_id, page) only. The merge looks each one up among the
# retrieved chunks and takes the snippet from there, so a citation can't be invented.


class EvidenceRef(_Strict):
    doc_id: Text
    page: Annotated[int, Field(ge=1)]


class TriageResult(_Strict):
    equipment: Text  # what the technician named, or a generic name ("mud pump")
    equipment_id: Text | None = None
    symptoms: list[Text]
    operating_context: Text | None = None
    open_questions: list[Text]  # shown in the work order; they don't stop the diagnosis
    # True only when the equipment or the problem can't be identified: then the graph asks
    # instead of diagnosing. No default, so the model must decide (run 1: questions alone
    # used to stop the graph, skipping the safety review on urgent hazards).
    critical_info_missing: StrictBool


class SearchPlan(_Strict):
    queries: list[Text]  # new searches; only the first few are run
    done: StrictBool  # the evidence is enough


class ProposedCause(_Strict):
    cause: Text
    likelihood: Likelihood
    evidence: list[EvidenceRef]


class Diagnosis(_Strict):
    causes: list[ProposedCause]
    recommended_actions: list[Text]


class BlockedAction(_Strict):
    # 1-based number of one of the diagnosis's recommended actions: a number, not reworded
    # text, so a paraphrase can't slip past the block.
    number: Annotated[int, Field(ge=1)]
    hazard: Text  # what doing it would endanger: blocks need a hazard, not a preference


class OtherBlockedAction(_Strict):
    action: Text  # an unsafe action nobody proposed (e.g. the technician's own plan)
    hazard: Text


class SafetyReview(_Strict):
    warnings: list[Text]
    blocked: list[BlockedAction]
    other_blocked: list[OtherBlockedAction]
    hazard: StrictBool  # a hazard the technician must not handle alone → escalate
    evidence: list[EvidenceRef]
