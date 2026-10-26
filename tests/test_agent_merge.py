"""The deterministic merge that builds the WorkOrder from the nodes' outputs.

Safety precedence, escalation and citation integrity are enforced here, in code, so no
LLM reply can break them.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from rig_assist.agents.merge import build_work_order
from rig_assist.agents.schemas import (
    Citation,
    Diagnosis,
    EvidenceRef,
    ProposedCause,
    SafetyReview,
    TriageResult,
)
from rig_assist.ingestion.models import Chunk
from rig_assist.rag.pipeline import SNIPPET_CHARS
from rig_assist.retrieval.models import ScoredChunk


def hit(doc_id: str, page: int, text: str, c: int = 0) -> ScoredChunk:
    chunk = Chunk(f"{doc_id}::p{page}::c{c}", doc_id, doc_id, page, None, "pdfplumber", text)
    return ScoredChunk(chunk, score=1.0)


HITS = [
    hit("pump", 12, "Worn liners cause pressure loss."),
    hit("pump", 12, "Second chunk on page 12.", c=1),
    hit("pump", 30, "Never open the fluid end while the pump is pressurised."),
    hit("bop", 5, "Accumulator precharge 1000 psi."),
]

TRIAGE = TriageResult(
    equipment="Triplex mud pump",
    equipment_id="MP-2",
    symptoms=["pressure dropping", "knocking"],
    operating_context="drilling at 9,000 ft",
    open_questions=[],
)


def ref(doc_id: str, page: int) -> EvidenceRef:
    return EvidenceRef(doc_id=doc_id, page=page)


def diagnosis(**overrides: Any) -> Diagnosis:
    data: dict[str, Any] = {
        "causes": [
            ProposedCause(cause="Worn liner", likelihood="high", evidence=[ref("pump", 12)]),
        ],
        "recommended_actions": [
            "Inspect the liners",
            "Open the fluid end to check the valves",
            "Check suction pressure",
        ],
    }
    return Diagnosis(**(data | overrides))


def safety(**overrides: Any) -> SafetyReview:
    data: dict[str, Any] = {
        "warnings": ["Bleed the pump down to zero pressure before opening it"],
        "blocked_action_numbers": [],
        "other_blocked_actions": [],
        "hazard": False,
        "evidence": [ref("pump", 30)],
    }
    return SafetyReview(**(data | overrides))


# --- safety precedence -------------------------------------------------------


def test_a_blocked_action_is_never_recommended() -> None:
    order = build_work_order(
        TRIAGE, HITS, diagnosis(), safety(blocked_action_numbers=[2], hazard=True)
    )
    assert order.recommended_actions == ["Inspect the liners", "Check suction pressure"]
    assert order.blocked_actions == ["Open the fluid end to check the valves"]


def test_safety_can_block_an_action_the_diagnosis_did_not_propose() -> None:
    order = build_work_order(
        TRIAGE,
        HITS,
        diagnosis(),
        safety(other_blocked_actions=["Bypass the relief valve"], hazard=True),
    )
    assert order.blocked_actions == ["Bypass the relief valve"]
    assert len(order.recommended_actions) == 3


def test_a_blocked_action_repeated_in_other_words_of_case_is_still_removed() -> None:
    order = build_work_order(
        TRIAGE,
        HITS,
        diagnosis(recommended_actions=["Inspect the liners", "Bypass  the RELIEF valve"]),
        safety(other_blocked_actions=["bypass the relief valve"], hazard=True),
    )
    assert order.recommended_actions == ["Inspect the liners"]
    assert order.blocked_actions == ["bypass the relief valve"]


def test_safety_warnings_are_kept() -> None:
    order = build_work_order(TRIAGE, HITS, diagnosis(), safety())
    assert order.safety_warnings == ["Bleed the pump down to zero pressure before opening it"]


def test_a_block_number_outside_the_proposed_actions_is_an_error() -> None:
    with pytest.raises(ValueError, match="action 4"):
        build_work_order(TRIAGE, HITS, diagnosis(), safety(blocked_action_numbers=[4]))


def test_duplicate_recommended_actions_are_merged() -> None:
    order = build_work_order(
        TRIAGE,
        HITS,
        diagnosis(recommended_actions=["Inspect the liners", "inspect the  liners"]),
        safety(),
    )
    assert order.recommended_actions == ["Inspect the liners"]


# --- escalation --------------------------------------------------------------


def test_no_escalation_when_a_cause_has_evidence_and_no_hazard() -> None:
    assert not build_work_order(TRIAGE, HITS, diagnosis(), safety()).escalate_to_engineer


def test_a_hazard_flag_escalates() -> None:
    assert build_work_order(TRIAGE, HITS, diagnosis(), safety(hazard=True)).escalate_to_engineer


def test_no_cause_with_evidence_escalates() -> None:
    unsupported = [ProposedCause(cause="Unknown", likelihood="low", evidence=[])]
    order = build_work_order(TRIAGE, HITS, diagnosis(causes=unsupported), safety())
    assert order.escalate_to_engineer


def test_no_causes_at_all_escalates() -> None:
    assert build_work_order(TRIAGE, HITS, diagnosis(causes=[]), safety()).escalate_to_engineer


def test_evidence_that_was_not_retrieved_does_not_count_as_support() -> None:
    invented = [ProposedCause(cause="Worn liner", likelihood="high", evidence=[ref("pump", 99)])]
    order = build_work_order(TRIAGE, HITS, diagnosis(causes=invented), safety())
    assert order.probable_causes[0].evidence == []
    assert order.escalate_to_engineer


# --- citation integrity ------------------------------------------------------


def test_every_citation_maps_to_a_retrieved_chunk() -> None:
    causes = [
        ProposedCause(cause="Worn liner", likelihood="high", evidence=[ref("pump", 12)]),
        ProposedCause(cause="Valve wear", likelihood="low", evidence=[ref("pump", 77)]),
    ]
    order = build_work_order(TRIAGE, HITS, diagnosis(causes=causes), safety())
    retrieved = {(h.chunk.doc_id, h.chunk.page) for h in HITS}
    assert order.citations
    assert all((c.doc_id, c.page) in retrieved for c in order.citations)
    for cause in order.probable_causes:
        assert all((c.doc_id, c.page) in retrieved for c in cause.evidence)


def test_the_snippet_comes_from_the_best_ranked_chunk_on_that_page() -> None:
    order = build_work_order(TRIAGE, HITS, diagnosis(), safety())
    assert order.probable_causes[0].evidence == [
        Citation(doc_id="pump", page=12, snippet="Worn liners cause pressure loss.")
    ]


def test_long_snippets_are_cut() -> None:
    hits = [hit("pump", 12, "word " * 200)]
    order = build_work_order(TRIAGE, hits, diagnosis(), safety(evidence=[]))
    snippet = order.citations[0].snippet
    assert len(snippet) == SNIPPET_CHARS and snippet.endswith("…")


def test_citations_are_the_cause_and_safety_evidence_once_each_in_order() -> None:
    causes = [
        ProposedCause(cause="A", likelihood="high", evidence=[ref("pump", 12), ref("bop", 5)]),
        ProposedCause(cause="B", likelihood="low", evidence=[ref("pump", 12)]),
    ]
    order = build_work_order(TRIAGE, HITS, diagnosis(causes=causes), safety())
    assert [(c.doc_id, c.page) for c in order.citations] == [
        ("pump", 12),
        ("bop", 5),
        ("pump", 30),
    ]


def test_causes_are_ranked_by_likelihood_keeping_the_order_within_a_level() -> None:
    causes = [
        ProposedCause(cause="low one", likelihood="low", evidence=[ref("pump", 12)]),
        ProposedCause(cause="high one", likelihood="high", evidence=[ref("pump", 12)]),
        ProposedCause(cause="medium", likelihood="medium", evidence=[ref("pump", 12)]),
        ProposedCause(cause="high two", likelihood="high", evidence=[ref("pump", 12)]),
    ]
    order = build_work_order(TRIAGE, HITS, diagnosis(causes=causes), safety())
    assert [c.cause for c in order.probable_causes] == ["high one", "high two", "medium", "low one"]


# --- triage fields and the missing-information route -------------------------


def test_equipment_and_symptoms_come_from_triage() -> None:
    order = build_work_order(TRIAGE, HITS, diagnosis(), safety())
    assert order.equipment == "Triplex mud pump (MP-2)"
    assert order.symptoms == ["pressure dropping", "knocking"]
    assert order.open_questions == []


def test_missing_information_gives_a_work_order_with_only_open_questions() -> None:
    triage = TriageResult(
        equipment="Mud pump",
        equipment_id=None,
        symptoms=["noise"],
        operating_context=None,
        open_questions=["Which pump: MP-1 or MP-2?", "Is the noise on suction or discharge?"],
    )
    order = build_work_order(triage, [], None, None)
    assert order.equipment == "Mud pump"
    assert order.open_questions == triage.open_questions
    assert order.probable_causes == [] and order.recommended_actions == []
    assert order.citations == []
    assert not order.escalate_to_engineer  # it asks first; escalation needs a diagnosis


def test_a_diagnosis_without_a_safety_review_is_refused() -> None:
    with pytest.raises(ValueError, match="safety"):
        build_work_order(TRIAGE, HITS, diagnosis(), None)


# --- node output schemas -----------------------------------------------------


def test_node_outputs_are_strict() -> None:
    with pytest.raises(ValidationError):
        SafetyReview.model_validate(
            {
                "warnings": [],
                "blocked_action_numbers": [0],
                "other_blocked_actions": [],
                "hazard": False,
                "evidence": [],
            }  # fmt: skip
        )
    with pytest.raises(ValidationError):
        TriageResult.model_validate(
            {"equipment": "pump", "symptoms": [], "open_questions": [], "extra": 1}
        )
