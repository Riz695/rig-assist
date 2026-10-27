"""The LLM nodes (triage, retrieval, diagnosis, safety) with scripted LLM replies and a
scripted retriever. The report node is `build_work_order` (test_agent_merge.py)."""

import json
from typing import Any

import pytest

from rig_assist.agents.nodes import (
    MAX_EVIDENCE,
    MAX_SEARCH_ROUNDS,
    run_diagnosis,
    run_retrieval,
    run_safety,
    run_triage,
)
from rig_assist.agents.schemas import Diagnosis, ProposedCause, TriageResult
from rig_assist.agents.structured import StructuredOutputError, complete_structured
from rig_assist.ingestion.models import Chunk
from rig_assist.llm.fake import FakeLlm
from rig_assist.llm.models import Message
from rig_assist.retrieval.models import ScoredChunk

DESCRIPTION = "Mud pump 2 is losing pressure and knocking since this morning."


def hit(doc_id: str, page: int, c: int = 0, text: str = "") -> ScoredChunk:
    chunk = Chunk(
        f"{doc_id}::p{page}::c{c}", doc_id, doc_id.title(), page, None, "pdfplumber",
        text or f"text of {doc_id} page {page}",
    )  # fmt: skip
    return ScoredChunk(chunk, score=1.0)


class ScriptedRetriever:
    def __init__(self, results: dict[str, list[ScoredChunk]] | None = None) -> None:
        self.results = results or {}
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.queries.append((query, k))
        return self.results.get(query, [])[:k]


TRIAGE = TriageResult(
    equipment="Triplex mud pump",
    equipment_id="MP-2",
    symptoms=["pressure loss", "knocking"],
    operating_context=None,
    open_questions=[],
    critical_info_missing=False,
)
BASE_QUERY = "Triplex mud pump: pressure loss; knocking"


def reply(**data: Any) -> str:
    return json.dumps(data)


# --- structured output: context checks ---------------------------------------


def test_a_failed_check_is_retried_like_an_invalid_reply() -> None:
    llm = FakeLlm(
        [reply(causes=[], recommended_actions=[]), reply(causes=[], recommended_actions=["a"])]
    )

    def needs_an_action(value: Diagnosis) -> None:
        if not value.recommended_actions:
            raise ValueError("propose at least one action")

    result = complete_structured(llm, [Message("user", "go")], Diagnosis, check=needs_an_action)
    assert result.attempts == 2
    assert "propose at least one action" in llm.calls[1][-1].content


# --- triage ------------------------------------------------------------------


def triage_reply(**overrides: Any) -> str:
    data = {
        "equipment": "Triplex mud pump",
        "equipment_id": None,
        "symptoms": ["pressure loss", "knocking"],
        "operating_context": None,
        "open_questions": [],
        "critical_info_missing": False,
    }
    return json.dumps(data | overrides)


def test_triage_sends_the_description_and_returns_the_result() -> None:
    llm = FakeLlm([triage_reply()])
    result = run_triage(llm, DESCRIPTION)
    assert result.value.symptoms == ["pressure loss", "knocking"]
    assert DESCRIPTION in llm.calls[0][-1].content
    assert llm.calls[0][0].role == "system"


def test_triage_keeps_the_equipment_id_the_technician_gave() -> None:
    llm = FakeLlm([triage_reply(equipment_id="MP-1")])
    result = run_triage(llm, DESCRIPTION, equipment_id="MP-2")
    assert result.value.equipment_id == "MP-2"
    assert "MP-2" in llm.calls[0][-1].content


def test_triage_can_ask_open_questions() -> None:
    llm = FakeLlm([triage_reply(open_questions=["Which pump?"])])
    assert run_triage(llm, "pump noise").value.open_questions == ["Which pump?"]


# --- retrieval (bounded loop) ------------------------------------------------


def plan(*queries: str, done: bool = False) -> str:
    return reply(queries=list(queries), done=done)


def test_the_first_search_uses_equipment_and_symptoms_without_the_llm() -> None:
    retriever = ScriptedRetriever({BASE_QUERY: [hit("pump", 12)]})
    llm = FakeLlm([plan(done=True)])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE)
    assert retriever.queries[0][0] == BASE_QUERY
    assert [h.chunk.chunk_id for h in result.hits] == ["pump::p12::c0"]
    assert result.queries == [BASE_QUERY]


def test_the_planner_sees_what_was_found_and_its_queries_run() -> None:
    retriever = ScriptedRetriever(
        {BASE_QUERY: [hit("pump", 12)], "liner wear limits": [hit("pump", 30)]}
    )
    llm = FakeLlm([plan("liner wear limits"), plan(done=True)])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE)
    assert "[pump p.12]" in llm.calls[0][-1].content
    assert [q for q, _ in retriever.queries] == [BASE_QUERY, "liner wear limits"]
    assert [h.chunk.page for h in result.hits] == [12, 30]


def test_hits_found_twice_are_kept_once_at_their_first_position() -> None:
    retriever = ScriptedRetriever(
        {BASE_QUERY: [hit("pump", 12), hit("pump", 30)], "q2": [hit("pump", 30), hit("pump", 7)]}
    )
    llm = FakeLlm([plan("q2"), plan(done=True)])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE)
    assert [h.chunk.page for h in result.hits] == [12, 30, 7]


def test_the_loop_is_bounded_even_if_the_planner_never_stops() -> None:
    retriever = ScriptedRetriever()
    llm = FakeLlm([plan(f"q{i}") for i in range(10)])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE)
    assert len(llm.calls) == MAX_SEARCH_ROUNDS
    assert result.rounds == MAX_SEARCH_ROUNDS


def test_a_repeated_query_is_not_searched_again() -> None:
    retriever = ScriptedRetriever()
    llm = FakeLlm([plan(BASE_QUERY.upper()), plan(done=True)])
    run_retrieval(llm, retriever, DESCRIPTION, TRIAGE)
    assert len(retriever.queries) == 1


def test_no_new_query_ends_the_loop() -> None:
    llm = FakeLlm([plan()])
    result = run_retrieval(llm, ScriptedRetriever(), DESCRIPTION, TRIAGE)
    assert len(llm.calls) == 1 and result.rounds == 1


def test_evidence_is_capped() -> None:
    many = [hit("pump", p) for p in range(1, 40)]
    retriever = ScriptedRetriever({BASE_QUERY: many[:20], "more": many[20:]})
    llm = FakeLlm([plan("more"), plan(done=True)])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE, per_query_k=20)
    assert len(result.hits) == MAX_EVIDENCE


def test_a_full_evidence_list_skips_the_planner() -> None:
    retriever = ScriptedRetriever({BASE_QUERY: [hit("pump", p) for p in range(1, 30)]})
    llm = FakeLlm([])
    result = run_retrieval(llm, retriever, DESCRIPTION, TRIAGE, per_query_k=30)
    assert llm.calls == [] and len(result.hits) == MAX_EVIDENCE


def test_retrieval_usage_adds_up_the_planner_calls() -> None:
    llm = FakeLlm([plan("x"), plan(done=True)])
    result = run_retrieval(llm, ScriptedRetriever(), DESCRIPTION, TRIAGE)
    assert result.usage.output_tokens == sum(len(r.split()) for r in [plan("x"), plan(done=True)])


# --- diagnosis ---------------------------------------------------------------


def diagnosis_reply() -> str:
    return reply(
        causes=[
            {
                "cause": "Worn liner",
                "likelihood": "high",
                "evidence": [{"doc_id": "pump", "page": 12}],
            }
        ],
        recommended_actions=["Inspect the liners", "Open the fluid end"],
    )


def test_diagnosis_reads_labelled_evidence_and_returns_causes() -> None:
    llm = FakeLlm([diagnosis_reply()])
    result = run_diagnosis(llm, DESCRIPTION, TRIAGE, [hit("pump", 12, text="Liners wear.")])
    assert result.value.causes[0].cause == "Worn liner"
    prompt = llm.calls[0][-1].content
    assert "[pump p.12]" in prompt and "Liners wear." in prompt and DESCRIPTION in prompt


def test_two_invalid_diagnoses_raise() -> None:
    llm = FakeLlm(["no", "still no"])
    with pytest.raises(StructuredOutputError):
        run_diagnosis(llm, DESCRIPTION, TRIAGE, [hit("pump", 12)])


# --- safety ------------------------------------------------------------------

DIAGNOSIS = Diagnosis(
    causes=[ProposedCause(cause="Worn liner", likelihood="high", evidence=[])],
    recommended_actions=["Inspect the liners", "Open the fluid end", "inspect the  liners"],
)


def safety_reply(**overrides: Any) -> str:
    data = {
        "warnings": ["Bleed down before opening"],
        "blocked": [{"number": 2, "hazard": "Opening a running pump"}],
        "other_blocked": [],
        "hazard": True,
        "evidence": [{"doc_id": "pump", "page": 30}],
    }
    return json.dumps(data | overrides)


def test_safety_numbers_the_deduplicated_actions_in_its_prompt() -> None:
    llm = FakeLlm([safety_reply()])
    run_safety(llm, ScriptedRetriever(), DESCRIPTION, TRIAGE, DIAGNOSIS, [])
    prompt = llm.calls[0][-1].content
    assert "1. Inspect the liners" in prompt and "2. Open the fluid end" in prompt
    assert "3." not in prompt


def test_safety_searches_the_manuals_for_hazards_and_returns_the_new_hits() -> None:
    retriever = ScriptedRetriever()
    retriever.results = {}
    llm = FakeLlm([safety_reply()])
    known = [hit("pump", 12)]

    def search(query: str, k: int) -> list[ScoredChunk]:
        retriever.queries.append((query, k))
        return [hit("pump", 30), hit("pump", 12)]

    retriever.search = search  # type: ignore[method-assign]
    result = run_safety(llm, retriever, DESCRIPTION, TRIAGE, DIAGNOSIS, known)
    query = retriever.queries[0][0]
    assert "Triplex mud pump" in query and "safety" in query.lower()
    assert [h.chunk.page for h in result.new_hits] == [30]  # page 12 was already known
    assert "[pump p.30]" in llm.calls[0][-1].content and "[pump p.12]" in llm.calls[0][-1].content
    assert [b.number for b in result.review.value.blocked] == [2]


def test_a_block_number_out_of_range_is_retried() -> None:
    llm = FakeLlm([safety_reply(blocked=[{"number": 3, "hazard": "x"}]), safety_reply()])
    result = run_safety(llm, ScriptedRetriever(), DESCRIPTION, TRIAGE, DIAGNOSIS, [])
    assert result.review.attempts == 2
    assert "3" in llm.calls[1][-1].content
