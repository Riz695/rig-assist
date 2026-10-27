"""The LangGraph troubleshooting graph: routing, end-to-end safety precedence and
citations, usage. Fake LLM (scripted node replies) and a scripted retriever."""

import json
from typing import Any

import pytest

from rig_assist.agents.graph import NODES, Step, TroubleshootingGraph
from rig_assist.agents.structured import StructuredOutputError
from rig_assist.ingestion.models import Chunk
from rig_assist.llm.fake import FakeLlm
from rig_assist.retrieval.models import ScoredChunk

DESCRIPTION = "MP-2 losing pressure and knocking. I was going to bypass the relief valve."


def hit(doc_id: str, page: int, text: str = "") -> ScoredChunk:
    text = text or f"{doc_id} page {page}"
    chunk = Chunk(f"{doc_id}::p{page}::c0", doc_id, doc_id, page, None, "pdfplumber", text)
    return ScoredChunk(chunk, score=1.0)


class Retriever:
    """The first query finds the cause pages; any query mentioning safety finds page 30."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.queries.append(query)
        if "safety" in query:
            return [hit("pump", 30, "Never open the fluid end under pressure.")]
        return [hit("pump", 12, "Worn liners cause pressure loss."), hit("pump", 14)]


def triage(open_questions: list[str] | None = None, critical: bool = False) -> str:
    return json.dumps(
        {
            "equipment": "Triplex mud pump",
            "equipment_id": "MP-2",
            "symptoms": ["pressure loss", "knocking"],
            "operating_context": None,
            "open_questions": open_questions or [],
            "critical_info_missing": critical,
        }
    )


PLAN_DONE = json.dumps({"queries": [], "done": True})
DIAGNOSIS = json.dumps(
    {
        "causes": [
            {
                "cause": "Worn liner",
                "likelihood": "high",
                "evidence": [{"doc_id": "pump", "page": 12}],
            },
            {
                "cause": "Suction blockage",
                "likelihood": "low",
                "evidence": [{"doc_id": "pump", "page": 99}],  # not retrieved
            },
        ],
        "recommended_actions": ["Inspect the liners", "Open the fluid end to check valves"],
    }
)


def safety(**overrides: Any) -> str:
    data = {
        "warnings": ["Bleed the pump to zero pressure and lock it out before opening"],
        "blocked": [{"number": 2, "hazard": "Opening the fluid end while pressurised"}],
        "other_blocked": [
            {"action": "Bypass the relief valve", "hazard": "No overpressure protection"}
        ],
        "hazard": False,
        "evidence": [{"doc_id": "pump", "page": 30}],
    }
    return json.dumps(data | overrides)


def full_run(**safety_overrides: Any) -> tuple[Any, FakeLlm, Retriever]:
    replies = [triage(), PLAN_DONE, DIAGNOSIS, safety(**safety_overrides)]
    llm, retriever = FakeLlm(replies), Retriever()
    result = TroubleshootingGraph(llm, retriever).run(DESCRIPTION)
    return result, llm, retriever


# --- routing -----------------------------------------------------------------


def test_normal_input_visits_every_node_in_order() -> None:
    result, llm, _ = full_run()
    assert result.visited == ["triage", "retrieval", "diagnosis", "safety", "report"]
    assert list(NODES) == result.visited
    assert len(llm.calls) == 4


def test_missing_critical_information_goes_straight_to_the_report() -> None:
    llm, retriever = FakeLlm([triage(["Which pump?"], critical=True)]), Retriever()
    result = TroubleshootingGraph(llm, retriever).run("pump noise")
    assert result.visited == ["triage", "report"]
    assert result.work_order.open_questions == ["Which pump?"]
    assert result.work_order.probable_causes == []
    assert retriever.queries == [] and len(llm.calls) == 1


def test_open_questions_without_missing_critical_info_still_get_a_full_diagnosis() -> None:
    replies = [triage(["Which shift noticed it?"]), PLAN_DONE, DIAGNOSIS, safety()]
    result = TroubleshootingGraph(FakeLlm(replies), Retriever()).run(DESCRIPTION)
    assert result.visited == list(NODES)
    assert result.work_order.open_questions == ["Which shift noticed it?"]
    assert result.work_order.blocked_actions  # the safety review ran


# --- end to end through the graph --------------------------------------------


def test_the_safety_node_blocks_an_unsafe_suggestion() -> None:
    order = full_run()[0].work_order
    assert order.recommended_actions == ["Inspect the liners"]
    assert order.blocked_actions == [
        "Open the fluid end to check valves",
        "Bypass the relief valve",
    ]
    assert order.safety_warnings == [
        "Bleed the pump to zero pressure and lock it out before opening"
    ]


def test_every_citation_maps_to_a_retrieved_chunk_including_safety_hits() -> None:
    result, _, _ = full_run()
    retrieved = {(h.chunk.doc_id, h.chunk.page) for h in result.hits}
    assert [(c.doc_id, c.page) for c in result.work_order.citations] == [("pump", 12), ("pump", 30)]
    assert all((c.doc_id, c.page) in retrieved for c in result.work_order.citations)
    assert result.work_order.probable_causes[1].evidence == []  # page 99 wasn't retrieved


def test_a_hazard_escalates_through_the_graph() -> None:
    assert full_run(hazard=True)[0].work_order.escalate_to_engineer
    assert not full_run()[0].work_order.escalate_to_engineer


def test_the_technicians_equipment_id_reaches_the_work_order() -> None:
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    result = TroubleshootingGraph(llm, Retriever()).run(DESCRIPTION, equipment_id="MP-9")
    assert result.work_order.equipment == "Triplex mud pump (MP-9)"


# --- usage, steps, errors ----------------------------------------------------


def test_usage_adds_up_every_llm_call() -> None:
    result, llm, _ = full_run()
    replies = [triage(), PLAN_DONE, DIAGNOSIS, safety()]
    assert result.usage.output_tokens == sum(len(r.split()) for r in replies)
    assert result.usage.input_tokens == sum(
        len(m.content.split()) for call in llm.calls for m in call
    )


def test_each_step_records_its_usage_and_time() -> None:
    ticks = iter(float(i) for i in range(100))
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    result = TroubleshootingGraph(llm, Retriever(), clock=lambda: next(ticks)).run(DESCRIPTION)
    assert [s.node for s in result.steps] == result.visited
    assert all(s.seconds == 1.0 for s in result.steps)
    assert result.steps[-1].usage.output_tokens == 0  # the report is code, not an LLM call
    assert sum(s.usage.output_tokens for s in result.steps) == result.usage.output_tokens
    assert [s.llm_calls for s in result.steps] == [1, 1, 1, 1, 0]


def test_retries_and_planner_rounds_count_as_llm_calls() -> None:
    plan = json.dumps({"queries": ["liner wear"], "done": False})
    replies = ["bad", triage(), plan, PLAN_DONE, DIAGNOSIS, safety()]
    result = TroubleshootingGraph(FakeLlm(replies), Retriever()).run(DESCRIPTION)
    assert [s.llm_calls for s in result.steps] == [2, 2, 1, 1, 0]
    assert result.llm_calls == 6


def test_the_queries_run_are_reported() -> None:
    result, _, retriever = full_run()
    assert result.queries == retriever.queries[:1]  # the safety search is separate
    assert len(retriever.queries) == 2


def test_a_node_failure_propagates() -> None:
    llm = FakeLlm([triage(), PLAN_DONE, "not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        TroubleshootingGraph(llm, Retriever()).run(DESCRIPTION)


# --- live progress -----------------------------------------------------------


class Recorder:
    """A progress listener that records what the graph reports, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.finished: list[Step] = []

    def node_started(self, node: str) -> None:
        self.events.append(("start", node))

    def node_finished(self, step: Step) -> None:
        self.events.append(("end", step.node))
        self.finished.append(step)


def test_progress_reports_each_node_as_it_starts_and_finishes() -> None:
    recorder = Recorder()
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    result = TroubleshootingGraph(llm, Retriever()).run(DESCRIPTION, progress=recorder)
    assert recorder.events == [(kind, node) for node in NODES for kind in ("start", "end")]
    assert recorder.finished == result.steps  # the same steps, reported as they happen


def test_progress_follows_the_missing_information_route() -> None:
    recorder = Recorder()
    llm = FakeLlm([triage(["Which pump?"], critical=True)])
    TroubleshootingGraph(llm, Retriever()).run("pump noise", progress=recorder)
    assert recorder.events == [
        ("start", "triage"),
        ("end", "triage"),
        ("start", "report"),
        ("end", "report"),
    ]


def test_a_failing_node_is_reported_started_but_never_finished() -> None:
    recorder = Recorder()
    llm = FakeLlm([triage(), PLAN_DONE, "not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        TroubleshootingGraph(llm, Retriever()).run(DESCRIPTION, progress=recorder)
    assert recorder.events[-1] == ("start", "diagnosis")
    assert ("end", "diagnosis") not in recorder.events
