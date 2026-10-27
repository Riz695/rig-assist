"""The troubleshooting graph (LangGraph): triage → retrieval → diagnosis → safety → report.

One routing decision: when triage finds critical information missing (open questions),
the graph goes straight to the report, which then carries only those questions. The
nodes are the plain functions in `nodes.py`; the report is `merge.build_work_order`, so
safety precedence, escalation and citation integrity are enforced in code.

Each node returns only what it adds to the state; `steps` (per-node usage and time)
accumulates through a reducer. With a tracer, each node is a span, and the LLM and the
retriever are wrapped so every call and search is recorded under its node. Node errors
(an LLM failure, two invalid replies) propagate out of `run`: the caller marks the job
failed.
"""

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from rig_assist.agents.merge import build_work_order
from rig_assist.agents.nodes import run_diagnosis, run_retrieval, run_safety, run_triage
from rig_assist.agents.schemas import Diagnosis, SafetyReview, TriageResult, WorkOrder
from rig_assist.llm.models import LlmClient, Usage
from rig_assist.rag.prompt import citation_label
from rig_assist.retrieval.models import Retriever, ScoredChunk
from rig_assist.tracing import NullTracer, ObservationType, TracedLlm, TracedRetriever, Tracer

NodeName = Literal["triage", "retrieval", "diagnosis", "safety", "report"]
NODES: tuple[NodeName, ...] = ("triage", "retrieval", "diagnosis", "safety", "report")
_SPAN_TYPE: dict[NodeName, ObservationType] = {
    "triage": "chain",
    "retrieval": "chain",
    "diagnosis": "chain",
    "safety": "guardrail",  # it can block actions
    "report": "chain",
}


def _for_trace(update: dict[str, Any]) -> dict[str, Any]:
    """A node's state update as JSON-friendly data: models dumped, chunks as labels."""
    out: dict[str, Any] = {}
    for key, value in update.items():
        if isinstance(value, BaseModel):
            out[key] = value.model_dump(mode="json")
        elif isinstance(value, list) and value and isinstance(value[0], ScoredChunk):
            out[key] = [citation_label(h.chunk.doc_id, h.chunk.page) for h in value]
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Step:
    node: NodeName
    usage: Usage
    seconds: float
    llm_calls: int  # retries included; 0 for the report (code)


class State(TypedDict, total=False):
    description: str
    equipment_id: str | None
    triage: TriageResult
    hits: list[ScoredChunk]  # retrieval's chunks, then safety's new ones
    queries: list[str]
    diagnosis: Diagnosis
    safety: SafetyReview
    work_order: WorkOrder
    steps: Annotated[list[Step], operator.add]  # each node appends its own


class _NodeFn(Protocol):
    """LangGraph's node signature: the parameter must be named `state`."""

    def __call__(self, state: State) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GraphResult:
    work_order: WorkOrder
    steps: list[Step]
    hits: list[ScoredChunk]
    queries: list[str]

    @property
    def visited(self) -> list[str]:
        return [step.node for step in self.steps]

    @property
    def usage(self) -> Usage:
        return sum((step.usage for step in self.steps), Usage())

    @property
    def llm_calls(self) -> int:
        return sum(step.llm_calls for step in self.steps)


class TroubleshootingGraph:
    def __init__(
        self,
        llm: LlmClient,
        retriever: Retriever,
        clock: Callable[[], float] = time.perf_counter,
        tracer: Tracer | None = None,
    ) -> None:
        self.tracer = tracer or NullTracer()
        self.llm: LlmClient = TracedLlm(llm, self.tracer) if tracer else llm
        self.retriever: Retriever = TracedRetriever(retriever, self.tracer) if tracer else retriever
        self.clock = clock
        self.graph = self._build()

    def _timed(
        self, node: NodeName, work: Callable[[State], tuple[dict[str, Any], Usage, int]]
    ) -> _NodeFn:
        """Wrap a node: run it in a span, then record its usage and time as a Step."""

        def run(state: State) -> dict[str, Any]:
            with self.tracer.observe(node, _SPAN_TYPE[node]) as span:
                start = self.clock()
                try:
                    update, usage, calls = work(state)
                except Exception as error:
                    span.update(level="ERROR", status_message=f"{type(error).__name__}: {error}")
                    raise
                seconds = self.clock() - start
                span.update(output=_for_trace(update), usage=usage)
            return update | {"steps": [Step(node, usage, seconds, calls)]}

        return run

    def _triage(self, state: State) -> tuple[dict[str, Any], Usage, int]:
        result = run_triage(self.llm, state["description"], state.get("equipment_id"))
        return {"triage": result.value}, result.usage, result.attempts

    def _retrieval(self, state: State) -> tuple[dict[str, Any], Usage, int]:
        result = run_retrieval(self.llm, self.retriever, state["description"], state["triage"])
        return {"hits": result.hits, "queries": result.queries}, result.usage, result.llm_calls

    def _diagnosis(self, state: State) -> tuple[dict[str, Any], Usage, int]:
        result = run_diagnosis(self.llm, state["description"], state["triage"], state["hits"])
        return {"diagnosis": result.value}, result.usage, result.attempts

    def _safety(self, state: State) -> tuple[dict[str, Any], Usage, int]:
        result = run_safety(
            self.llm,
            self.retriever,
            state["description"],
            state["triage"],
            state["diagnosis"],
            state["hits"],
        )
        hits = state["hits"] + result.new_hits
        review = result.review
        return {"safety": review.value, "hits": hits}, review.usage, review.attempts

    def _report(self, state: State) -> tuple[dict[str, Any], Usage, int]:
        order = build_work_order(
            state["triage"],
            state.get("hits", []),
            state.get("diagnosis"),
            state.get("safety"),
        )
        return {"work_order": order}, Usage(), 0

    @staticmethod
    def _after_triage(state: State) -> NodeName:
        return "report" if state["triage"].open_questions else "retrieval"

    def _build(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("triage", self._timed("triage", self._triage))
        graph.add_node("retrieval", self._timed("retrieval", self._retrieval))
        graph.add_node("diagnosis", self._timed("diagnosis", self._diagnosis))
        graph.add_node("safety", self._timed("safety", self._safety))
        graph.add_node("report", self._timed("report", self._report))
        graph.add_edge(START, "triage")
        graph.add_conditional_edges("triage", self._after_triage, ["retrieval", "report"])
        graph.add_edge("retrieval", "diagnosis")
        graph.add_edge("diagnosis", "safety")
        graph.add_edge("safety", "report")
        graph.add_edge("report", END)
        return graph.compile()

    def run(self, description: str, equipment_id: str | None = None) -> GraphResult:
        state = self.graph.invoke(
            {"description": description, "equipment_id": equipment_id, "steps": []}
        )
        return GraphResult(
            work_order=state["work_order"],
            steps=state["steps"],
            hits=state.get("hits", []),
            queries=state.get("queries", []),
        )
