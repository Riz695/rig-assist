"""The troubleshooting graph (LangGraph): triage → retrieval → diagnosis → safety → report.

One routing decision: when triage finds critical information missing (open questions),
the graph goes straight to the report, which then carries only those questions. The
nodes are the plain functions in `nodes.py`; the report is `merge.build_work_order`, so
safety precedence, escalation and citation integrity are enforced in code.

Each node returns only what it adds to the state; `steps` (per-node usage and time)
accumulates through a reducer. Node errors (an LLM failure, two invalid replies)
propagate out of `run`: the caller marks the job failed.
"""

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rig_assist.agents.merge import build_work_order
from rig_assist.agents.nodes import run_diagnosis, run_retrieval, run_safety, run_triage
from rig_assist.agents.schemas import Diagnosis, SafetyReview, TriageResult, WorkOrder
from rig_assist.llm.models import LlmClient, Usage
from rig_assist.retrieval.models import Retriever, ScoredChunk

NodeName = Literal["triage", "retrieval", "diagnosis", "safety", "report"]
NODES: tuple[NodeName, ...] = ("triage", "retrieval", "diagnosis", "safety", "report")


@dataclass(frozen=True)
class Step:
    node: NodeName
    usage: Usage
    seconds: float


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


class TroubleshootingGraph:
    def __init__(
        self,
        llm: LlmClient,
        retriever: Retriever,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.clock = clock
        self.graph = self._build()

    def _timed(
        self, node: NodeName, work: Callable[[State], tuple[dict[str, Any], Usage]]
    ) -> _NodeFn:
        """Wrap a node: run it, then record its usage and time as a Step."""

        def run(state: State) -> dict[str, Any]:
            start = self.clock()
            update, usage = work(state)
            return update | {"steps": [Step(node, usage, self.clock() - start)]}

        return run

    def _triage(self, state: State) -> tuple[dict[str, Any], Usage]:
        result = run_triage(self.llm, state["description"], state.get("equipment_id"))
        return {"triage": result.value}, result.usage

    def _retrieval(self, state: State) -> tuple[dict[str, Any], Usage]:
        result = run_retrieval(self.llm, self.retriever, state["description"], state["triage"])
        return {"hits": result.hits, "queries": result.queries}, result.usage

    def _diagnosis(self, state: State) -> tuple[dict[str, Any], Usage]:
        result = run_diagnosis(self.llm, state["description"], state["triage"], state["hits"])
        return {"diagnosis": result.value}, result.usage

    def _safety(self, state: State) -> tuple[dict[str, Any], Usage]:
        result = run_safety(
            self.llm,
            self.retriever,
            state["description"],
            state["triage"],
            state["diagnosis"],
            state["hits"],
        )
        hits = state["hits"] + result.new_hits
        return {"safety": result.review.value, "hits": hits}, result.review.usage

    def _report(self, state: State) -> tuple[dict[str, Any], Usage]:
        order = build_work_order(
            state["triage"],
            state.get("hits", []),
            state.get("diagnosis"),
            state.get("safety"),
        )
        return {"work_order": order}, Usage()

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
