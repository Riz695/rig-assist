"""The SQLite job store and the job runner (lifecycle queued → running → done | failed)."""

import json
from pathlib import Path
from typing import Any

import pytest

from rig_assist.agents.graph import GraphResult, Step
from rig_assist.agents.schemas import WorkOrder
from rig_assist.agents.structured import StructuredOutputError
from rig_assist.jobs import INTERRUPTED, JobStore, run_job
from rig_assist.llm.models import LlmError, Usage
from rig_assist.llm.usage import Prices

ORDER = WorkOrder(
    equipment="Mud pump",
    symptoms=["noise"],
    probable_causes=[],
    recommended_actions=[],
    safety_warnings=[],
    blocked_actions=[],
    escalate_to_engineer=True,
    open_questions=[],
    citations=[],
)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs" / "jobs.sqlite")


# --- store -------------------------------------------------------------------


def test_a_new_job_is_queued_with_its_request(store: JobStore) -> None:
    job = store.create({"description": "pump noise", "equipment_id": None})
    assert job.status == "queued"
    assert job.request == {"description": "pump noise", "equipment_id": None}
    assert job.result is None and job.error is None and job.started_at is None
    assert store.get(job.id) == job


def test_job_ids_are_unique(store: JobStore) -> None:
    ids = {store.create({"description": "x"}).id for _ in range(20)}
    assert len(ids) == 20


def test_an_unknown_id_is_none(store: JobStore) -> None:
    assert store.get("no-such-job") is None


def test_done_stores_the_result_usage_and_steps(store: JobStore) -> None:
    job = store.create({"description": "x"})
    store.start(job.id)
    steps = [{"node": "triage", "seconds": 0.5, "input_tokens": 10, "output_tokens": 3}]
    store.finish(job.id, ORDER.model_dump(), {"input_tokens": 10, "output_tokens": 3}, steps)
    done = store.get(job.id)
    assert done is not None and done.status == "done"
    assert WorkOrder.model_validate(done.result) == ORDER
    assert done.usage == {"input_tokens": 10, "output_tokens": 3}
    assert done.steps == steps
    assert done.started_at is not None and done.finished_at is not None


def test_failed_stores_the_error(store: JobStore) -> None:
    job = store.create({"description": "x"})
    store.start(job.id)
    store.fail(job.id, "LLM unreachable")
    failed = store.get(job.id)
    assert failed is not None and failed.status == "failed"
    assert failed.error == "LLM unreachable" and failed.result is None


@pytest.mark.parametrize("move", ["finish_queued", "start_twice", "fail_after_done"])
def test_invalid_transitions_are_refused(store: JobStore, move: str) -> None:
    job = store.create({"description": "x"})
    with pytest.raises(ValueError, match="cannot"):
        if move == "finish_queued":
            store.finish(job.id, {}, {}, [])
        elif move == "start_twice":
            store.start(job.id)
            store.start(job.id)
        else:
            store.start(job.id)
            store.finish(job.id, {}, {}, [])
            store.fail(job.id, "late")


def test_the_store_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "jobs.sqlite"
    job = JobStore(path).create({"description": "x"})
    assert JobStore(path).get(job.id) == job


def test_unfinished_jobs_fail_after_a_restart(store: JobStore) -> None:
    queued = store.create({"description": "a"})
    running = store.create({"description": "b"})
    store.start(running.id)
    done = store.create({"description": "c"})
    store.start(done.id)
    store.finish(done.id, {}, {}, [])
    assert store.fail_unfinished() == 2
    for job_id in (queued.id, running.id):
        job = store.get(job_id)
        assert job is not None and job.status == "failed" and job.error == INTERRUPTED
    assert store.get(done.id).status == "done"  # type: ignore[union-attr]


# --- runner ------------------------------------------------------------------


class FakeGraph:
    def __init__(self, store: JobStore, error: Exception | None = None) -> None:
        self.store = store
        self.error = error
        self.seen_status: list[str] = []
        self.calls: list[tuple[str, str | None]] = []

    def run(self, description: str, equipment_id: str | None = None) -> GraphResult:
        self.calls.append((description, equipment_id))
        [job] = self.store.all()
        self.seen_status.append(job.status)
        if self.error:
            raise self.error
        steps = [
            Step("triage", Usage(100, 20), 0.25, llm_calls=1),
            Step("report", Usage(), 0.01, llm_calls=0),
        ]
        return GraphResult(work_order=ORDER, steps=steps, hits=[], queries=[])


def test_run_job_marks_running_then_done(store: JobStore) -> None:
    job = store.create({"description": "pump noise", "equipment_id": "MP-2"})
    graph = FakeGraph(store)
    run_job(store, graph, job.id, prices=Prices(input_per_1m=1.0, output_per_1m=2.0))
    assert graph.seen_status == ["running"]
    assert graph.calls == [("pump noise", "MP-2")]
    done = store.get(job.id)
    assert done is not None and done.status == "done"
    assert done.usage == {
        "llm_calls": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "cost_usd": pytest.approx((100 * 1.0 + 20 * 2.0) / 1_000_000),
    }
    assert done.steps[0] == {
        "node": "triage",
        "seconds": 0.25,
        "input_tokens": 100,
        "output_tokens": 20,
    }
    assert json.dumps(done.result)  # stored as plain JSON


@pytest.mark.parametrize(
    "error",
    [
        LlmError("connection refused"),
        StructuredOutputError("invalid Diagnosis after 2 attempts", reply="x", usage=Usage()),
        RuntimeError("bug"),
    ],
)
def test_run_job_stores_any_failure(store: JobStore, error: Exception) -> None:
    job = store.create({"description": "x"})
    run_job(store, FakeGraph(store, error), job.id)
    failed = store.get(job.id)
    assert failed is not None and failed.status == "failed"
    assert str(error) in (failed.error or "")
    assert type(error).__name__ in (failed.error or "")


def test_usage_without_prices_has_no_cost(store: JobStore) -> None:
    job = store.create({"description": "x"})
    run_job(store, FakeGraph(store), job.id)
    done: Any = store.get(job.id)
    assert done.usage["cost_usd"] is None
