"""Background jobs for `/diagnose`: a SQLite job store and the runner.

Lifecycle: queued → running → done | failed, enforced by the store. Jobs run in the API
process (FastAPI background tasks), so a restart kills running jobs: at startup the app
marks every unfinished job failed (`fail_unfinished`) instead of leaving it "running"
forever. At scale, a real queue (Celery + Redis, Azure Service Bus) with separate
workers would replace this: jobs survive restarts and workers scale independently.

SQLite (standard library) is enough for one API process; each call opens its own
connection, so FastAPI's worker threads can use the store at the same time.

Live progress: while a job runs, the graph reports each node to the store, so
`current_node` names the node working now and `steps` grows as nodes finish. A UI polling
`GET /jobs/{id}` sees the graph advance. A failed job keeps the node it failed in.
"""

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from rig_assist.agents.graph import GraphResult, NodeName, Progress, Step
from rig_assist.llm.usage import Prices, cost_usd
from rig_assist.tracing import NullTracer, Tracer

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]
INTERRUPTED = "Interrupted by a restart of the API before it finished. Submit it again."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    result TEXT,
    error TEXT,
    usage TEXT,
    steps TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    trace_url TEXT,
    current_node TEXT
)
"""
# Columns added after the first version, for databases created before them.
_ADDED_COLUMNS = {"trace_url": "TEXT", "current_node": "TEXT"}


@dataclass(frozen=True)
class Job:
    id: str
    status: JobStatus
    request: dict[str, Any]
    result: dict[str, Any] | None  # the WorkOrder as JSON, when done
    error: str | None  # when failed
    usage: dict[str, Any] | None  # llm_calls, tokens, cost, when done
    steps: list[dict[str, Any]]  # per node: seconds and tokens
    created_at: str  # ISO 8601, UTC
    started_at: str | None
    finished_at: str | None
    trace_url: str | None  # the run's trace, when tracing is on
    current_node: str | None  # the node working now; when failed, the one that failed


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commits, or rolls back on an exception
                connection.execute(_SCHEMA)
                columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
                for name, kind in _ADDED_COLUMNS.items():
                    if name not in columns:
                        connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            status=cast(JobStatus, row["status"]),
            request=json.loads(row["request"]),
            result=_loads(row["result"]),
            error=row["error"],
            usage=_loads(row["usage"]),
            steps=json.loads(row["steps"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            trace_url=row["trace_url"],
            current_node=row["current_node"],
        )

    def create(self, request: Mapping[str, Any]) -> Job:
        job_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs (id, status, request, created_at) VALUES (?, 'queued', ?, ?)",
                (job_id, json.dumps(dict(request)), _now()),
            )
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else self._job(row)

    def all(self) -> list[Job]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at, id").fetchall()
        return [self._job(row) for row in rows]

    def _update_running(self, job_id: str, sql_set: str, params: tuple[Any, ...]) -> None:
        """Apply `SET sql_set` only while the job is running (live progress)."""
        with self._connect() as db:
            updated = db.execute(
                f"UPDATE jobs SET {sql_set} WHERE id = ? AND status = 'running'",
                (*params, job_id),
            ).rowcount
        if updated == 0:
            job = self.get(job_id)
            current = "unknown" if job is None else job.status
            raise ValueError(f"cannot record progress on job {job_id}: it is {current}")

    def node_started(self, job_id: str, node: str) -> None:
        self._update_running(job_id, "current_node = ?", (node,))

    def node_finished(self, job_id: str, step: Mapping[str, Any]) -> None:
        # json_insert(..., '$[#]', ...) appends to the JSON array inside SQLite, in one
        # statement: no read-modify-write race with another writer.
        self._update_running(
            job_id,
            "steps = json_insert(steps, '$[#]', json(?)), current_node = NULL",
            (json.dumps(dict(step)),),
        )

    def _move(
        self, job_id: str, to: JobStatus, allowed_from: tuple[JobStatus, ...], **fields: Any
    ) -> None:
        """Set `to` (and `fields`) only if the job is in an allowed status."""
        columns = ", ".join(f"{name} = ?" for name in fields)
        placeholders = ", ".join("?" for _ in allowed_from)
        with self._connect() as db:
            updated = db.execute(
                # Column names come from this module's own keyword arguments, never input.
                f"UPDATE jobs SET status = ?, {columns} "
                f"WHERE id = ? AND status IN ({placeholders})",
                (to, *fields.values(), job_id, *allowed_from),
            ).rowcount
        if updated == 0:
            job = self.get(job_id)
            current = "unknown" if job is None else job.status
            raise ValueError(f"cannot move job {job_id} from {current} to {to}")

    def start(self, job_id: str) -> None:
        self._move(job_id, "running", ("queued",), started_at=_now())

    def finish(
        self,
        job_id: str,
        result: Mapping[str, Any],
        usage: Mapping[str, Any],
        steps: list[dict[str, Any]],
        trace_url: str | None = None,
    ) -> None:
        self._move(
            job_id,
            "done",
            ("running",),
            result=json.dumps(result),
            usage=json.dumps(usage),
            steps=json.dumps(steps),
            finished_at=_now(),
            trace_url=trace_url,
            current_node=None,
        )

    def fail(self, job_id: str, error: str, trace_url: str | None = None) -> None:
        self._move(
            job_id,
            "failed",
            ("queued", "running"),
            error=error,
            finished_at=_now(),
            trace_url=trace_url,
        )

    def fail_unfinished(self) -> int:
        """At startup: jobs queued or running were lost with the old process."""
        with self._connect() as db:
            return db.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
                "WHERE status IN ('queued', 'running')",
                (INTERRUPTED, _now()),
            ).rowcount


class Graph(Protocol):
    def run(
        self, description: str, equipment_id: str | None = None, progress: Progress | None = None
    ) -> GraphResult: ...


def step_json(step: Step) -> dict[str, Any]:
    """A step as stored on the job and returned by the API."""
    return {
        "node": step.node,
        "seconds": step.seconds,
        "input_tokens": step.usage.input_tokens,
        "output_tokens": step.usage.output_tokens,
    }


class _StoreProgress:
    """Writes the graph's progress to the job. Progress is a nicety: a failed write
    (e.g. the database is locked) is logged and the job carries on."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self.store = store
        self.job_id = job_id

    def node_started(self, node: NodeName) -> None:
        try:
            self.store.node_started(self.job_id, node)
        except (sqlite3.Error, ValueError) as error:
            logger.warning("job %s: progress not recorded: %s", self.job_id, error)

    def node_finished(self, step: Step) -> None:
        try:
            self.store.node_finished(self.job_id, step_json(step))
        except (sqlite3.Error, ValueError) as error:
            logger.warning("job %s: progress not recorded: %s", self.job_id, error)


def run_job(
    store: JobStore,
    graph: Graph,
    job_id: str,
    prices: Prices | None = None,
    tracer: Tracer | None = None,
) -> None:
    """Run one job to the end, as one trace. Never raises: any error is stored on the job."""
    job = store.get(job_id)
    if job is None:
        raise ValueError(f"unknown job {job_id}")
    tracer = tracer or NullTracer()
    store.start(job_id)
    with tracer.observe(
        "diagnose", "agent", input=job.request, metadata={"job_id": job_id}
    ) as root:
        trace_url = tracer.current_trace_url()
        try:
            result = graph.run(
                job.request["description"],
                job.request.get("equipment_id"),
                progress=_StoreProgress(store, job_id),
            )
        except Exception as error:  # noqa: BLE001  any failure becomes the job's error
            message = f"{type(error).__name__}: {error}"
            root.update(level="ERROR", status_message=message)
            store.fail(job_id, message, trace_url)
            return
        order = result.work_order.model_dump(mode="json")
        root.update(output=order, usage=result.usage)
    usage = result.usage
    store.finish(
        job_id,
        order,
        {
            "llm_calls": result.llm_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(usage, prices),
        },
        [step_json(step) for step in result.steps],
        trace_url,
    )
