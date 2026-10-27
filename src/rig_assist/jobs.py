"""Background jobs for `/diagnose`: a SQLite job store and the runner.

Lifecycle: queued → running → done | failed, enforced by the store. Jobs run in the API
process (FastAPI background tasks), so a restart kills running jobs: at startup the app
marks every unfinished job failed (`fail_unfinished`) instead of leaving it "running"
forever. At scale, a real queue (Celery + Redis, Azure Service Bus) with separate
workers would replace this: jobs survive restarts and workers scale independently.

SQLite (standard library) is enough for one API process; each call opens its own
connection, so FastAPI's worker threads can use the store at the same time.
"""

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from rig_assist.agents.graph import GraphResult
from rig_assist.llm.usage import Prices, cost_usd

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
    finished_at TEXT
)
"""


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
    ) -> None:
        self._move(
            job_id,
            "done",
            ("running",),
            result=json.dumps(result),
            usage=json.dumps(usage),
            steps=json.dumps(steps),
            finished_at=_now(),
        )

    def fail(self, job_id: str, error: str) -> None:
        self._move(job_id, "failed", ("queued", "running"), error=error, finished_at=_now())

    def fail_unfinished(self) -> int:
        """At startup: jobs queued or running were lost with the old process."""
        with self._connect() as db:
            return db.execute(
                "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? "
                "WHERE status IN ('queued', 'running')",
                (INTERRUPTED, _now()),
            ).rowcount


class Graph(Protocol):
    def run(self, description: str, equipment_id: str | None = None) -> GraphResult: ...


def run_job(store: JobStore, graph: Graph, job_id: str, prices: Prices | None = None) -> None:
    """Run one job to the end. Never raises: any error is stored on the job."""
    job = store.get(job_id)
    if job is None:
        raise ValueError(f"unknown job {job_id}")
    store.start(job_id)
    try:
        result = graph.run(job.request["description"], job.request.get("equipment_id"))
    except Exception as error:  # noqa: BLE001  any failure becomes the job's error
        store.fail(job_id, f"{type(error).__name__}: {error}")
        return
    usage = result.usage
    store.finish(
        job_id,
        result.work_order.model_dump(mode="json"),
        {
            "llm_calls": result.llm_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost_usd(usage, prices),
        },
        [
            {
                "node": step.node,
                "seconds": step.seconds,
                "input_tokens": step.usage.input_tokens,
                "output_tokens": step.usage.output_tokens,
            }
            for step in result.steps
        ],
    )
