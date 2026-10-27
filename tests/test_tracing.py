"""Tracing: off by default (NullTracer), Langfuse when keys are set. A fake tracer records
the span tree with contextvars, the same way OpenTelemetry nests spans, so these tests
check that LLM calls and searches land under their node even inside LangGraph."""

import contextvars
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_agent_graph import DESCRIPTION, DIAGNOSIS, PLAN_DONE, Retriever, safety, triage

from rig_assist.agents.graph import TroubleshootingGraph
from rig_assist.config import Settings
from rig_assist.jobs import JobStore, run_job
from rig_assist.llm.fake import FakeLlm
from rig_assist.llm.models import LlmError, Message, Usage
from rig_assist.tracing import (
    LangfuseTracer,
    NullTracer,
    TracedLlm,
    TracedRetriever,
    make_tracer,
)

# --- a recording tracer ------------------------------------------------------


@dataclass
class Span:
    name: str
    as_type: str
    parent: str | None
    input: Any
    updates: dict[str, Any] = field(default_factory=dict)


class FakeTracer:
    def __init__(self, url: str | None = "https://langfuse.example/trace/t1") -> None:
        self.spans: list[Span] = []
        self.url = url
        self._current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            "span", default=None
        )

    @contextmanager
    def observe(
        self, name: str, as_type: str, input: Any = None, metadata: Any = None
    ) -> Iterator[Any]:
        parent = self._current.get()
        span = Span(name, as_type, parent.name if parent else None, input)
        self.spans.append(span)
        token = self._current.set(span)
        try:
            yield _Updater(span)
        finally:
            self._current.reset(token)

    def current_trace_url(self) -> str | None:
        return self.url if self._current.get() is not None else None

    def flush(self) -> None:
        pass

    def tree(self) -> list[tuple[str, str | None]]:
        return [(s.name, s.parent) for s in self.spans]


class _Updater:
    def __init__(self, span: Span) -> None:
        self.span = span

    def update(self, **fields: Any) -> None:
        self.span.updates |= fields


# --- null tracer and wrappers ------------------------------------------------


def test_the_null_tracer_does_nothing() -> None:
    tracer = NullTracer()
    with tracer.observe("x", "span", input={"a": 1}) as span:
        span.update(output="y", usage=Usage(1, 2))
    assert tracer.current_trace_url() is None
    tracer.flush()


def test_traced_llm_records_a_generation() -> None:
    tracer = FakeTracer()
    llm = TracedLlm(FakeLlm(["hello there"], model="qwen"), tracer)
    response = llm.complete([Message("user", "hi")], max_tokens=50)
    assert response.text == "hello there"
    assert llm.model == "qwen"
    [span] = tracer.spans
    assert (span.name, span.as_type) == ("llm", "generation")
    assert span.input == [{"role": "user", "content": "hi"}]
    assert span.updates == {"output": "hello there", "model": "qwen", "usage": Usage(1, 2)}


def test_a_failed_llm_call_is_marked_and_still_raised() -> None:
    tracer = FakeTracer()
    with pytest.raises(LlmError):
        TracedLlm(FakeLlm([]), tracer).complete([Message("user", "hi")])
    assert tracer.spans[0].updates["level"] == "ERROR"
    assert "no scripted reply" in tracer.spans[0].updates["status_message"]


def test_traced_retriever_records_the_query_and_the_pages_found() -> None:
    tracer = FakeTracer()
    hits = TracedRetriever(Retriever(), tracer).search("liner wear", 5)
    assert len(hits) == 2
    [span] = tracer.spans
    assert (span.name, span.as_type) == ("search", "retriever")
    assert span.input == {"query": "liner wear", "k": 5}
    assert span.updates["output"] == ["[pump p.12]", "[pump p.14]"]


# --- the graph and the job ---------------------------------------------------


def test_a_job_is_one_trace_with_nodes_llm_calls_and_searches_nested(tmp_path: Path) -> None:
    tracer = FakeTracer()
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create({"description": DESCRIPTION, "equipment_id": None})
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    graph = TroubleshootingGraph(llm, Retriever(), tracer=tracer)
    run_job(store, graph, job.id, tracer=tracer)

    assert tracer.tree() == [
        ("diagnose", None),
        ("triage", "diagnose"),
        ("llm", "triage"),
        ("retrieval", "diagnose"),
        ("search", "retrieval"),
        ("llm", "retrieval"),
        ("diagnosis", "diagnose"),
        ("llm", "diagnosis"),
        ("safety", "diagnose"),
        ("search", "safety"),
        ("llm", "safety"),
        ("report", "diagnose"),
    ]
    types = {s.name: s.as_type for s in tracer.spans}
    assert types["diagnose"] == "agent" and types["safety"] == "guardrail"
    assert types["triage"] == "chain"
    done = store.get(job.id)
    assert done is not None and done.trace_url == "https://langfuse.example/trace/t1"


def test_the_root_span_records_the_request_and_the_work_order(tmp_path: Path) -> None:
    tracer = FakeTracer()
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create({"description": DESCRIPTION, "equipment_id": "MP-2"})
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    run_job(store, TroubleshootingGraph(llm, Retriever(), tracer=tracer), job.id, tracer=tracer)
    root = tracer.spans[0]
    assert root.input == {"description": DESCRIPTION, "equipment_id": "MP-2"}
    assert root.updates["output"]["blocked_actions"] == [
        "Open the fluid end to check valves",
        "Bypass the relief valve",
    ]


def test_a_failed_job_keeps_its_trace_url_and_marks_the_root(tmp_path: Path) -> None:
    tracer = FakeTracer()
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create({"description": DESCRIPTION})
    graph = TroubleshootingGraph(FakeLlm(["bad", "bad"]), Retriever(), tracer=tracer)
    run_job(store, graph, job.id, tracer=tracer)
    failed = store.get(job.id)
    assert failed is not None and failed.status == "failed"
    assert failed.trace_url == "https://langfuse.example/trace/t1"
    assert tracer.spans[0].updates["level"] == "ERROR"


def test_without_a_tracer_the_job_has_no_trace_url(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create({"description": DESCRIPTION})
    llm = FakeLlm([triage(), PLAN_DONE, DIAGNOSIS, safety()])
    run_job(store, TroubleshootingGraph(llm, Retriever()), job.id)
    assert store.get(job.id).trace_url is None  # type: ignore[union-attr]


def test_an_existing_job_table_gets_the_trace_url_column(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:  # the table as the first version created it
        db.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, request TEXT NOT "
            "NULL, result TEXT, error TEXT, usage TEXT, steps TEXT NOT NULL DEFAULT '[]', "
            "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT)"
        )
        db.execute(
            "INSERT INTO jobs (id, status, request, created_at) VALUES ('old', 'done', ?, 'x')",
            (json.dumps({"description": "x"}),),
        )
    store = JobStore(path)
    assert store.get("old").trace_url is None  # type: ignore[union-attr]
    assert store.create({"description": "y"}).trace_url is None


# --- Langfuse adapter --------------------------------------------------------


class FakeLangfuse:
    """The few Langfuse client methods the adapter uses."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[Any]:
        self.started.append(kwargs)
        outer = self

        class Observation:
            def update(self, **fields: Any) -> None:
                outer.updates.append(fields)

        yield Observation()

    def get_current_trace_id(self) -> str | None:
        return "abc123"

    def get_trace_url(self, *, trace_id: str | None = None) -> str | None:
        return f"https://cloud.langfuse.com/project/p/traces/{trace_id}"

    def flush(self) -> None:
        self.flushed = True


def test_the_langfuse_adapter_maps_usage_and_types() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)  # type: ignore[arg-type]
    with tracer.observe("llm", "generation", input=[{"role": "user"}], metadata={"k": 1}) as obs:
        obs.update(output="hi", model="qwen", usage=Usage(10, 3))
        url = tracer.current_trace_url()
    assert client.started == [
        {"name": "llm", "as_type": "generation", "input": [{"role": "user"}], "metadata": {"k": 1}}
    ]
    assert client.updates == [
        {"output": "hi", "model": "qwen", "usage_details": {"input": 10, "output": 3}}
    ]
    assert url == "https://cloud.langfuse.com/project/p/traces/abc123"
    tracer.flush()
    assert client.flushed


def test_make_tracer_is_off_without_keys() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(make_tracer(settings), NullTracer)


def test_make_tracer_uses_langfuse_when_both_keys_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, Any]] = []

    class Langfuse:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)

    monkeypatch.setattr("rig_assist.tracing.Langfuse", Langfuse)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-1",
        langfuse_base_url="https://us.cloud.langfuse.com",
    )
    assert isinstance(make_tracer(settings), LangfuseTracer)
    assert made == [
        {
            "public_key": "pk-lf-1",
            "secret_key": "sk-lf-1",
            "base_url": "https://us.cloud.langfuse.com",
            "environment": "dev",
        }
    ]


def test_one_key_alone_is_an_error() -> None:
    with pytest.raises(ValueError, match="LANGFUSE"):
        Settings(_env_file=None, langfuse_public_key="pk-lf-1")  # type: ignore[call-arg]
