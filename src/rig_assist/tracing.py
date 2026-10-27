"""Optional tracing of /diagnose runs in Langfuse: one trace per job, a span per graph
node, a generation per LLM call and a retriever span per search.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: `NullTracer` then does
nothing. Code depends only on the small `Tracer` interface, so tests use a fake.

Langfuse's SDK is built on OpenTelemetry: the current span lives in a `contextvars`
context, which is how a node's LLM calls nest under it (LangGraph copies the context
into the threads it runs nodes on). Spans are exported in the background; `flush()`
sends what's left (the app calls it on shutdown).
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal, Protocol

from langfuse import Langfuse

from rig_assist.config import Settings
from rig_assist.llm.models import LlmClient, LlmError, LlmResponse, Message, Usage
from rig_assist.rag.prompt import citation_label
from rig_assist.retrieval.models import Retriever, ScoredChunk

ObservationType = Literal["span", "agent", "chain", "retriever", "generation", "guardrail"]
Level = Literal["DEBUG", "DEFAULT", "WARNING", "ERROR"]


class Observation(Protocol):
    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: Usage | None = None,
        level: Level | None = None,
        status_message: str | None = None,
    ) -> None: ...


class Tracer(Protocol):
    def observe(
        self, name: str, as_type: ObservationType, input: Any = None, metadata: Any = None
    ) -> AbstractContextManager[Observation]: ...

    def current_trace_url(self) -> str | None:
        """The link to the trace being recorded, inside an `observe` block."""
        ...

    def flush(self) -> None: ...


# --- off ---------------------------------------------------------------------


class _NoObservation:
    def update(self, **fields: Any) -> None:
        pass


class NullTracer:
    @contextmanager
    def observe(
        self, name: str, as_type: ObservationType, input: Any = None, metadata: Any = None
    ) -> Iterator[Observation]:
        yield _NoObservation()

    def current_trace_url(self) -> str | None:
        return None

    def flush(self) -> None:
        pass


# --- Langfuse ----------------------------------------------------------------


class _LangfuseObservation:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: Usage | None = None,
        level: Level | None = None,
        status_message: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "output": output,
            "model": model,
            "level": level,
            "status_message": status_message,
        }
        if usage is not None:  # Langfuse's names for input and output tokens
            fields["usage_details"] = {"input": usage.input_tokens, "output": usage.output_tokens}
        self._observation.update(**{k: v for k, v in fields.items() if v is not None})


class LangfuseTracer:
    def __init__(self, client: Langfuse) -> None:
        self.client = client

    @contextmanager
    def observe(
        self, name: str, as_type: ObservationType, input: Any = None, metadata: Any = None
    ) -> Iterator[Observation]:
        with self.client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=metadata
        ) as observation:
            yield _LangfuseObservation(observation)

    def current_trace_url(self) -> str | None:
        trace_id = self.client.get_current_trace_id()
        return self.client.get_trace_url(trace_id=trace_id) if trace_id else None

    def flush(self) -> None:
        self.client.flush()


def make_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return NullTracer()
    assert settings.langfuse_secret_key is not None  # langfuse_enabled checked both keys
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.app_env,
    )
    return LangfuseTracer(client)


# --- wrappers ----------------------------------------------------------------


class TracedLlm:
    """An LlmClient whose calls are recorded as generations (prompt, reply, tokens)."""

    def __init__(self, llm: LlmClient, tracer: Tracer) -> None:
        self.llm = llm
        self.tracer = tracer
        self.model = llm.model

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse:
        prompt = [{"role": m.role, "content": m.content} for m in messages]
        metadata = {"max_tokens": max_tokens}
        with self.tracer.observe("llm", "generation", input=prompt, metadata=metadata) as obs:
            try:
                response = self.llm.complete(messages, max_tokens)
            except LlmError as error:
                obs.update(level="ERROR", status_message=str(error))
                raise
            obs.update(output=response.text, model=response.model, usage=response.usage)
            return response


class TracedRetriever:
    """A Retriever whose searches are recorded (query, k, the pages found)."""

    def __init__(self, retriever: Retriever, tracer: Tracer) -> None:
        self.retriever = retriever
        self.tracer = tracer

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        with self.tracer.observe("search", "retriever", input={"query": query, "k": k}) as obs:
            hits = self.retriever.search(query, k)
            obs.update(output=[citation_label(h.chunk.doc_id, h.chunk.page) for h in hits])
            return hits
