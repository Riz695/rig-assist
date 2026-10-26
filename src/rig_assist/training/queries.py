"""Synthetic training queries: an LLM writes questions that each chunk answers.

These become the (query, chunk) pairs the embedder is fine-tuned on (`pairs.py`). The prompt
asks for the kinds of questions a technician would type, deliberately worded differently from
the manual, because that is where the base model is weak (Phase 3: paraphrased questions).

Replies are strict JSON, validated; an invalid one gets one retry that says what was wrong,
then the chunk is recorded as "unparseable" (no pairs, counted). Requests run in parallel
(`workers`) and each result is appended to a JSON Lines cache as it arrives, so a stopped run
resumes. A cached record is reused only for the same config (model, prompt hash, count) and
the same chunk text (a re-chunked index gets new queries).
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from rig_assist.ingestion.models import Chunk
from rig_assist.llm import LlmClient, Message, Usage
from rig_assist.llm.json_reply import parse_json_object

QUERIES_PER_CHUNK = 3
QUERY_MAX_TOKENS = 1024  # three short questions; thinking is switched off on the server
MAX_QUERY_CHARS = 300

# Part of the training data's definition: its hash is in the cache config and the MLflow run.
QUERY_SYSTEM_PROMPT = """\
You write search queries for training a search engine over offshore drilling and oil & gas \
equipment manuals. You get one excerpt from a manual. Write questions that a rig technician, \
driller or marine crew member might type when looking for the information in this excerpt.

Rules:
- Each question must be answerable from this excerpt alone.
- Vary them: one describes a situation or symptom in the technician's own words; one is a \
short keyword-style search; one asks for a specific fact (a value, a step, a limit, a who \
or when).
- Use your own words. Avoid copying phrases from the excerpt; use the synonyms and plain \
terms people use on a rig. Keep codes, part names and units where a technician would use them.
- Do not mention "the excerpt", "the text", "this section" or "the manual".
- Write in English.

Reply with only this JSON object, no other text:
{"queries": ["<question 1>", "<question 2>", ...]}"""

RETRY_PROMPT = (
    "That reply was not valid: {error}. Reply again with only the JSON object "
    '{{"queries": [...]}} holding exactly {n} non-empty questions.'
)

Outcome = Literal["ok", "unparseable"]
Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
]


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    queries: list[Query]


class QueryReplyError(ValueError):
    """The LLM's reply is not a valid list of queries."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def query_messages(chunk: Chunk, n: int = QUERIES_PER_CHUNK) -> list[Message]:
    lines = [f"Manual: {chunk.doc_title}"]
    if chunk.section:
        lines.append(f"Section: {chunk.section}")
    lines += [f"Page: {chunk.page}", "", "Excerpt:", chunk.text, "", f"Write {n} questions."]
    return [Message("system", QUERY_SYSTEM_PROMPT), Message("user", "\n".join(lines))]


def parse_queries(text: str, n: int = QUERIES_PER_CHUNK) -> list[str]:
    try:
        reply = _Reply.model_validate(parse_json_object(text))
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise QueryReplyError(problems) from error
    except ValueError as error:
        raise QueryReplyError(str(error)) from error
    if len(reply.queries) != n:
        raise QueryReplyError(f"expected {n} queries, got {len(reply.queries)}")
    return reply.queries


@dataclass(frozen=True)
class QueryRecord:
    chunk_id: str
    text_sha256: str  # of the chunk text the queries were written for
    queries: list[str]  # empty when unparseable
    outcome: Outcome
    raw_reply: str | None  # the last reply when unparseable, for debugging
    attempts: int
    input_tokens: int
    output_tokens: int
    config: Mapping[str, str]


def _record(
    chunk: Chunk,
    queries: list[str],
    outcome: Outcome,
    raw_reply: str | None,
    attempts: int,
    usage: Usage,
    config: Mapping[str, str],
) -> QueryRecord:
    return QueryRecord(
        chunk_id=chunk.chunk_id,
        text_sha256=text_sha256(chunk.text),
        queries=queries,
        outcome=outcome,
        raw_reply=raw_reply,
        attempts=attempts,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        config=dict(config),
    )


def generate_for_chunk(
    llm: LlmClient, chunk: Chunk, config: Mapping[str, str], n: int = QUERIES_PER_CHUNK
) -> QueryRecord:
    """Ask for `n` queries; one retry on an invalid reply. LLM errors propagate."""
    messages = query_messages(chunk, n)
    usage = Usage()
    for attempt in (1, 2):
        response = llm.complete(messages, max_tokens=QUERY_MAX_TOKENS)
        usage += response.usage
        try:
            queries = parse_queries(response.text, n)
        except QueryReplyError as error:
            messages = [
                *messages,
                Message("assistant", response.text),
                Message("user", RETRY_PROMPT.format(error=error, n=n)),
            ]
            continue
        return _record(chunk, queries, "ok", None, attempt, usage, config)
    return _record(chunk, [], "unparseable", response.text, 2, usage, config)


class QueryCache:
    """Query records in a JSON Lines file, one per chunk, appended as they're made."""

    def __init__(self, path: Path, config: Mapping[str, str]) -> None:
        self.path = path
        self.config = dict(config)

    def load(self) -> dict[str, QueryRecord]:
        """Records made with this config, by chunk_id (the last one wins)."""
        if not self.path.exists():
            return {}
        records = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            if data["config"] == self.config:
                records[data["chunk_id"]] = QueryRecord(**data)
        return records

    def append(self, record: QueryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def generate_queries(
    llm: LlmClient,
    chunks: Iterable[Chunk],
    cache: QueryCache,
    n: int = QUERIES_PER_CHUNK,
    workers: int = 8,
    progress: Callable[[str], None] = lambda line: None,
) -> dict[str, QueryRecord]:
    """A record per chunk, by chunk_id; only chunks without a current cached record are asked.

    Up to `workers` requests run at once. On an LLM error, queued chunks are cancelled, the
    ones already finished stay cached, and the error propagates (run again to continue).
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    chunks = list(chunks)
    cached = cache.load()
    done = {
        c.chunk_id: cached[c.chunk_id]
        for c in chunks
        if c.chunk_id in cached and cached[c.chunk_id].text_sha256 == text_sha256(c.text)
    }
    todo = [c for c in chunks if c.chunk_id not in done]
    progress(f"{len(done)} cached, {len(todo)} to generate")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_for_chunk, llm, c, cache.config, n) for c in todo]
        try:
            for finished, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                cache.append(record)  # only this thread writes the file
                done[record.chunk_id] = record
                if finished % 50 == 0 or finished == len(todo):
                    progress(f"{finished}/{len(todo)} generated")
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    return done


def queries_by_chunk(records: Mapping[str, QueryRecord]) -> dict[str, list[str]]:
    """chunk_id -> queries, for `build_pairs`; unparseable chunks give no pairs."""
    return {cid: list(r.queries) for cid, r in records.items() if r.outcome == "ok"}


def query_stats(records: Sequence[QueryRecord]) -> dict[str, int]:
    return {
        "chunks": len(records),
        "ok": sum(r.outcome == "ok" for r in records),
        "unparseable": sum(r.outcome == "unparseable" for r in records),
        "retried": sum(r.attempts == 2 for r in records),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
    }
