import json
import re
import threading
from pathlib import Path

import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.llm import FakeLlm, LlmError, LlmResponse, Message, Usage
from rig_assist.training.queries import (
    QUERY_MAX_TOKENS,
    QUERY_SYSTEM_PROMPT,
    QueryCache,
    QueryRecord,
    QueryReplyError,
    generate_for_chunk,
    generate_queries,
    parse_queries,
    queries_by_chunk,
    query_messages,
    text_sha256,
)

CONFIG = {"llm_model": "fake", "prompt_sha256": "abc", "queries_per_chunk": "3"}
TEXT = "Test the BOP stack every 14 days and record the result in the IADC report."


def chunk(page: int, text: str = TEXT, section: str | None = "4.2 BOP testing") -> Chunk:
    return Chunk(
        chunk_id=f"bop-manual::p{page}::c0",
        doc_id="bop-manual",
        doc_title="Bop Manual",
        page=page,
        section=section,
        parser="pdfplumber",
        text=text,
    )


def reply(*queries: str) -> str:
    return json.dumps({"queries": list(queries)})


GOOD = reply("How often is the BOP tested?", "BOP test interval", "Where do I log a BOP test?")


# --- Prompt -----------------------------------------------------------------


def test_prompt_asks_for_technician_questions_as_json() -> None:
    for word in ("technician", "JSON", "queries"):
        assert word in QUERY_SYSTEM_PROMPT


def test_messages_hold_the_title_section_text_and_count() -> None:
    messages = query_messages(chunk(12), n=3)
    assert [m.role for m in messages] == ["system", "user"]
    user = messages[1].content
    assert "Bop Manual" in user
    assert "4.2 BOP testing" in user
    assert TEXT in user
    assert "3" in user


def test_messages_without_a_section_leave_it_out() -> None:
    assert "Section" not in query_messages(chunk(12, section=None), n=3)[1].content


# --- parse_queries ------------------------------------------------------------


def test_parses_queries_and_trims_them() -> None:
    assert parse_queries(reply(" A? ", "B?", "C?"), n=3) == ["A?", "B?", "C?"]


def test_accepts_a_json_code_fence() -> None:
    assert len(parse_queries(f"```json\n{GOOD}\n```", n=3)) == 3


@pytest.mark.parametrize(
    "text",
    [
        "Here are three questions: ...",  # not JSON
        '["A?", "B?", "C?"]',  # not an object
        reply("A?", "B?"),  # too few
        reply("A?", "B?", "C?", "D?"),  # too many
        reply("A?", "", "C?"),  # blank
        json.dumps({"queries": ["A?", 2, "C?"]}),  # not a string
        json.dumps({"queries": ["A?", "B?", "C?"], "notes": "x"}),  # unknown field
        reply("A?", "B?", "x" * 301),  # too long to be a question
    ],
)
def test_rejects_invalid_replies(text: str) -> None:
    with pytest.raises(QueryReplyError):
        parse_queries(text, n=3)


# --- generate_for_chunk -------------------------------------------------------


def test_one_call_gives_an_ok_record() -> None:
    llm = FakeLlm(replies=[GOOD])
    rec = generate_for_chunk(llm, chunk(12), CONFIG, n=3)
    assert rec.outcome == "ok"
    assert rec.chunk_id == "bop-manual::p12::c0"
    assert rec.queries == [
        "How often is the BOP tested?",
        "BOP test interval",
        "Where do I log a BOP test?",
    ]
    assert rec.text_sha256 == text_sha256(TEXT)
    assert rec.attempts == 1
    assert rec.config == CONFIG
    assert llm.max_tokens == [QUERY_MAX_TOKENS]
    assert rec.input_tokens > 0 and rec.output_tokens > 0


def test_an_invalid_reply_is_retried_once_with_the_error() -> None:
    llm = FakeLlm(replies=["Sure! Questions:", GOOD])
    rec = generate_for_chunk(llm, chunk(12), CONFIG, n=3)
    assert (rec.outcome, rec.attempts) == ("ok", 2)
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    assert "JSON" in llm.calls[1][3].content


def test_two_invalid_replies_are_recorded_as_unparseable() -> None:
    llm = FakeLlm(replies=["nope", reply("only one?")])
    rec = generate_for_chunk(llm, chunk(12), CONFIG, n=3)
    assert rec.outcome == "unparseable"
    assert rec.queries == []
    assert rec.raw_reply == reply("only one?")
    assert len(llm.calls) == 2


# --- QueryCache ----------------------------------------------------------------


def test_cache_round_trips_and_ignores_other_configs(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    rec = generate_for_chunk(FakeLlm(replies=[GOOD]), chunk(12), CONFIG, n=3)
    QueryCache(path, CONFIG).append(rec)
    assert QueryCache(path, CONFIG).load() == {rec.chunk_id: rec}
    assert QueryCache(path, {**CONFIG, "prompt_sha256": "def"}).load() == {}
    assert QueryCache(tmp_path / "none.jsonl", CONFIG).load() == {}


# --- generate_queries ------------------------------------------------------------


class EchoLlm:
    """Replies with three questions naming the page in the prompt; safe across threads."""

    model = "echo"

    def __init__(self, fail_on_page: int | None = None) -> None:
        self.fail_on_page = fail_on_page
        self.pages: list[int] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse:
        page = int(re.search(r"Page: (\d+)", messages[-1].content).group(1))  # type: ignore[union-attr]
        with self._lock:
            self.pages.append(page)
        if page == self.fail_on_page:
            raise LlmError("HTTP 503")
        text = reply(*(f"Question {i} about page {page}?" for i in range(3)))
        return LlmResponse(text=text, usage=Usage(10, 5), model=self.model)


def test_generates_for_every_chunk_in_parallel_and_saves_each(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "queries.jsonl", CONFIG)
    chunks = [chunk(p) for p in range(1, 21)]
    records = generate_queries(EchoLlm(), chunks, cache, n=3, workers=4)
    assert set(records) == {c.chunk_id for c in chunks}
    assert records["bop-manual::p7::c0"].queries[0] == "Question 0 about page 7?"
    assert set(cache.load()) == set(records)


def test_cached_chunks_are_not_asked_again(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "queries.jsonl", CONFIG)
    generate_queries(EchoLlm(), [chunk(1), chunk(2)], cache, n=3, workers=2)
    llm = EchoLlm()
    lines: list[str] = []
    generate_queries(
        llm, [chunk(1), chunk(2), chunk(3)], cache, n=3, workers=2, progress=lines.append
    )
    assert llm.pages == [3]
    assert "2 cached, 1 to generate" in lines[0]


def test_a_chunk_whose_text_changed_is_generated_again(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "queries.jsonl", CONFIG)
    generate_queries(EchoLlm(), [chunk(1)], cache, n=3, workers=1)
    llm = EchoLlm()
    records = generate_queries(llm, [chunk(1, text="Re-chunked text.")], cache, n=3, workers=1)
    assert llm.pages == [1]
    assert records["bop-manual::p1::c0"].text_sha256 == text_sha256("Re-chunked text.")


def test_a_failure_keeps_finished_chunks_and_raises(tmp_path: Path) -> None:
    cache = QueryCache(tmp_path / "queries.jsonl", CONFIG)
    chunks = [chunk(p) for p in range(1, 6)]
    with pytest.raises(LlmError):
        generate_queries(EchoLlm(fail_on_page=3), chunks, cache, n=3, workers=1)
    assert set(cache.load()) == {"bop-manual::p1::c0", "bop-manual::p2::c0"}


def test_workers_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_queries(EchoLlm(), [chunk(1)], QueryCache(tmp_path / "q.jsonl", CONFIG), workers=0)


# --- queries_by_chunk --------------------------------------------------------------


def test_queries_by_chunk_keeps_only_ok_records() -> None:
    ok = QueryRecord("a", "h", ["A?"], "ok", None, 1, 1, 1, CONFIG)
    bad = QueryRecord("b", "h", [], "unparseable", "nope", 2, 1, 1, CONFIG)
    assert queries_by_chunk({"a": ok, "b": bad}) == {"a": ["A?"]}
