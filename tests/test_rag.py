from collections.abc import Callable, Iterator

import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.llm import FakeLlm, Prices, Usage
from rig_assist.rag import (
    MAX_ANSWER_TOKENS,
    NOT_FOUND,
    SNIPPET_CHARS,
    SYSTEM_PROMPT,
    Citation,
    RagPipeline,
    build_messages,
    extract_citations,
)
from rig_assist.retrieval.models import ScoredChunk


def hit(doc_id: str, page: int, text: str, index: int = 0, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}::p{page}::c{index}",
        doc_id=doc_id,
        doc_title=doc_id.replace("-", " ").title(),
        page=page,
        section="4.2 Pressure tests" if doc_id == "bop-manual" else None,
        parser="pdfplumber",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=score)


HITS = [
    hit("bop-manual", 12, "Pressure test the BOP stack every 14 days."),
    hit("safety-manual", 61, "H2S alarms trigger at 10 ppm."),
]


class FakeRetriever:
    """Returns fixed hits and records the queries it was asked."""

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.queries.append((query, k))
        return self.hits[:k]


def fake_clock(*times: float) -> Callable[[], float]:
    ticks: Iterator[float] = iter(times)
    return lambda: next(ticks)


def pipeline(
    hits: list[ScoredChunk],
    replies: list[str],
    **kwargs: object,
) -> tuple[RagPipeline, FakeLlm, FakeRetriever]:
    retriever = FakeRetriever(hits)
    llm = FakeLlm(replies=replies)
    return RagPipeline(retriever, llm, **kwargs), llm, retriever  # type: ignore[arg-type]


# --- Prompt -------------------------------------------------------------


def test_messages_are_the_system_prompt_then_the_question_with_context() -> None:
    messages = build_messages("How often is the BOP tested?", HITS)
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    assert "How often is the BOP tested?" in messages[1].content


def test_each_context_block_starts_with_its_citation_label() -> None:
    user = build_messages("q", HITS)[1].content
    assert "[bop-manual p.12]" in user
    assert "Pressure test the BOP stack every 14 days." in user
    assert user.index("[bop-manual p.12]") < user.index("Pressure test the BOP")
    assert user.index("[bop-manual p.12]") < user.index("[safety-manual p.61]")


def test_context_blocks_name_the_document_and_section() -> None:
    user = build_messages("q", HITS)[1].content
    assert "Bop Manual" in user
    assert "4.2 Pressure tests" in user


def test_context_has_no_numbered_labels_the_model_could_copy() -> None:
    assert "[1]" not in build_messages("q", HITS)[1].content


def test_system_prompt_demands_grounding_citations_and_the_not_found_reply() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered
    assert "cite" in lowered
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "[example-manual p.7]" in SYSTEM_PROMPT  # a worked example of the format


def test_not_found_says_so_in_plain_words() -> None:
    assert "not found in the manuals" in NOT_FOUND.lower()


# --- Citation parsing ---------------------------------------------------


def test_extracts_citations_in_order_of_first_appearance_without_duplicates() -> None:
    answer = (
        "Test every 14 days [bop-manual p.12]. "
        "Alarm at 10 ppm [safety-manual p.61] [bop-manual p.12]."
    )
    assert extract_citations(answer) == [("bop-manual", 12), ("safety-manual", 61)]


def test_extracts_several_citations_in_one_bracket() -> None:
    answer = (
        "Both apply [bop-manual p.12; safety-manual p.61] and [bop-manual p.13, bop-manual p.14]."
    )
    assert extract_citations(answer) == [
        ("bop-manual", 12),
        ("safety-manual", 61),
        ("bop-manual", 13),
        ("bop-manual", 14),
    ]


def test_tolerates_a_space_after_p_dot() -> None:
    assert extract_citations("Weekly [bop-manual p. 12].") == [("bop-manual", 12)]


@pytest.mark.parametrize(
    "answer", ["Weekly [1].", "Weekly [sic].", "Weekly (bop-manual p.12).", ""]
)
def test_ignores_anything_that_is_not_a_label(answer: str) -> None:
    assert extract_citations(answer) == []


# --- Pipeline: answered -------------------------------------------------


def test_answer_with_valid_citations_is_returned_with_them() -> None:
    answer = "Every 14 days [bop-manual p.12]."
    rag, llm, retriever = pipeline(HITS, [answer])
    result = rag.ask("How often is the BOP tested?")

    assert result.found is True
    assert result.answer == answer
    assert result.citations == [
        Citation(
            doc_id="bop-manual",
            doc_title="Bop Manual",
            page=12,
            snippet="Pressure test the BOP stack every 14 days.",
        )
    ]
    assert result.refusal_reason is None
    assert retriever.queries == [("How often is the BOP tested?", 5)]
    assert llm.calls == [build_messages("How often is the BOP tested?", HITS)]


def test_citations_follow_the_answer_order_not_the_retrieval_order() -> None:
    rag, _, _ = pipeline(HITS, ["Alarm [safety-manual p.61]. Test [bop-manual p.12]."])
    pages = [(c.doc_id, c.page) for c in rag.ask("q").citations]
    assert pages == [("safety-manual", 61), ("bop-manual", 12)]


def test_snippet_comes_from_the_best_ranked_chunk_on_the_cited_page() -> None:
    hits = [
        hit("bop-manual", 12, "Best chunk on page 12.", index=1),
        hit("bop-manual", 12, "Second chunk on page 12.", index=0),
    ]
    rag, _, _ = pipeline(hits, ["Yes [bop-manual p.12]."])
    assert rag.ask("q").citations[0].snippet == "Best chunk on page 12."


def test_long_snippets_are_cut_and_whitespace_collapsed() -> None:
    text = "Line one\n\n   line two " + "x" * 1000
    rag, _, _ = pipeline([hit("bop-manual", 12, text)], ["Yes [bop-manual p.12]."])
    snippet = rag.ask("q").citations[0].snippet
    assert snippet.startswith("Line one line two x")
    assert len(snippet) == SNIPPET_CHARS
    assert snippet.endswith("…")


def test_the_answer_cap_leaves_room_for_hidden_thinking() -> None:
    # Gemini thought 600-900 tokens before a ~200-token answer; 1,024 cut answers off.
    rag, llm, _ = pipeline(HITS, ["Yes [bop-manual p.12]."])
    rag.ask("q")
    assert llm.max_tokens == [MAX_ANSWER_TOKENS]
    assert MAX_ANSWER_TOKENS >= 4096


def test_top_k_controls_how_many_chunks_reach_the_prompt() -> None:
    rag, llm, retriever = pipeline(HITS, ["Yes [bop-manual p.12]."], top_k=1)
    rag.ask("q")
    assert retriever.queries == [("q", 1)]
    assert "[safety-manual p.61]" not in llm.calls[0][1].content


# --- Pipeline: "not found" ----------------------------------------------


def test_nothing_retrieved_is_not_found_without_calling_the_llm() -> None:
    rag, llm, _ = pipeline([], [])
    result = rag.ask("How do I calibrate the flux capacitor?")
    assert result.found is False
    assert result.answer == NOT_FOUND
    assert result.citations == []
    assert result.refusal_reason == "no_results"
    assert llm.calls == []
    assert result.usage == Usage()


def test_relevance_gate_rejecting_the_hits_skips_the_llm() -> None:
    rag, llm, _ = pipeline(HITS, [], is_relevant=lambda hits: False)
    result = rag.ask("q")
    assert (result.found, result.refusal_reason) == (False, "not_relevant")
    assert llm.calls == []


def test_relevance_gate_sees_the_retrieved_hits() -> None:
    seen: list[list[ScoredChunk]] = []

    def gate(hits: list[ScoredChunk]) -> bool:
        seen.append(hits)
        return True

    rag, _, _ = pipeline(HITS, ["Yes [bop-manual p.12]."], is_relevant=gate)
    rag.ask("q")
    assert seen == [HITS]


@pytest.mark.parametrize(
    "reply",
    [NOT_FOUND, "not found in the manuals.", f"  {NOT_FOUND} The context covers only BOP tests."],
)
def test_llm_saying_not_found_is_not_found(reply: str) -> None:
    rag, _, _ = pipeline(HITS, [reply])
    result = rag.ask("q")
    assert (result.found, result.answer, result.refusal_reason) == (
        False,
        NOT_FOUND,
        "llm_not_found",
    )
    assert result.citations == []


def test_an_answer_that_mentions_a_gap_later_is_still_an_answer() -> None:
    reply = "Every 14 days [bop-manual p.12]. The test pressure is not found in the manuals."
    rag, _, _ = pipeline(HITS, [reply])
    assert rag.ask("q").found is True


def test_answer_without_citations_is_not_found() -> None:
    rag, _, _ = pipeline(HITS, ["Every 14 days."])
    result = rag.ask("q")
    assert (result.found, result.answer, result.refusal_reason) == (
        False,
        NOT_FOUND,
        "no_citations",
    )


@pytest.mark.parametrize(
    "reply",
    [
        "Every 14 days [bop-manual p.99].",  # right document, page not retrieved
        "Every 14 days [made-up-manual p.12].",  # invented document
        "Every 14 days [bop-manual p.12] [safety-manual p.62].",  # one good, one invented
    ],
)
def test_any_citation_outside_the_retrieved_chunks_is_not_found(reply: str) -> None:
    rag, _, _ = pipeline(HITS, [reply])
    result = rag.ask("q")
    assert (result.found, result.answer, result.refusal_reason) == (
        False,
        NOT_FOUND,
        "invalid_citation",
    )
    assert result.citations == []


def test_the_rejected_llm_text_is_kept_for_debugging() -> None:
    rag, _, _ = pipeline(HITS, ["Every 14 days [bop-manual p.99]."])
    assert rag.ask("q").llm_answer == "Every 14 days [bop-manual p.99]."


# --- Pipeline: usage, cost, latency -------------------------------------


def test_usage_and_cost_come_from_the_llm_call() -> None:
    rag, llm, _ = pipeline(HITS, ["Every 14 days [bop-manual p.12]."], prices=Prices(0.40, 1.60))
    result = rag.ask("q")
    expected = Usage(
        input_tokens=sum(len(m.content.split()) for m in llm.calls[0]),
        output_tokens=5,
    )
    assert result.usage == expected
    assert result.llm_calls == 1
    assert result.model == "fake"
    assert result.cost_usd == pytest.approx(
        (expected.input_tokens * 0.40 + expected.output_tokens * 1.60) / 1e6
    )


def test_cost_is_unknown_without_prices_and_zero_without_a_call() -> None:
    rag, _, _ = pipeline(HITS, ["Every 14 days [bop-manual p.12]."])
    assert rag.ask("q").cost_usd is None

    rag, _, _ = pipeline([], [], prices=Prices(0.40, 1.60))
    result = rag.ask("q")
    assert (result.cost_usd, result.llm_calls, result.model) == (0.0, 0, None)


def test_latency_is_split_into_retrieval_and_llm() -> None:
    # clock reads: start, after retrieval, after the LLM
    rag, _, _ = pipeline(HITS, ["Yes [bop-manual p.12]."], clock=fake_clock(10.0, 10.5, 12.0))
    result = rag.ask("q")
    assert result.retrieval_s == pytest.approx(0.5)
    assert result.llm_s == pytest.approx(1.5)
    assert result.total_s == pytest.approx(2.0)


def test_latency_without_an_llm_call() -> None:
    rag, _, _ = pipeline([], [], clock=fake_clock(10.0, 10.25))
    result = rag.ask("q")
    assert (result.retrieval_s, result.llm_s, result.total_s) == (0.25, 0.0, 0.25)


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RagPipeline(FakeRetriever(HITS), FakeLlm(), top_k=0)
