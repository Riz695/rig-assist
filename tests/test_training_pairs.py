import pytest

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.ingestion.models import Chunk
from rig_assist.training.pairs import (
    build_pairs,
    eval_pages,
    split_of,
    trainable_chunks,
)

LONG = " ".join(["word"] * 40)


def chunk(doc: str, page: int, index: int = 0, text: str = LONG) -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::p{page}::c{index}",
        doc_id=doc,
        doc_title=doc.title(),
        page=page,
        section=None,
        parser="pdfplumber",
        text=text,
    )


def question(qid: str, doc: str | None, pages: list[int]) -> EvalQuestion:
    answerable = doc is not None
    return EvalQuestion(
        id=qid,
        question="How often?",
        expected_doc_id=doc,
        expected_pages=pages,
        reference_answer="Every 14 days." if answerable else "Not in the manuals.",
        evidence_quote="every 14 days" if answerable else None,
        type="factual" if answerable else "unanswerable",
        difficulty="easy",
        wording="manual",
    )


# --- eval_pages and trainable_chunks --------------------------------------


def test_eval_pages_are_every_expected_doc_and_page() -> None:
    questions = [
        question("q1", "bop", [12, 13]),
        question("q2", "rig", [4]),
        question("q3", None, []),  # unanswerable: no pages
    ]
    assert eval_pages(questions) == {("bop", 12), ("bop", 13), ("rig", 4)}


def test_trainable_chunks_skip_eval_pages_and_short_chunks() -> None:
    chunks = [
        chunk("bop", 12),  # eval page
        chunk("bop", 12, 1),  # eval page, another chunk
        chunk("bop", 14),
        chunk("bop", 15, text="SECTION 4 Annex"),  # too short to ask about
        chunk("rig", 12),  # same page number, other manual: fine
    ]
    kept = trainable_chunks(chunks, excluded={("bop", 12)}, min_words=30)
    assert [c.chunk_id for c in kept] == ["bop::p14::c0", "rig::p12::c0"]


# --- split_of -------------------------------------------------------------


def test_split_is_stable_and_shared_by_a_page() -> None:
    assert split_of("bop", 7, 0.5) == split_of("bop", 7, 0.5)  # same answer every call


def test_split_puts_roughly_the_validation_fraction_of_pages_in_val() -> None:
    splits = [split_of(f"doc{d}", page, 0.1) for d in range(5) for page in range(1, 401)]
    share = splits.count("val") / len(splits)
    assert 0.07 < share < 0.13


def test_zero_validation_fraction_puts_everything_in_train() -> None:
    assert {split_of("bop", page, 0.0) for page in range(1, 200)} == {"train"}


def test_split_fraction_must_be_below_one() -> None:
    with pytest.raises(ValueError):
        split_of("bop", 1, 1.0)


# --- build_pairs ----------------------------------------------------------

CHUNKS = [chunk("bop", 14), chunk("bop", 15), chunk("rig", 3)]
QUERIES = {
    "bop::p14::c0": ["How often is the BOP tested?", "BOP test interval"],
    "bop::p15::c0": ["What pressure for the annular test?"],
    "rig::p3::c0": ["Who signs the permit to work?"],
}


def test_pairs_link_each_query_to_its_chunk_text_and_page() -> None:
    pairs = build_pairs(CHUNKS, QUERIES, excluded=set(), seed=0)
    assert len(pairs) == 4
    pair = next(p for p in pairs if p.query == "Who signs the permit to work?")
    assert (pair.chunk_id, pair.doc_id, pair.page) == ("rig::p3::c0", "rig", 3)
    assert pair.positive == LONG
    assert pair.split == split_of("rig", 3, 0.1)


def test_same_seed_gives_the_same_pairs_in_the_same_order() -> None:
    first = build_pairs(CHUNKS, QUERIES, excluded=set(), seed=42)
    assert build_pairs(CHUNKS, QUERIES, excluded=set(), seed=42) == first


def test_input_order_does_not_change_the_output() -> None:
    reordered = dict(reversed(list(QUERIES.items())))
    assert build_pairs(CHUNKS[::-1], reordered, excluded=set(), seed=1) == build_pairs(
        CHUNKS, QUERIES, excluded=set(), seed=1
    )


def test_another_seed_shuffles_differently_but_keeps_the_same_pairs() -> None:
    many = {f"bop::p{p}::c0": [f"question about page {p}"] for p in range(1, 31)}
    chunks = [chunk("bop", p) for p in range(1, 31)]
    a = build_pairs(chunks, many, excluded=set(), seed=1)
    b = build_pairs(chunks, many, excluded=set(), seed=2)
    assert a != b
    assert sorted(a, key=lambda p: p.query) == sorted(b, key=lambda p: p.query)


def test_seed_does_not_move_pages_between_splits() -> None:
    many = {f"bop::p{p}::c0": [f"question about page {p}"] for p in range(1, 101)}
    chunks = [chunk("bop", p) for p in range(1, 101)]
    split_a = {p.page: p.split for p in build_pairs(chunks, many, excluded=set(), seed=1)}
    split_b = {p.page: p.split for p in build_pairs(chunks, many, excluded=set(), seed=9)}
    assert split_a == split_b


def test_duplicate_queries_for_one_chunk_become_one_pair() -> None:
    queries = {"bop::p14::c0": ["BOP test interval", "  bop TEST   interval ", "BOP test interval"]}
    pairs = build_pairs(CHUNKS, queries, excluded=set(), seed=0)
    assert [p.query for p in pairs] == ["BOP test interval"]


def test_a_query_given_for_two_chunks_is_dropped_as_ambiguous() -> None:
    # With in-batch negatives, one chunk would be trained as the *wrong* answer to it.
    queries = {
        "bop::p14::c0": ["Pressure test?", "Stack height?"],
        "bop::p15::c0": ["pressure test?"],
    }
    pairs = build_pairs(CHUNKS, queries, excluded=set(), seed=0)
    assert [p.query for p in pairs] == ["Stack height?"]


def test_blank_queries_are_skipped() -> None:
    pairs = build_pairs(CHUNKS, {"bop::p14::c0": ["", "   ", "Real question?"]}, excluded=set())
    assert [p.query for p in pairs] == ["Real question?"]


def test_eval_pages_never_become_positives_even_if_queries_exist() -> None:
    # Leakage guard: a training query whose answer is an eval page would teach the test set.
    pairs = build_pairs(CHUNKS, QUERIES, excluded={("bop", 14)}, seed=0)
    assert all((p.doc_id, p.page) != ("bop", 14) for p in pairs)
    assert len(pairs) == 2


def test_queries_for_an_unknown_chunk_are_an_error() -> None:
    with pytest.raises(ValueError, match="bop::p99::c0"):
        build_pairs(CHUNKS, {"bop::p99::c0": ["Stale question?"]}, excluded=set())
