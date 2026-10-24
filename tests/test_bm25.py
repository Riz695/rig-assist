import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.bm25 import BM25Index, tokenize
from rig_assist.retrieval.models import ScoredChunk


def make_chunk(text: str, page: int, doc_id: str = "drilling-manual") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::p{page}::c0",
        doc_id=doc_id,
        doc_title="Drilling Manual",
        page=page,
        section="3.1 Well Control" if page < 10 else None,
        parser="pdfplumber",
        text=text,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("The BOP stack must be pressure tested weekly", page=3),
        make_chunk("Riser tension limits depend on water depth", page=7),
        make_chunk("Crane lifts require a banksman and a lift plan", page=12),
        make_chunk("Mud weight is checked every shift by the mud engineer", page=15),
        make_chunk("H2S detectors alarm at 10 ppm; see API RP-54", page=21, doc_id="safety"),
    ]


@pytest.fixture
def index(chunks: list[Chunk]) -> BM25Index:
    return BM25Index.build(chunks)


# --- tokenize -----------------------------------------------------------


def test_tokenize_lowercases() -> None:
    assert tokenize("BOP Stack") == ["bop", "stack"]


def test_tokenize_keeps_acronyms_with_digits() -> None:
    assert "h2s" in tokenize("Detect H2S gas.")


def test_tokenize_keeps_hyphenated_codes() -> None:
    assert "rp-54" in tokenize("See API RP-54 for details")


def test_tokenize_drops_punctuation() -> None:
    assert tokenize("pressure, tested; (weekly)") == ["pressure", "tested", "weekly"]


# --- search -------------------------------------------------------------


def test_search_ranks_keyword_match_first(index: BM25Index) -> None:
    results = index.search("BOP pressure test", k=3)
    assert results[0].chunk.page == 3


def test_search_finds_codes_and_acronyms(index: BM25Index) -> None:
    assert index.search("RP-54", k=1)[0].chunk.doc_id == "safety"
    assert index.search("h2s alarm", k=1)[0].chunk.page == 21


def test_search_returns_the_full_chunk_with_a_positive_score(
    index: BM25Index, chunks: list[Chunk]
) -> None:
    top = index.search("riser", k=1)[0]
    assert isinstance(top, ScoredChunk)
    assert top.chunk == chunks[1]
    assert top.score > 0


def test_search_orders_by_descending_score(index: BM25Index) -> None:
    scores = [r.score for r in index.search("mud weight pressure tested", k=5)]
    assert len(scores) > 1
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(index: BM25Index) -> None:
    assert len(index.search("the mud bop riser crane", k=2)) == 2


def test_search_skips_chunks_with_no_match(index: BM25Index) -> None:
    assert index.search("helicopter", k=5) == []


@pytest.mark.parametrize("query", ["", "   ", "?!.,"])
def test_search_with_empty_query_returns_empty(index: BM25Index, query: str) -> None:
    assert index.search(query, k=5) == []


def test_search_rejects_non_positive_k(index: BM25Index) -> None:
    with pytest.raises(ValueError):
        index.search("bop", k=0)


def test_search_on_empty_index_returns_empty() -> None:
    assert BM25Index.build([]).search("bop", k=5) == []


def test_index_reports_its_size(index: BM25Index) -> None:
    assert len(index) == 5
    assert len(BM25Index.build([])) == 0


# --- small corpora ------------------------------------------------------


def test_search_works_on_a_two_chunk_index() -> None:
    """Okapi's IDF is exactly 0 for a term in 1 of 2 chunks, so nothing would be found.

    A fresh install with one small uploaded manual must still be searchable.
    """
    index = BM25Index.build(
        [make_chunk("Check the pump liners every shift", 1), make_chunk("Close the annular", 2)]
    )
    results = index.search("annular", k=5)
    assert [r.chunk.page for r in results] == [2]
    assert results[0].score > 0


def test_search_works_on_a_single_chunk_index() -> None:
    index = BM25Index.build([make_chunk("Close the annular preventer", 1)])
    assert [r.chunk.page for r in index.search("annular", k=5)] == [1]


def test_a_term_found_in_most_chunks_still_scores_above_zero(chunks: list[Chunk]) -> None:
    common = [make_chunk(f"pump note {i}", page=i) for i in range(1, 5)] + chunks
    assert all(r.score > 0 for r in BM25Index.build(common).search("pump", k=10))


# --- Okapi IDF (v1's formula, for the v1 baseline) ------------------------


def test_okapi_idf_reproduces_v1s_blind_spot_on_a_two_chunk_corpus() -> None:
    # Okapi IDF = ln(N - n + 0.5) - ln(n + 0.5) = ln(1.5 / 1.5) = 0 for 1 of 2 chunks.
    two = [make_chunk("mud pump liners", page=1), make_chunk("riser tension", page=2)]
    assert BM25Index(two, idf="okapi").search("liners", k=5) == []
    assert BM25Index(two, idf="lucene").search("liners", k=5)[0].chunk.page == 1


def test_okapi_scores_match_rank_bm25s_plain_okapi_exactly(chunks: list[Chunk]) -> None:
    from rank_bm25 import BM25Okapi

    plain = BM25Okapi([tokenize(c.text) for c in chunks])
    expected = plain.get_scores(tokenize("pressure tested riser"))
    hits = BM25Index(chunks, idf="okapi").search("pressure tested riser", k=len(chunks))
    by_page = {hit.chunk.page: hit.score for hit in hits}
    for chunk, score in zip(chunks, expected, strict=True):
        if score > 0:
            assert by_page[chunk.page] == pytest.approx(score)


def test_lucene_is_the_default_idf(chunks: list[Chunk]) -> None:
    assert BM25Index(chunks).idf == "lucene"
    assert BM25Index.build(chunks, idf="okapi").idf == "okapi"


def test_an_unknown_idf_raises(chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="idf"):
        BM25Index(chunks, idf="tfidf")  # type: ignore[arg-type]
