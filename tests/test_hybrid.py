import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rig_assist.retrieval.models import ScoredChunk


def sc(chunk_id: str, score: float = 0.0, page: int = 1) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="drilling-manual",
        doc_title="Drilling Manual",
        page=page,
        section=None,
        parser="pdfplumber",
        text=f"text of {chunk_id}",
    )
    return ScoredChunk(chunk=chunk, score=score)


def ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


# --- reciprocal_rank_fusion (hand-computed) -----------------------------


def test_rank_one_in_both_lists_scores_two_over_sixty_one() -> None:
    fused = reciprocal_rank_fusion([[sc("a")], [sc("a")]], k=60, top_n=5)
    assert ids(fused) == ["a"]
    assert fused[0].score == pytest.approx(2 / 61)


def test_hand_computed_scores_and_order() -> None:
    #           list 1   list 2
    # rank 1:     a        b
    # rank 2:     b        c
    fused = reciprocal_rank_fusion([[sc("a"), sc("b")], [sc("b"), sc("c")]], k=60, top_n=5)
    scores = {r.chunk.chunk_id: r.score for r in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)
    assert ids(fused) == ["b", "a", "c"]


def test_a_chunk_in_only_one_list_still_appears_with_its_single_contribution() -> None:
    fused = reciprocal_rank_fusion([[sc("a"), sc("x")], [sc("z")]], k=60, top_n=5)
    scores = {r.chunk.chunk_id: r.score for r in fused}
    assert set(scores) == {"a", "x", "z"}
    assert scores["x"] == pytest.approx(1 / 62)


def test_ties_break_by_chunk_id_whatever_the_list_order() -> None:
    first = reciprocal_rank_fusion([[sc("b")], [sc("a")]], k=60, top_n=5)
    second = reciprocal_rank_fusion([[sc("a")], [sc("b")]], k=60, top_n=5)
    assert ids(first) == ids(second) == ["a", "b"]


def test_k_controls_how_much_top_ranks_dominate() -> None:
    # "a" is 1st in one list; "b" is 2nd in both. Small k favours the single top rank.
    lists = [[sc("a"), sc("b")], [sc("c"), sc("b")]]
    small_k = {r.chunk.chunk_id: r.score for r in reciprocal_rank_fusion(lists, k=1, top_n=5)}
    large_k = {r.chunk.chunk_id: r.score for r in reciprocal_rank_fusion(lists, k=60, top_n=5)}
    assert small_k["a"] == pytest.approx(1 / 2)
    assert small_k["b"] == pytest.approx(2 / 3)
    assert small_k["a"] / small_k["b"] > large_k["a"] / large_k["b"]


def test_original_scores_are_ignored_only_ranks_matter() -> None:
    fused = reciprocal_rank_fusion([[sc("a", 99.0), sc("b", 98.0)], [sc("b", 0.1)]], top_n=5)
    assert ids(fused) == ["b", "a"]


def test_truncates_to_top_n() -> None:
    fused = reciprocal_rank_fusion([[sc("a"), sc("b"), sc("c")]], k=60, top_n=2)
    assert ids(fused) == ["a", "b"]


def test_keeps_the_chunk_metadata() -> None:
    fused = reciprocal_rank_fusion([[sc("a", page=9)]], top_n=5)
    assert fused[0].chunk.page == 9
    assert fused[0].chunk.text == "text of a"


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([[], []], top_n=5) == []
    assert reciprocal_rank_fusion([], top_n=5) == []


@pytest.mark.parametrize(("k", "top_n"), [(0, 5), (-1, 5), (60, 0)])
def test_rejects_invalid_parameters(k: int, top_n: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[sc("a")]], k=k, top_n=top_n)


# --- HybridRetriever (with fake retrievers) ------------------------------


class FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.calls.append((query, k))
        return self.results[:k]


def test_hybrid_fuses_both_retrievers_without_duplicates() -> None:
    dense = FakeRetriever([sc("a"), sc("b")])
    bm25 = FakeRetriever([sc("b"), sc("c")])
    fused = HybridRetriever(dense=dense, bm25=bm25).search("q", k=5)
    assert ids(fused) == ["b", "a", "c"]


def test_hybrid_asks_each_retriever_for_k_each_results() -> None:
    dense, bm25 = FakeRetriever([]), FakeRetriever([])
    HybridRetriever(dense=dense, bm25=bm25, k_each=20).search("BOP test", k=5)
    assert dense.calls == [("BOP test", 20)]
    assert bm25.calls == [("BOP test", 20)]


def test_hybrid_respects_k() -> None:
    dense = FakeRetriever([sc("a"), sc("b"), sc("c")])
    bm25 = FakeRetriever([sc("d")])
    assert len(HybridRetriever(dense=dense, bm25=bm25).search("q", k=2)) == 2


def test_hybrid_score_is_rrf_and_dense_score_is_kept() -> None:
    dense = FakeRetriever([sc("a", 0.83), sc("b", 0.61)])
    bm25 = FakeRetriever([sc("a", 9.0), sc("z", 4.0)])
    fused = {r.chunk.chunk_id: r for r in HybridRetriever(dense=dense, bm25=bm25).search("q", k=5)}
    assert fused["a"].score == pytest.approx(2 / 61)
    assert fused["a"].dense_score == pytest.approx(0.83)
    assert fused["b"].dense_score == pytest.approx(0.61)
    assert fused["z"].dense_score is None  # only BM25 found it
