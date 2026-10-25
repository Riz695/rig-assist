"""Cross-encoder reranking: a fake first stage + a fake cross-encoder (no model)."""

import pytest
from conftest import FakeCrossEncoder

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval import rerank
from rig_assist.retrieval.models import ScoredChunk
from rig_assist.retrieval.rerank import FastEmbedCrossEncoder, RerankRetriever


def sc(chunk_id: str, text: str, score: float = 0.0, dense: float | None = None) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="drilling-manual",
        doc_title="Drilling Manual",
        page=1,
        section=None,
        parser="pdfplumber",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=score, dense_score=dense)


def ids(results: list[ScoredChunk]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


class FakeFirstStage:
    """Returns its fixed ranking (cut to k) and records the k it was asked for."""

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.asked_k: list[int] = []

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        self.asked_k.append(k)
        return self.hits[:k]


FIRST_STAGE = [
    sc("a", "crane lifts need a banksman", score=0.9),
    sc("b", "mud pump liner worn pressure dropping", score=0.8, dense=0.71),
    sc("c", "mud pump maintenance schedule", score=0.7),
]


def test_reorders_the_first_stage_by_cross_encoder_score() -> None:
    reranker = RerankRetriever(FakeFirstStage(FIRST_STAGE), FakeCrossEncoder())
    assert ids(reranker.search("mud pump pressure dropping", k=3)) == ["b", "c", "a"]


def test_score_is_the_cross_encoder_score_and_dense_score_is_kept() -> None:
    reranker = RerankRetriever(FakeFirstStage(FIRST_STAGE), FakeCrossEncoder())
    top = reranker.search("mud pump pressure dropping", k=1)
    assert len(top) == 1
    assert top[0].score == 4.0
    assert top[0].dense_score == 0.71
    assert top[0].chunk == FIRST_STAGE[1].chunk


def test_asks_the_first_stage_for_the_candidate_count_not_k() -> None:
    first = FakeFirstStage(FIRST_STAGE)
    RerankRetriever(first, FakeCrossEncoder(), candidates=20).search("pump", k=5)
    assert first.asked_k == [20]


def test_asks_for_k_when_k_is_larger_than_the_candidate_count() -> None:
    first = FakeFirstStage(FIRST_STAGE)
    RerankRetriever(first, FakeCrossEncoder(), candidates=2).search("pump", k=3)
    assert first.asked_k == [3]


def test_scores_the_query_against_each_candidate_text_in_one_call() -> None:
    encoder = FakeCrossEncoder()
    RerankRetriever(FakeFirstStage(FIRST_STAGE), encoder).search("mud pump", k=2)
    assert encoder.calls == [("mud pump", [hit.chunk.text for hit in FIRST_STAGE])]


def test_no_candidates_returns_empty_without_calling_the_model() -> None:
    encoder = FakeCrossEncoder()
    assert RerankRetriever(FakeFirstStage([]), encoder).search("pump", k=5) == []
    assert encoder.calls == []


def test_ties_keep_the_first_stage_order() -> None:
    # No query word appears anywhere: every score is 0, so the first stage decides.
    reranker = RerankRetriever(FakeFirstStage(FIRST_STAGE), FakeCrossEncoder())
    assert ids(reranker.search("riser tension", k=3)) == ["a", "b", "c"]


def test_k_below_one_raises() -> None:
    with pytest.raises(ValueError):
        RerankRetriever(FakeFirstStage(FIRST_STAGE), FakeCrossEncoder()).search("pump", k=0)


def test_candidates_below_one_raises() -> None:
    with pytest.raises(ValueError):
        RerankRetriever(FakeFirstStage(FIRST_STAGE), FakeCrossEncoder(), candidates=0)


class CountingTextCrossEncoder:
    """Stands in for fastembed's TextCrossEncoder and counts how often it's loaded."""

    loads = 0

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        type(self).loads += 1

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


def test_unload_drops_the_model_and_the_next_score_reloads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rerank, "TextCrossEncoder", CountingTextCrossEncoder)
    CountingTextCrossEncoder.loads = 0
    cross_encoder = FastEmbedCrossEncoder("some/model")
    cross_encoder.score("q", ["a"])
    cross_encoder.score("q", ["b"])
    assert CountingTextCrossEncoder.loads == 1  # loaded once, then reused
    cross_encoder.unload()
    cross_encoder.unload()  # unloading twice is harmless
    assert cross_encoder.score("q", ["c"]) == [0.0]
    assert CountingTextCrossEncoder.loads == 2


# --- a real cross-encoder (small model, ~80 MB once; skipped when unavailable) ---------


@pytest.mark.integration
def test_real_cross_encoder_ranks_the_relevant_text_highest() -> None:
    # The adapter is model-agnostic; a small model keeps the download light for CI.
    try:
        encoder = FastEmbedCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
        scores = encoder.score(
            "why is the mud pump losing pressure",
            ["Crane lifts need a banksman", "A worn mud pump liner causes pressure loss"],
        )
    except Exception as error:  # no network / no cache
        pytest.skip(f"reranker model unavailable: {error}")
    assert len(scores) == 2
    assert scores[1] > scores[0]
