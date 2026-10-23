"""Mode dispatch: each search mode routes to the right retriever (fakes, no server)."""

from pathlib import Path

import pytest
from conftest import FakeCrossEncoder, KeywordEmbedder
from qdrant_client import QdrantClient

from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.dense import DenseIndex
from rig_assist.retrieval.hybrid import HybridRetriever
from rig_assist.retrieval.rerank import RerankRetriever
from rig_assist.retrieval.service import SEARCH_MODES, SearchIndex


@pytest.fixture
def dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())


@pytest.fixture
def encoder() -> FakeCrossEncoder:
    return FakeCrossEncoder()


@pytest.fixture
def index(tmp_path: Path, dense: DenseIndex, encoder: FakeCrossEncoder) -> SearchIndex:
    return SearchIndex(IndexStore(tmp_path / "index"), dense=dense, cross_encoder=encoder)


def test_the_modes_are_the_four_from_the_plan() -> None:
    assert SEARCH_MODES == ("bm25", "dense", "hybrid", "hybrid_rerank")


def test_bm25_mode_uses_the_bm25_index(index: SearchIndex) -> None:
    assert index.retriever("bm25") is index.bm25()


def test_dense_mode_uses_the_dense_index(index: SearchIndex, dense: DenseIndex) -> None:
    assert index.retriever("dense") is dense


def test_hybrid_mode_fuses_dense_and_bm25_with_v1_settings(
    index: SearchIndex, dense: DenseIndex
) -> None:
    hybrid = index.retriever("hybrid")
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.dense is dense
    assert hybrid.bm25 is index.bm25()
    assert (hybrid.k_each, hybrid.rrf_k) == (20, 60)


def test_hybrid_rerank_mode_reranks_the_fused_top_20(
    index: SearchIndex, dense: DenseIndex, encoder: FakeCrossEncoder
) -> None:
    reranker = index.retriever("hybrid_rerank")
    assert isinstance(reranker, RerankRetriever)
    assert reranker.cross_encoder is encoder
    assert reranker.candidates == 20
    assert isinstance(reranker.first_stage, HybridRetriever)
    assert reranker.first_stage.dense is dense


def test_hybrid_uses_the_bm25_index_rebuilt_after_a_refresh(index: SearchIndex) -> None:
    before = index.bm25()
    index.refresh()
    hybrid = index.retriever("hybrid")
    assert isinstance(hybrid, HybridRetriever)
    assert hybrid.bm25 is index.bm25() is not before


def test_an_unknown_mode_raises(index: SearchIndex) -> None:
    with pytest.raises(ValueError, match="semantic"):
        index.retriever("semantic")  # type: ignore[arg-type]
