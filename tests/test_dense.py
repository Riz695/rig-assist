"""Dense retrieval: Qdrant in-memory mode (no server) + a fake embedder (no model)."""

import uuid

import pytest
from conftest import KeywordEmbedder
from qdrant_client import QdrantClient

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.dense import DenseIndex, point_id
from rig_assist.retrieval.embedder import FastEmbedEmbedder


def make_chunk(doc_id: str, page: int, text: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::p{page}::c{index}",
        doc_id=doc_id,
        doc_title=doc_id.replace("-", " ").title(),
        page=page,
        section="3.1 Mud Pumps" if page == 1 else None,
        parser="pdfplumber",
        text=text,
    )


DRILLING = [
    make_chunk("drilling-manual", 1, "Check the pump liners every shift"),
    make_chunk("drilling-manual", 2, "Close the BOP when a kick is detected"),
    make_chunk("drilling-manual", 3, "Riser tension depends on water depth"),
]
OFFSHORE = [make_chunk("offshore-manual", 7, "Crane lifts need a banksman")]


@pytest.fixture
def index() -> DenseIndex:
    dense = DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())
    dense.upsert_document("drilling-manual", DRILLING)
    dense.upsert_document("offshore-manual", OFFSHORE)
    return dense


def test_search_ranks_the_most_similar_chunk_first(index: DenseIndex) -> None:
    assert index.search("pump problems", k=3)[0].chunk == DRILLING[0]


def test_search_returns_full_metadata_and_cosine_scores(index: DenseIndex) -> None:
    top = index.search("crane", k=1)[0]
    assert top.chunk == OFFSHORE[0]
    assert 0.9 < top.score <= 1.0 + 1e-6
    assert top.dense_score == top.score


def test_search_orders_by_descending_score_and_respects_k(index: DenseIndex) -> None:
    hits = index.search("bop kick riser", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_search_on_an_empty_index_returns_empty() -> None:
    dense = DenseIndex(QdrantClient(":memory:"), "empty", KeywordEmbedder())
    assert dense.search("pump", k=5) == []
    assert dense.count() == 0


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_returns_empty(index: DenseIndex, query: str) -> None:
    assert index.search(query, k=5) == []


def test_search_rejects_non_positive_k(index: DenseIndex) -> None:
    with pytest.raises(ValueError):
        index.search("pump", k=0)


def test_reupserting_a_document_does_not_duplicate_chunks(index: DenseIndex) -> None:
    before = index.count()
    index.upsert_document("drilling-manual", DRILLING)
    assert index.count() == before == 4


def test_reupserting_with_fewer_chunks_removes_the_stale_ones(index: DenseIndex) -> None:
    index.upsert_document("drilling-manual", DRILLING[:1])
    assert index.count_document("drilling-manual") == 1
    assert index.count_document("offshore-manual") == 1, "other documents are untouched"
    assert all(
        h.chunk.page == 1
        for h in index.search("riser bop", k=5)
        if h.chunk.doc_id == "drilling-manual"
    )


def test_point_ids_are_deterministic_uuids_of_the_chunk_id() -> None:
    first, second = point_id("drilling-manual::p1::c0"), point_id("drilling-manual::p1::c0")
    assert first == second
    assert uuid.UUID(first).version == 5
    assert point_id("drilling-manual::p1::c1") != first


def test_query_prefix_is_applied_to_queries_only() -> None:
    embedder = KeywordEmbedder(query_prefix="Represent this: ")
    dense = DenseIndex(QdrantClient(":memory:"), "prefix", embedder)
    dense.upsert_document("drilling-manual", DRILLING)
    dense.search("pump", k=1)
    assert embedder.queries == ["pump"]  # the embedder adds the prefix itself


def test_collection_is_created_with_the_embedder_dimension() -> None:
    client = QdrantClient(":memory:")
    DenseIndex(client, "dims", KeywordEmbedder()).upsert_document("drilling-manual", DRILLING)
    info = client.get_collection("dims")
    assert info.config.params.vectors.size == KeywordEmbedder().dim  # type: ignore[union-attr]


# --- the real model (downloads ~67 MB once; skipped when that's impossible) -----


@pytest.mark.integration
def test_real_bge_small_embeds_384_dims_and_finds_the_related_text() -> None:
    try:
        embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
        vectors = embedder.embed_passages(
            ["The mud pump liner is worn and leaking", "Crane lifts need a banksman"]
        )
    except Exception as error:  # no network / no cache
        pytest.skip(f"embedding model unavailable: {error}")
    assert embedder.dim == 384
    assert len(vectors) == 2 and len(vectors[0]) == 384
    dense = DenseIndex(QdrantClient(":memory:"), "real", embedder)
    dense.upsert_document(
        "d",
        [
            make_chunk("d", 1, "The mud pump liner is worn and leaking"),
            make_chunk("d", 2, "Crane lifts need a banksman"),
        ],
    )
    assert dense.search("why is the pump losing pressure", k=1)[0].chunk.page == 1
