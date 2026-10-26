import json
import math
from pathlib import Path

import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.training.negatives import (
    MinedPair,
    mine_negatives,
    mining_stats,
    write_mined,
)
from rig_assist.training.pairs import TrainingPair, split_of

# Pages whose split is known, found by hashing (split_of is stable: a hash of doc:page).
TRAIN_PAGES = [p for p in range(1, 200) if split_of("bop", p) == "train"]
VAL_PAGE = next(p for p in range(1, 200) if split_of("bop", p) == "val")


class VectorEmbedder:
    """Returns the vector assigned to each text, so a test sets every similarity exactly.

    A text with no assigned vector gets one orthogonal to everything else (similarity 0).
    """

    def __init__(self, vectors: dict[str, list[float]], dim: int = 4) -> None:
        self.vectors = vectors
        self._dim = dim
        self.queries: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        return self.vectors.get(text, [0.0] * (self._dim - 1) + [1.0])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


def chunk(page: int, index: int, text: str, doc: str = "bop") -> Chunk:
    return Chunk(
        chunk_id=f"{doc}::p{page}::c{index}",
        doc_id=doc,
        doc_title=doc.title(),
        page=page,
        section=None,
        parser="pdfplumber",
        text=text,
    )


def pair(query: str, positive: Chunk) -> TrainingPair:
    return TrainingPair(
        query=query,
        chunk_id=positive.chunk_id,
        doc_id=positive.doc_id,
        page=positive.page,
        positive=positive.text,
        split=split_of(positive.doc_id, positive.page),
    )


P1, P2, P3, P4, P5 = TRAIN_PAGES[:5]


def test_never_the_positive_or_a_chunk_on_its_page() -> None:
    positive = chunk(P1, 0, "accumulator precharge pressure check")
    same_page = chunk(P1, 1, "accumulator precharge pressure nitrogen bottles")
    other = chunk(P2, 0, "accumulator bottle inspection interval")
    vectors = {
        "precharge?": [1.0, 0.0, 0.0, 0.0],
        positive.text: [1.0, 0.0, 0.0, 0.0],
        same_page.text: [0.9, 0.1, 0.0, 0.0],  # most similar of the candidates
        other.text: [0.6, 0.8, 0.0, 0.0],
    }
    [mined] = mine_negatives(
        [pair("precharge?", positive)], [positive, same_page, other], VectorEmbedder(vectors)
    )
    assert mined.negative_ids == [other.chunk_id]
    assert mined.negatives == [other.text]


def test_hardest_first_by_dense_similarity_and_n_negatives() -> None:
    positive = chunk(P1, 0, "positive text")
    near = chunk(P2, 0, "near text")
    mid = chunk(P3, 0, "mid text")
    far = chunk(P4, 0, "far text")
    vectors = {
        "q": [1.0, 0.0, 0.0, 0.0],
        positive.text: [1.0, 0.0, 0.0, 0.0],
        near.text: [0.8, 0.6, 0.0, 0.0],
        mid.text: [0.6, 0.8, 0.0, 0.0],
        far.text: [0.2, 0.98, 0.0, 0.0],
    }
    pool = [far, positive, mid, near]
    [mined] = mine_negatives([pair("q", positive)], pool, VectorEmbedder(vectors), n_negatives=2)
    assert mined.negative_ids == [near.chunk_id, mid.chunk_id]


def test_candidates_scoring_close_to_the_positive_are_dropped_as_likely_false_negatives() -> None:
    positive = chunk(P1, 0, "positive text")
    twin = chunk(P2, 0, "twin text")  # 0.99 of the positive's score: probably also an answer
    near = chunk(P3, 0, "near text")
    vectors = {
        "q": [1.0, 0.0, 0.0, 0.0],
        positive.text: [0.9, 0.436, 0.0, 0.0],  # similarity 0.9
        twin.text: [0.891, 0.0, 0.454, 0.0],  # 0.891 = 99% of 0.9
        near.text: [0.8, 0.0, 0.0, 0.6],  # 0.8 = 89% of 0.9
    }
    [mined] = mine_negatives(
        [pair("q", positive)],
        [positive, twin, near],
        VectorEmbedder(vectors),
        max_relative_score=0.95,
    )
    assert mined.negative_ids == [near.chunk_id]


def test_a_chunk_with_the_same_text_as_the_positive_is_never_a_negative() -> None:
    positive = chunk(P1, 0, "Check the  accumulator precharge.")
    copy = chunk(P2, 0, "check the accumulator precharge.")  # the same text on another page
    other = chunk(P3, 0, "Inspect the riser.")
    vectors = {
        "q": [1.0, 0.0, 0.0, 0.0],
        positive.text: [0.9, 0.436, 0.0, 0.0],
        copy.text: [0.5, 0.866, 0.0, 0.0],  # far below the margin; only the text gives it away
        other.text: [0.4, 0.0, 0.917, 0.0],
    }
    pool = [positive, copy, other]
    [mined] = mine_negatives([pair("q", positive)], pool, VectorEmbedder(vectors), n_negatives=2)
    assert mined.negative_ids == [other.chunk_id]


def test_train_pairs_never_get_negatives_from_validation_pages() -> None:
    positive = chunk(P1, 0, "positive text")
    val_chunk = chunk(VAL_PAGE, 0, "validation page text")
    train_chunk = chunk(P2, 0, "train page text")
    val_positive = chunk(VAL_PAGE, 1, "validation positive")
    vectors = {
        "q": [1.0, 0.0, 0.0, 0.0],
        "vq": [1.0, 0.0, 0.0, 0.0],
        positive.text: [1.0, 0.0, 0.0, 0.0],
        val_positive.text: [1.0, 0.0, 0.0, 0.0],
        val_chunk.text: [0.8, 0.6, 0.0, 0.0],  # hardest, but a validation page
        train_chunk.text: [0.6, 0.8, 0.0, 0.0],
    }
    pool = [positive, val_chunk, train_chunk, val_positive]
    train_mined, val_mined = mine_negatives(
        [pair("q", positive), pair("vq", val_positive)], pool, VectorEmbedder(vectors)
    )
    assert train_mined.split == "train"
    assert train_mined.negative_ids == [train_chunk.chunk_id]
    # A validation pair is never trained on, so any page other than its own will do.
    assert val_mined.split == "val"
    assert val_mined.negative_ids == [train_chunk.chunk_id]


def test_bm25_finds_candidates_the_embedder_ranks_too_low() -> None:
    positive = chunk(P1, 0, "kill line valve torque values")
    lexical = chunk(P2, 0, "kill line flushing procedure")  # shares words with the query
    dense_hits = [chunk(TRAIN_PAGES[2 + i], 0, f"semantic neighbour {i}") for i in range(3)]
    vectors = {
        "kill line torque": [1.0, 0.0, 0.0, 0.0],
        positive.text: [1.0, 0.0, 0.0, 0.0],
        lexical.text: [0.1, 0.0, 0.995, 0.0],  # the embedder thinks it's unrelated
        **{c.text: [0.7, 0.714, 0.0, 0.0] for c in dense_hits},
    }
    pool = [positive, lexical, *dense_hits]
    [mined] = mine_negatives(
        [pair("kill line torque", positive)],
        pool,
        VectorEmbedder(vectors),
        n_negatives=4,
        top_k=2,  # dense alone offers 2 semantic neighbours, never the lexical chunk
    )
    # BM25's allowed hits: only the lexical chunk (no shared words elsewhere); dense's top 2
    # allowed: two neighbours. Ranked by dense similarity, the lexical chunk comes last.
    assert mined.negative_ids == [dense_hits[0].chunk_id, dense_hits[1].chunk_id, lexical.chunk_id]
    assert mined.sources == ["dense", "dense", "bm25"]


def test_a_pair_can_end_up_with_fewer_negatives_than_asked() -> None:
    positive = chunk(P1, 0, "positive text")
    twin = chunk(P2, 0, "twin text")
    vectors = {
        "q": [1.0, 0.0, 0.0, 0.0],
        positive.text: [1.0, 0.0, 0.0, 0.0],
        twin.text: [1.0, 0.0, 0.0, 0.0],  # as good as the positive: dropped
    }
    [mined] = mine_negatives([pair("q", positive)], [positive, twin], VectorEmbedder(vectors))
    assert mined.negative_ids == []
    assert mined.negatives == []


def test_order_and_fields_of_the_pairs_are_kept_and_the_result_is_deterministic() -> None:
    chunks = [chunk(p, 0, f"text about item {p}") for p in (P1, P2, P3, P4, P5)]
    # Unit vectors 0.4 rad apart: neighbours score cos(0.4) = 0.92, under the 0.95 margin.
    angles = {i: 0.4 * i for i in range(5)}
    vectors = {
        c.text: [math.cos(angles[i]), math.sin(angles[i]), 0.0, 0.0] for i, c in enumerate(chunks)
    }
    vectors |= {f"query {i}": vectors[c.text] for i, c in enumerate(chunks)}
    pairs = [pair(f"query {i}", c) for i, c in enumerate(chunks)][::-1]

    first = mine_negatives(pairs, chunks, VectorEmbedder(vectors), n_negatives=2)
    second = mine_negatives(pairs, list(reversed(chunks)), VectorEmbedder(vectors), n_negatives=2)
    assert first == second
    assert first[-1].negative_ids == [chunks[1].chunk_id, chunks[2].chunk_id]  # query 0
    assert [m.query for m in first] == [p.query for p in pairs]
    assert [m.chunk_id for m in first] == [p.chunk_id for p in pairs]
    assert first[0].positive == pairs[0].positive


def test_the_queries_are_embedded_with_embed_query() -> None:
    positive = chunk(P1, 0, "positive text")
    embedder = VectorEmbedder({})
    mine_negatives([pair("how often?", positive)], [positive], embedder)
    assert embedder.queries == ["how often?"]


def test_a_positive_missing_from_the_pool_is_an_error() -> None:
    positive = chunk(P1, 0, "positive text")
    other = chunk(P2, 0, "other text")
    with pytest.raises(ValueError, match="not in the pool"):
        mine_negatives([pair("q", positive)], [other], VectorEmbedder({}))


@pytest.mark.parametrize(
    "kwargs",
    [{"n_negatives": 0}, {"top_k": 0}, {"max_relative_score": 0.0}, {"max_relative_score": 1.5}],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, float]) -> None:
    positive = chunk(P1, 0, "positive text")
    with pytest.raises(ValueError):
        mine_negatives([pair("q", positive)], [positive], VectorEmbedder({}), **kwargs)  # type: ignore[arg-type]


def mined(split: str, sources: list[str]) -> MinedPair:
    return MinedPair(
        query="q",
        chunk_id="bop::p1::c0",
        doc_id="bop",
        page=1,
        positive="text",
        split=split,  # type: ignore[arg-type]
        negative_ids=[f"bop::p{i + 2}::c0" for i in range(len(sources))],
        negatives=[f"negative {i}" for i in range(len(sources))],
        sources=sources,  # type: ignore[arg-type]
    )


def test_mining_stats_count_pairs_negatives_and_sources() -> None:
    stats = mining_stats(
        [mined("train", ["dense", "both"]), mined("train", ["bm25"]), mined("val", [])],
        n_negatives=2,
    )
    assert stats == {
        "pairs": 3,
        "short": 2,  # fewer negatives than asked
        "no_negative": 1,
        "negatives": 3,
        "sources": {"bm25": 1, "dense": 1, "both": 1},
    }


def test_write_mined_writes_one_file_per_split(tmp_path: Path) -> None:
    rows = [mined("train", ["dense"]), mined("val", ["bm25"]), mined("train", [])]
    counts = write_mined(rows, tmp_path)
    assert counts == {"train": 2, "val": 1}
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert [r["negative_ids"] for r in train] == [["bop::p2::c0"], []]
    assert train[0]["negatives"] == ["negative 0"]
    assert train[0]["sources"] == ["dense"]
