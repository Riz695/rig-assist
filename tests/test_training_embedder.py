from pathlib import Path

import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.training.negatives import MinedPair, write_mined
from rig_assist.training.train_embedder import (
    TrainConfig,
    history_metrics,
    ir_eval_data,
    load_mined,
    train_columns,
)


def chunk(page: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"bop::p{page}::c0",
        doc_id="bop",
        doc_title="Bop",
        page=page,
        section=None,
        parser="pdfplumber",
        text=text,
    )


def mined(query: str, page: int, split: str = "train", negative: int | None = None) -> MinedPair:
    return MinedPair(
        query=query,
        chunk_id=f"bop::p{page}::c0",
        doc_id="bop",
        page=page,
        positive=f"text {page}",
        split=split,  # type: ignore[arg-type]
        negative_ids=[] if negative is None else [f"bop::p{negative}::c0"],
        negatives=[] if negative is None else [f"text {negative}"],
        sources=[] if negative is None else ["dense"],
    )


ROWS = [
    mined("q1", 1, negative=2),
    mined("q2", 2),  # no negative was found
    mined("v1", 3, split="val", negative=1),
    mined("q3", 4, negative=1),
]


def test_train_columns_are_triplets_and_skip_pairs_without_a_negative() -> None:
    columns = train_columns(ROWS, hard_negatives=True)
    assert columns == {
        "anchor": ["q1", "q3"],
        "positive": ["text 1", "text 4"],
        "negative": ["text 2", "text 1"],
    }


def test_without_hard_negatives_every_train_pair_is_used() -> None:
    columns = train_columns(ROWS, hard_negatives=False)
    assert columns == {"anchor": ["q1", "q2", "q3"], "positive": ["text 1", "text 2", "text 4"]}


def test_the_query_prefix_goes_on_anchors_only() -> None:
    columns = train_columns(ROWS[:1], hard_negatives=True, query_prefix="Q: ")
    assert columns["anchor"] == ["Q: q1"]
    assert columns["positive"] == ["text 1"]
    assert columns["negative"] == ["text 2"]


def test_ir_eval_data_uses_validation_queries_against_the_whole_pool() -> None:
    pool = [chunk(p, f"text {p}") for p in (1, 2, 3, 4)]
    queries, corpus, relevant = ir_eval_data(ROWS, pool)
    assert queries == {"val-0": "v1"}
    assert corpus == {c.chunk_id: c.text for c in pool}
    assert relevant == {"val-0": {"bop::p3::c0"}}


def test_ir_eval_data_puts_the_prefix_on_queries() -> None:
    pool = [chunk(3, "text 3")]
    queries, _, _ = ir_eval_data(ROWS, pool, query_prefix="Q: ")
    assert queries == {"val-0": "Q: v1"}


def test_ir_eval_data_rejects_a_positive_missing_from_the_pool() -> None:
    with pytest.raises(ValueError, match="not in the pool"):
        ir_eval_data(ROWS, [chunk(1, "text 1")])


def test_load_mined_reads_what_write_mined_wrote(tmp_path: Path) -> None:
    write_mined(ROWS, tmp_path)
    loaded = load_mined(tmp_path)
    assert sorted(loaded, key=lambda m: m.query) == sorted(ROWS, key=lambda m: m.query)


def test_config_params_are_strings_for_mlflow() -> None:
    params = TrainConfig(epochs=2, batch_size=16).params()
    assert params["epochs"] == "2"
    assert params["batch_size"] == "16"
    assert params["base_model"] == "BAAI/bge-small-en-v1.5"
    assert params["mini_batch_size"] == "8"
    assert all(isinstance(v, str) for v in params.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 1},
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"mini_batch_size": 0},
        {"batch_size": 16, "mini_batch_size": 32},  # a mini-batch can't exceed the batch
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)  # type: ignore[arg-type]


def test_the_mini_batch_only_changes_memory_not_the_batch() -> None:
    config = TrainConfig(batch_size=32, mini_batch_size=4)
    assert config.batch_size == 32  # in-batch negatives still come from all 32 rows
    assert config.mini_batch_size == 4


def test_history_metrics_turn_the_trainer_log_into_stepped_mlflow_metrics() -> None:
    history = [
        {"loss": 2.1, "grad_norm": 5.0, "learning_rate": 1e-5, "epoch": 0.1, "step": 10},
        {"eval_val_cosine_mrr@10": 0.62, "epoch": 1.0, "step": 96},
        {"train_runtime": 80.5, "train_loss": 1.8, "epoch": 1.0, "step": 96},
    ]
    assert history_metrics(history) == [
        ("loss", 2.1, 10),
        ("grad_norm", 5.0, 10),
        ("learning_rate", 1e-5, 10),
        ("eval_val_cosine_mrr_at_10", 0.62, 96),
        ("train_runtime", 80.5, 96),
        ("train_loss", 1.8, 96),
    ]


def test_history_metrics_skip_entries_without_a_step_and_non_numbers() -> None:
    history: list[dict[str, object]] = [
        {"loss": 1.0},
        {"note": "text", "flag": True, "loss": 0.5, "step": 3},
    ]
    assert history_metrics(history) == [("loss", 0.5, 3)]
