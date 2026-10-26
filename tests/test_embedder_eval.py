"""Phase 5 eval: the main index's chunks re-embedded by another model (the fine-tuned one,
and the base model exported the same way), compared with the served embedder.

Fakes stand in for Qdrant and the models; `make_configs` only builds objects (nothing is
loaded or embedded until a config runs).
"""

from pathlib import Path

import pytest
from test_run_retrieval import (
    BM25,
    DENSE,
    QUESTIONS,
    FakeTracker,
    ScriptedRetriever,
    rerank_params,
)

from rig_assist.config import Settings
from rig_assist.evaluation.metrics import sign_test
from rig_assist.evaluation.retrieval_eval import QuestionResult
from rig_assist.evaluation.run_retrieval import (
    ALL_CONFIGS,
    EMBEDDER_CONFIGS,
    EMBEDDER_MODES,
    EMBEDDER_VARIANTS,
    ConfigResult,
    EvalConfig,
    embedder_collection,
    finetune_rows,
    make_configs,
    paired_counts,
    render_report,
    run_configs,
    sync_dense,
)
from rig_assist.ingestion.models import Chunk
from rig_assist.ingestion.store import DocumentInfo, IndexStore
from rig_assist.ingestion.variants import INDEX_VARIANTS, variant_collection
from rig_assist.retrieval.models import ScoredChunk

# --- sign test ---------------------------------------------------------------


def test_sign_test_hand_computed() -> None:
    # 6 better, 1 worse: P(X <= 1) for Binomial(7, 0.5) = 8/128, two-sided = 16/128.
    assert sign_test(6, 1) == pytest.approx(0.125)
    assert sign_test(1, 6) == pytest.approx(0.125)


def test_sign_test_even_split_and_no_changes_are_p_1() -> None:
    assert sign_test(3, 3) == 1.0
    assert sign_test(0, 0) == 1.0


def test_sign_test_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        sign_test(-1, 2)


# --- configs and collections -------------------------------------------------


def test_embedder_configs_are_each_variant_in_each_mode() -> None:
    assert set(EMBEDDER_VARIANTS) == {"base_fp32", "ft"}
    assert EMBEDDER_MODES == ("dense", "hybrid", "hybrid_rerank")
    expected = {f"{v}_{m}" for v in EMBEDDER_VARIANTS for m in EMBEDDER_MODES}
    assert set(EMBEDDER_CONFIGS) == expected
    assert set(EMBEDDER_CONFIGS) <= set(ALL_CONFIGS)


def test_each_embedder_has_its_own_collection() -> None:
    names = {embedder_collection("rig", v) for v in EMBEDDER_VARIANTS}
    names |= {variant_collection("rig", v) for v in INDEX_VARIANTS}
    assert len(names) == len(EMBEDDER_VARIANTS) + len(INDEX_VARIANTS)
    assert embedder_collection("rig", "ft") == "rig__emb_ft"


def test_make_configs_points_an_embedder_config_at_its_model_and_collection(
    tmp_path: Path,
) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, data_index_dir=tmp_path / "index", models_dir=tmp_path / "models"
    )
    [config] = make_configs(settings, ["ft_hybrid"])
    assert config.name == "ft_hybrid"
    assert config.params["mode"] == "hybrid"
    assert config.params["embedder_variant"] == "ft"
    assert config.params["embedding_model"] == str(tmp_path / "models" / EMBEDDER_VARIANTS["ft"])
    assert config.params["qdrant_collection"] == "rig_assist_chunks__emb_ft"
    assert config.params["query_prefix"] == ""
    assert config.prepare is not None  # fills the collection before the run


# --- filling a collection ----------------------------------------------------


class FakeDense:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.upserts: list[tuple[str, int]] = []

    def count_document(self, doc_id: str) -> int:
        return self.counts.get(doc_id, 0)

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        self.upserts.append((doc_id, len(chunks)))
        self.counts[doc_id] = len(chunks)


def store_with(tmp_path: Path, chunks_per_doc: dict[str, int]) -> IndexStore:
    store = IndexStore(tmp_path / "index")
    for doc_id, n in chunks_per_doc.items():
        chunks = [
            Chunk(f"{doc_id}::p{i}::c0", doc_id, doc_id, i + 1, None, "pdfplumber", f"t{i}")
            for i in range(n)
        ]
        info = DocumentInfo(doc_id, doc_id, f"{doc_id}.pdf", "ab" * 32, n, n, [])
        store.put(info, chunks)
    return store


def test_sync_embeds_only_documents_whose_point_count_differs(tmp_path: Path) -> None:
    store = store_with(tmp_path, {"bop": 3, "mud-pump": 2, "crane": 1})
    dense = FakeDense({"bop": 3, "mud-pump": 1})  # crane missing, mud-pump short
    assert sync_dense(store, dense) == ["crane", "mud-pump"]
    assert dense.upserts == [("crane", 1), ("mud-pump", 2)]


def test_sync_twice_embeds_nothing_the_second_time(tmp_path: Path) -> None:
    store = store_with(tmp_path, {"bop": 2})
    dense = FakeDense({})
    sync_dense(store, dense)
    assert sync_dense(store, dense) == []
    assert len(dense.upserts) == 1


def test_prepare_runs_before_the_warm_up_query() -> None:
    order: list[str] = []
    retriever = ScriptedRetriever(DENSE.results)
    original = retriever.search

    def search(query: str, k: int) -> list[ScoredChunk]:
        order.append("search")
        return original(query, k)

    retriever.search = search  # type: ignore[method-assign]
    config = EvalConfig("ft_dense", retriever, {}, prepare=lambda: order.append("prepare"))
    run_configs([config], QUESTIONS, FakeTracker())
    assert order[:2] == ["prepare", "search"]


# --- paired comparison and report --------------------------------------------


def result(name: str, ranks: list[int | None], params: dict[str, str]) -> ConfigResult:
    results = [QuestionResult(f"q{i}", r, 0.1, None) for i, r in enumerate(ranks)]
    return ConfigResult(name, params, results, {})


def test_paired_counts_compare_reciprocal_rank_question_by_question() -> None:
    base = result("dense", [1, 2, None, 3, 11], {})
    other = result("ft_dense", [1, 1, 4, None, None], {})
    # q0 same, q1 better, q2 better (found vs not), q3 worse, q4 same (both miss at 10)
    assert paired_counts(other, base) == (2, 1, 2)


def test_paired_counts_need_the_same_questions() -> None:
    with pytest.raises(ValueError):
        paired_counts(result("a", [1], {}), result("b", [1, 2], {}))


def embedder_params(mode: str, variant: str | None = None) -> dict[str, str]:
    params = {"mode": mode, "parser": "pdfplumber", "query_prefix": ""}
    if variant:
        params |= {"embedder_variant": variant, "embedding_model": f"models/{variant}"}
    else:
        params["embedding_model"] = "BAAI/bge-small-en-v1.5"
    return params


def embedder_run() -> list[ConfigResult]:
    configs = [
        EvalConfig("dense", ScriptedRetriever(BM25.results), embedder_params("dense")),
        EvalConfig("ft_dense", ScriptedRetriever(DENSE.results), embedder_params("dense", "ft")),
        EvalConfig("bm25", ScriptedRetriever(BM25.results), {"mode": "bm25"}),
    ]
    return run_configs(configs, QUESTIONS, FakeTracker())


def test_report_compares_embedders_mode_by_mode_with_paired_counts() -> None:
    report = render_report(embedder_run(), title="Retrieval eval")
    section = report.split("## Embedder comparison")[1].split("\n## ")[0]
    assert "| mode | embedder | Hit@1 | Hit@5 | MRR@10 |" in section
    served = "| dense | BAAI/bge-small-en-v1.5 (served) | 0/2 (0.00) | 2/2 (1.00) | 0.500 |"
    assert served in section
    # ft finds q1 at rank 1 instead of 2 (better), q2 at rank 2 in both (same)
    assert "| dense | ft | 1/2 (0.50) | 2/2 (1.00) | 0.750 |" in section
    assert section.rstrip().endswith("| 1 / 0 | 1.00 |")


def test_no_embedder_comparison_without_an_embedder_variant() -> None:
    configs = [EvalConfig("dense", ScriptedRetriever(DENSE.results), embedder_params("dense"))]
    report = render_report(run_configs(configs, QUESTIONS, FakeTracker()), title="Retrieval eval")
    assert "## Embedder comparison" not in report


def test_embedder_variants_stay_out_of_the_reranker_comparison() -> None:
    configs = [
        EvalConfig("hybrid_rerank", ScriptedRetriever(DENSE.results), rerank_params("big/m", 20)),
        EvalConfig("r2", ScriptedRetriever(DENSE.results), rerank_params("small/m", 10)),
        EvalConfig(
            "ft_hybrid_rerank",
            ScriptedRetriever(DENSE.results),
            {**rerank_params("big/m", 20), "embedder_variant": "ft"},
        ),
    ]
    report = render_report(run_configs(configs, QUESTIONS, FakeTracker()), title="Retrieval eval")
    rerank_section = report.split("## Reranker comparison")[1].split("\n## ")[0]
    assert rerank_section.count("| big/m |") == 1


def test_finetune_rows_are_the_served_modes_and_the_embedder_variants() -> None:
    rows = finetune_rows(embedder_run())
    assert [r.name for r in rows] == ["dense", "ft_dense"]
