"""Run the frozen eval set through every retrieval config, track each run, write the report.

`make eval` runs `main()`. Configs: v2's four search modes on the main index, other
rerankers and candidate counts on the main index (`RERANK_VARIANTS`), the same
modes on each index variant (`olmocr_all_<mode>`, for the parser comparison; see
`rig_assist.ingestion.variants`), and the two v1 baselines on their own indexes (see
`rig_assist.v1_baseline`). Every number in the report has a tracked run with the same
params (git commit, eval-set hash, index contents).
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from rig_assist.config import Settings, get_settings
from rig_assist.evaluation.eval_set import EvalQuestion, load_eval_set, load_reference_pages
from rig_assist.evaluation.metrics import RefusalResult, refusal_metrics
from rig_assist.evaluation.retrieval_eval import (
    CONTEXT_K,
    RESULTS_K,
    QuestionResult,
    Summary,
    choose_threshold,
    evaluate,
    refusal_cases,
    summarize,
)
from rig_assist.evaluation.tracking import MlflowTracker, NullTracker, Tracker
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.store import IndexStore
from rig_assist.ingestion.variants import (
    INDEX_VARIANTS,
    IndexVariant,
    variant_collection,
    variant_store_dir,
)
from rig_assist.retrieval.dense import make_dense_index
from rig_assist.retrieval.models import Retriever
from rig_assist.retrieval.rerank import FastEmbedCrossEncoder, RerankRetriever
from rig_assist.retrieval.service import SEARCH_MODES, SearchIndex
from rig_assist.v1_baseline.pipeline import (
    V1_CONFIGS,
    make_v1_retriever,
    v1_collection,
    v1_store_dir,
)

logger = logging.getLogger(__name__)

V1_THRESHOLD = 0.6  # v1 kept chunks with dense_score >= 0.6
WARM_UP_QUERY = "warm-up query: load the models before timing"
# The manual with no text layer: its text only exists through OCR (PLAN.md, Phase 1).
SCANNED_DOCS = frozenset({"marine-offshore-log-ops-manual"})
SUBSETS = ("table", "scanned", "paraphrased", "manual_wording")
VARIANT_CONFIGS = tuple(f"{variant}_{mode}" for variant in INDEX_VARIANTS for mode in SEARCH_MODES)
# Reranker choice: other cross-encoders and candidate counts on the main index, next to
# `hybrid_rerank` (settings.reranker_model, 20 candidates). Name → (model, candidates).
RERANK_VARIANTS: dict[str, tuple[str, int]] = {
    "hybrid_rerank_bge_base_c10": ("BAAI/bge-reranker-base", 10),
    "hybrid_rerank_minilm_c10": ("Xenova/ms-marco-MiniLM-L-6-v2", 10),
    "hybrid_rerank_minilm_c20": ("Xenova/ms-marco-MiniLM-L-6-v2", 20),
    "hybrid_rerank_jina_tiny_c10": ("jinaai/jina-reranker-v1-tiny-en", 10),
    "hybrid_rerank_jina_tiny_c20": ("jinaai/jina-reranker-v1-tiny-en", 20),
}
ALL_CONFIGS = (*SEARCH_MODES, *RERANK_VARIANTS, *VARIANT_CONFIGS, *V1_CONFIGS)
# The `parser` param: the main index uses pdfplumber wherever there is a text layer
# (olmOCR only for pages without one); a variant's parser is in its name.
MAIN_PARSER = "pdfplumber"
VARIANT_PARSER: dict[IndexVariant, str] = {"olmocr_all": "olmocr"}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    retriever: Retriever
    params: dict[str, str]


@dataclass
class ConfigResult:
    name: str
    params: dict[str, str]
    results: list[QuestionResult]
    summaries: dict[str, Summary]
    refusal_v1: RefusalResult | None = None
    tuned_threshold: float | None = None
    refusal_tuned: RefusalResult | None = None
    run_id: str | None = field(default=None)


def _metrics(result: ConfigResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for subset, summary in result.summaries.items():
        for key, value in asdict(summary).items():
            metrics[f"{subset}/{key}"] = float(value)
    for prefix, refusal in (
        ("refusal_v1", result.refusal_v1),
        ("refusal_tuned", result.refusal_tuned),
    ):
        if refusal is None:
            continue
        for key in ("precision", "recall"):
            value = getattr(refusal, key)
            if value is not None:  # undefined (0/0) values are left out, not logged as 0
                metrics[f"{prefix}/{key}"] = value
    if result.tuned_threshold is not None:
        metrics["refusal_tuned/threshold"] = result.tuned_threshold
    return metrics


def run_configs(
    configs: Sequence[EvalConfig],
    questions: Sequence[EvalQuestion],
    tracker: Tracker,
    common_params: Mapping[str, str] | None = None,
    scanned_docs: frozenset[str] = SCANNED_DOCS,
) -> list[ConfigResult]:
    out = []
    for config in configs:
        logger.info("Evaluating %s", config.name)
        config.retriever.search(
            WARM_UP_QUERY, RESULTS_K
        )  # model loading isn't a question's latency
        results = evaluate(config.retriever, questions)
        result = ConfigResult(
            name=config.name,
            params={**(common_params or {}), **config.params},
            results=results,
            summaries=summarize(results, questions, scanned_docs),
        )
        cases = refusal_cases(results, questions)
        if any(c.top_score is not None for c in cases):
            result.refusal_v1 = refusal_metrics(cases, V1_THRESHOLD)
            result.tuned_threshold = choose_threshold(cases)
            result.refusal_tuned = refusal_metrics(cases, result.tuned_threshold)
        per_question = json.dumps([asdict(r) for r in results], indent=1)
        result.run_id = tracker.log_run(
            config.name, result.params, _metrics(result), {"per_question.json": per_question}
        )
        out.append(result)
    return out


# --- Markdown report -------------------------------------------------------


def _count(share: float, n: int) -> str:
    return f"{round(share * n)}/{n} ({share:.2f})"


def _fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(results: Sequence[ConfigResult], title: str, notes: Sequence[str] = ()) -> str:
    lines = [f"# {title}", "", *notes, ""] if notes else [f"# {title}", ""]
    n = results[0].summaries["all"].n if results else 0
    lines += [
        f"## Overall ({n} answerable questions)",
        "",
        "Hit@k: share of questions with a chunk from an expected page in the top k. "
        "MRR@10: mean of 1/rank of the first such chunk. Latency per query, after one "
        "warm-up query.",
        "",
        "| config | Hit@1 | Hit@5 | MRR@10 | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.summaries["all"]
        lines.append(
            f"| {r.name} | {_count(s.hit_at_1, s.n)} | {_count(s.hit_at_5, s.n)} | "
            f"{s.mrr_at_10:.3f} | {s.p50_latency_s:.3f} s | {s.p95_latency_s:.3f} s |"
        )
    for subset in SUBSETS:
        rows = [r for r in results if subset in r.summaries]
        if not rows:
            continue
        lines += [
            "",
            f"### Subset: {subset} ({rows[0].summaries[subset].n} questions)",
            "",
            "| config | Hit@1 | Hit@5 | MRR@10 |",
            "|---|---|---|---|",
        ]
        for r in rows:
            s = r.summaries[subset]
            lines.append(
                f"| {r.name} | {_count(s.hit_at_1, s.n)} | {_count(s.hit_at_5, s.n)} | "
                f"{s.mrr_at_10:.3f} |"
            )
    lines += _parser_comparison(results)
    lines += _reranker_comparison(results)
    lines += [
        "",
        '## Refusal ("not found")',
        "",
        "Refuse when no top-5 chunk has a dense (cosine) score at or above the threshold. "
        "Positive class = refused. The tuned threshold maximises refusal F1 **on this same "
        "eval set**, so its numbers are optimistic. Not applicable to BM25 (no cosine score).",
        "",
        f"| config | precision @ {V1_THRESHOLD} (v1) | recall @ {V1_THRESHOLD} (v1) | "
        "tuned threshold | precision @ tuned | recall @ tuned |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.refusal_v1 is None or r.refusal_tuned is None:
            continue
        lines.append(
            f"| {r.name} | {_fmt(r.refusal_v1.precision)} | {_fmt(r.refusal_v1.recall)} | "
            f"{_fmt(r.tuned_threshold, 3)} | {_fmt(r.refusal_tuned.precision)} | "
            f"{_fmt(r.refusal_tuned.recall)} |"
        )
    run_ids = [f"- {r.name}: `{r.run_id}`" for r in results if r.run_id]
    if run_ids:
        lines += ["", "## MLflow runs", "", *run_ids]
    return "\n".join(lines) + "\n"


def _parser_comparison(results: Sequence[ConfigResult]) -> list[str]:
    """Rows per (mode, parser) for the modes that were run with more than one parser."""
    by_mode: dict[str, list[ConfigResult]] = {}
    for r in results:
        if "parser" in r.params:
            by_mode.setdefault(r.params["mode"], []).append(r)
    compared = {mode: rows for mode, rows in by_mode.items() if len(rows) > 1}
    if not compared:
        return []
    lines = [
        "",
        "## Parser comparison",
        "",
        "Same questions, same retrieval; only the page text differs. `pdfplumber` = the "
        "main index (text layer; olmOCR only for pages without one), `olmocr` = olmOCR "
        "text for every page. The scanned manual is olmOCR text in both.",
        "",
        "| mode | parser | Hit@5 | MRR@10 | table Hit@5 | table MRR@10 "
        "| scanned Hit@5 | scanned MRR@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, rows in compared.items():
        for r in rows:
            cells = []
            for subset in ("all", "table", "scanned"):
                s = r.summaries.get(subset)
                cells += (
                    ["n/a", "n/a"] if s is None else [_count(s.hit_at_5, s.n), f"{s.mrr_at_10:.3f}"]
                )
            lines.append(f"| {mode} | {r.params['parser']} | {' | '.join(cells)} |")
    return lines


def _reranker_comparison(results: Sequence[ConfigResult]) -> list[str]:
    """Quality and latency per (reranker, candidates) on the main index, vs plain hybrid."""
    main = [r for r in results if r.params.get("parser", MAIN_PARSER) == MAIN_PARSER]
    reranked = [r for r in main if "reranker_model" in r.params]
    if len(reranked) < 2:
        return []
    lines = [
        "",
        "## Reranker comparison",
        "",
        "Main index. Each cross-encoder rescores the hybrid retriever's top `candidates` "
        "chunks. `none (hybrid)` is the same first stage without reranking. Latency is the "
        "whole query (first stage + reranking), on this machine's CPU.",
        "",
        "| reranker | candidates | Hit@1 | Hit@5 | MRR@10 | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [("none (hybrid)", "–", r) for r in main if r.params.get("mode") == "hybrid"]
    rows += [(r.params["reranker_model"], r.params["rerank_candidates"], r) for r in reranked]
    for model, candidates, r in rows:
        s = r.summaries["all"]
        lines.append(
            f"| {model} | {candidates} | {_count(s.hit_at_1, s.n)} | {_count(s.hit_at_5, s.n)} "
            f"| {s.mrr_at_10:.3f} | {s.p50_latency_s:.3f} s | {s.p95_latency_s:.3f} s |"
        )
    return lines


# --- Real configs (CLI) ----------------------------------------------------


def _index_params(store: IndexStore) -> dict[str, str]:
    """What the index contains, read from the index itself (so runs are reproducible)."""
    chunks = list(store.chunks())
    parsers = Counter(chunk.parser for chunk in chunks)
    return {
        "index_dir": str(store.root),
        "index_docs": str(len(store.documents())),
        "index_chunks": str(len(chunks)),
        "index_chunks_by_parser": json.dumps(dict(sorted(parsers.items()))),
    }


def make_configs(settings: Settings, names: Sequence[str]) -> list[EvalConfig]:
    configs = []
    common = {"embedding_model": settings.embedding_model, "rrf_k": "60", "k_each": "20"}
    # One cross-encoder per model (loaded on first use), shared by its candidate counts.
    cross_encoders: dict[str, FastEmbedCrossEncoder] = {}

    def cross_encoder_for(model: str) -> FastEmbedCrossEncoder:
        if model not in cross_encoders:
            cross_encoders[model] = FastEmbedCrossEncoder(model, cache_dir=str(settings.models_dir))
        return cross_encoders[model]

    cross_encoder = cross_encoder_for(settings.reranker_model)
    # (config name prefix, chunk store, Qdrant collection, parser) for each v2 index
    v2_indexes = [("", settings.data_index_dir, settings.qdrant_collection, MAIN_PARSER)]
    for variant in INDEX_VARIANTS:
        v2_indexes.append(
            (
                f"{variant}_",
                variant_store_dir(settings.data_index_dir, variant),
                variant_collection(settings.qdrant_collection, variant),
                VARIANT_PARSER[variant],
            )
        )
    for prefix, store_dir, collection, parser in v2_indexes:
        modes = [mode for mode in SEARCH_MODES if prefix + mode in names]
        # Reranker variants run on the main index only.
        rerank_variants = [name for name in RERANK_VARIANTS if not prefix and name in names]
        if not modes and not rerank_variants:
            continue
        store = IndexStore(store_dir)
        dense = make_dense_index(settings.model_copy(update={"qdrant_collection": collection}))
        index = SearchIndex(store, dense, cross_encoder)
        index_params = {**common, **_index_params(store), "bm25_idf": "lucene"}
        index_params |= {"qdrant_collection": collection, "parser": parser}
        for mode in modes:
            params = {**index_params, "mode": mode}
            if mode == "hybrid_rerank":
                params |= {"reranker_model": settings.reranker_model, "rerank_candidates": "20"}
            configs.append(EvalConfig(prefix + mode, index.retriever(mode), params))
        for name in rerank_variants:
            model, candidates = RERANK_VARIANTS[name]
            params = {**index_params, "mode": "hybrid_rerank", "reranker_model": model}
            params["rerank_candidates"] = str(candidates)
            retriever = RerankRetriever(
                index.retriever("hybrid"), cross_encoder_for(model), candidates=candidates
            )
            configs.append(EvalConfig(name, retriever, params))
    for config in V1_CONFIGS:
        if config not in names:
            continue
        collection = v1_collection(settings.qdrant_collection, config)
        store = IndexStore(v1_store_dir(settings.data_index_dir, config))
        dense = make_dense_index(settings.model_copy(update={"qdrant_collection": collection}))
        params = {**common, **_index_params(store), "mode": "v1_hybrid", "bm25_idf": "okapi"}
        params["qdrant_collection"] = collection
        configs.append(EvalConfig(config, make_v1_retriever(store, dense), params))
    return configs


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], check=False).returncode != 0
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval over eval/questions.jsonl.")
    parser.add_argument("--config", choices=ALL_CONFIGS, action="append", help="repeatable")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/retrieval.md"))
    parser.add_argument("--no-tracking", action="store_true", help="don't log to MLflow")
    args = parser.parse_args(argv)
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "alembic", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    names = args.config or list(ALL_CONFIGS)
    print("Validating the eval set against the manuals (about a minute)...", flush=True)
    pages = load_reference_pages(settings.data_raw_dir, PageCache(settings.data_parsed_dir))
    questions = load_eval_set(args.questions, pages)
    configs = make_configs(settings, names)
    for config in configs:
        if config.params["index_chunks"] == "0":
            print(
                f"{config.name}: index is empty; run make ingest / ingest-olmocr / ingest-v1",
                file=sys.stderr,
            )
            return 1
    tracker: Tracker = (
        NullTracker()
        if args.no_tracking
        else MlflowTracker(
            settings.mlflow_tracking_uri, "retrieval-eval", settings.mlflow_artifact_dir
        )
    )
    common = {
        "git_commit": _git_commit(),
        "eval_set": str(args.questions),
        "eval_set_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "questions": str(len(questions)),
        "results_k": str(RESULTS_K),
        "context_k": str(CONTEXT_K),
    }
    try:
        results = run_configs(configs, questions, tracker, common)
    except Exception as error:  # Qdrant down, model download failed...
        print(f"Eval failed: {error}", file=sys.stderr)
        return 1
    notes = [
        f"Generated by `make eval` on {date.today().isoformat()} at commit "
        f"`{common['git_commit']}`.",
        f"Eval set: `{args.questions}` (sha256 `{common['eval_set_sha256'][:12]}…`), "
        f"{len(questions)} questions. Ground truth = doc + PDF page.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, "Retrieval eval", notes), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
