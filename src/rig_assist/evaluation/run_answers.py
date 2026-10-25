"""Answer the eval set with the served RAG pipeline, compare "not found" rules, judge answers.

`make eval-answers` runs `main()`: the `/ask` pipeline as served (hybrid_rerank, top 5,
the LLM from .env), one call per question, cached in `data/eval/answers/` so a run that
hits a rate limit resumes. Then the same LLM judges each answer (`judge.py`), cached the
same way. The report goes to `eval/results/answers.md` and MLflow (experiment
`answer-eval`). The caches hold manual text, so they stay out of Git.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from rig_assist.config import get_settings
from rig_assist.evaluation.answer_eval import (
    AnswerCache,
    AnswerRecord,
    answer_stats,
    generate_answers,
    refusal_strategies,
    safe_threshold,
)
from rig_assist.evaluation.eval_set import load_eval_set, load_reference_pages
from rig_assist.evaluation.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    JudgeCache,
    JudgeRecord,
    judge_answers,
    judge_stats,
)
from rig_assist.evaluation.metrics import RefusalCase, RefusalResult
from rig_assist.evaluation.retrieval_eval import choose_threshold
from rig_assist.evaluation.run_retrieval import _git_commit
from rig_assist.evaluation.tracking import MlflowTracker, NullTracker, Tracker
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.store import IndexStore
from rig_assist.llm import LlmError, make_llm
from rig_assist.rag import MAX_ANSWER_TOKENS, SYSTEM_PROMPT, RagPipeline
from rig_assist.retrieval.dense import make_dense_index
from rig_assist.retrieval.rerank import FastEmbedCrossEncoder
from rig_assist.retrieval.service import SearchIndex, SearchMode

MODE: SearchMode = "hybrid_rerank"
TOP_K = 5
EVAL_RETRIES = 6  # more patient than the API's default: free tiers return 429/503 often
STRATEGY_LABELS = {
    "llm": "LLM only (the pipeline's own refusals)",
    "reranker": "Reranker gate only (LLM not called below it)",
    "llm_or_reranker": "LLM or reranker gate",
}


def _fmt(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _strategy_rows(records: Sequence[AnswerRecord], threshold: float, label: str) -> list[str]:
    rows = []
    for name, result in refusal_strategies(records, threshold).items():
        if name == "llm" and label != "tuned":
            continue  # the LLM row doesn't depend on the threshold; shown once
        shown = STRATEGY_LABELS[name] + ("" if name == "llm" else f" @ {threshold:.2f} ({label})")
        rows.append(_strategy_row(shown, result))
    return rows


def _strategy_row(label: str, r: RefusalResult) -> str:
    return (
        f"| {label} | {r.true_pos} | {r.false_pos} | {r.false_neg} | {r.true_neg} "
        f"| {_fmt(r.precision)} | {_fmt(r.recall)} |"
    )


def _histogram_row(label: str, counts: dict[int, int]) -> str:
    return f"| {label} | {counts[2]} | {counts[1]} | {counts[0]} |"


def _judge_section(verdicts: Sequence[JudgeRecord]) -> list[str]:
    stats = judge_stats(verdicts)
    return [
        "## LLM judge",
        "",
        "Scores 0–2 from the judge prompt in `src/rig_assist/evaluation/judge.py`. "
        "Correctness compares the answer with the eval set's reference answer; faithfulness "
        "checks every claim against the context the RAG LLM was given. Refusals are scored "
        "by rule, without the judge: an answerable question refused is 0, an unanswerable one "
        "refused is 2, and a refusal claims nothing, so it has no faithfulness score.",
        "",
        "| Scale | 2 | 1 | 0 |",
        "|---|---|---|---|",
        _histogram_row(
            "Correctness, answerable questions (refused = 0)", stats.answerable_correctness
        ),
        _histogram_row(
            "Correctness, unanswerable questions (refused = 2)", stats.unanswerable_correctness
        ),
        _histogram_row("Faithfulness, answers judged", stats.faithfulness),
        "",
        f"Unparseable verdicts (invalid twice): {stats.unparseable}. Judge calls: "
        f"{stats.judge_calls} ({stats.input_tokens:,} input / {stats.output_tokens:,} "
        "output tokens).",
        "",
    ]


def render_report(
    records: Sequence[AnswerRecord],
    notes: Sequence[str] = (),
    verdicts: Sequence[JudgeRecord] | None = None,
) -> str:
    stats = answer_stats(records)
    cases = [RefusalCase(r.top_score, r.answerable) for r in records]
    tuned = choose_threshold(cases)
    safe = safe_threshold(records)
    lines = ["# Answer eval: single-agent RAG (`/ask`)", "", *notes, ""]
    lines += [
        "## Answers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answerable questions answered | {stats.answered}/{stats.answerable} |",
        f"| … citing an expected page | {stats.cited_expected_page}/{stats.answerable} |",
        f"| Unanswerable questions answered anyway | "
        f"{stats.unanswerable_answered}/{stats.unanswerable} |",
        "| Refusal reasons | "
        + (", ".join(f"{k} {v}" for k, v in stats.refusal_reasons.items()) or "none")
        + " |",
        f"| Tokens (input / output) | {stats.input_tokens:,} / {stats.output_tokens:,} |",
        "",
        '"Citing an expected page" is a cheap correctness proxy (the answer cites a page '
        "the eval set marks as containing the answer); the LLM judge measures correctness "
        "and faithfulness properly.",
        "",
        '## "Not found" rules',
        "",
        "Refused = positive class. TP: unanswerable refused; FP: answerable refused; "
        "FN: unanswerable answered; TN: answerable answered.",
        "",
        "| Rule | TP | FP | FN | TN | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
        *_strategy_rows(records, tuned, "tuned"),
        *_strategy_rows(records, safe, "safe"),
        "",
        f"Reranker score = best cross-encoder logit among the {TOP_K} context chunks. "
        f"**Tuned** = best refusal F1 on this same set ({stats.unanswerable} unanswerable "
        "questions), so optimistic. **Safe** = the lowest score of any answerable question, "
        "the highest threshold that refuses none of them on this set.",
        "",
    ]
    if verdicts is not None:
        lines += _judge_section(verdicts)
    header = (
        "| id | answerable | found | refusal reason | top reranker score | cites expected page |"
    )
    if verdicts is not None:
        header += " faithfulness | correctness |"
    lines += ["## Per question", "", header, "|---" * header.count(" |") + "|"]
    by_id = {v.id: v for v in verdicts or []}
    for r in records:
        row = (
            f"| {r.id} | {'yes' if r.answerable else 'no'} | {'yes' if r.found else 'no'} "
            f"| {r.refusal_reason or ''} | {_fmt(r.top_score)} "
            f"| {'yes' if r.cited_expected_page else ('' if not r.answerable else 'no')} |"
        )
        if verdicts is not None:
            v = by_id.get(r.id)
            faith = "–" if v is None or v.faithfulness is None else str(v.faithfulness)
            correct = "–" if v is None or v.correctness is None else str(v.correctness)
            row += f" {faith} | {correct} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def _judge_metrics(verdicts: Sequence[JudgeRecord]) -> dict[str, float]:
    stats = judge_stats(verdicts)
    metrics = {
        "judge_unparseable": float(stats.unparseable),
        "judge_calls": float(stats.judge_calls),
        "judge_input_tokens": float(stats.input_tokens),
        "judge_output_tokens": float(stats.output_tokens),
    }
    for name, counts in (
        ("correctness_answerable", stats.answerable_correctness),
        ("correctness_unanswerable", stats.unanswerable_correctness),
        ("faithfulness", stats.faithfulness),
    ):
        for score, count in counts.items():
            metrics[f"judge_{name}_{score}"] = float(count)
    return metrics


def _metrics(records: Sequence[AnswerRecord]) -> dict[str, float]:
    stats = answer_stats(records)
    metrics = {
        "answered": float(stats.answered),
        "cited_expected_page": float(stats.cited_expected_page),
        "unanswerable_answered": float(stats.unanswerable_answered),
        "input_tokens": float(stats.input_tokens),
        "output_tokens": float(stats.output_tokens),
    }
    tuned = choose_threshold([RefusalCase(r.top_score, r.answerable) for r in records])
    metrics["reranker_threshold_tuned"] = tuned
    for name, result in refusal_strategies(records, tuned).items():
        for metric in ("precision", "recall"):
            value = getattr(result, metric)
            if value is not None:
                metrics[f"refusal_{metric}_{name}"] = value
    return metrics


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer eval over eval/questions.jsonl.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/answers.md"))
    parser.add_argument("--cache", type=Path, default=Path("data/eval/answers/answers.jsonl"))
    parser.add_argument("--judge-cache", type=Path, default=Path("data/eval/answers/judge.jsonl"))
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--no-tracking", action="store_true", help="don't log to MLflow")
    args = parser.parse_args(argv)
    settings = get_settings().model_copy(update={"llm_max_retries": EVAL_RETRIES})
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "alembic", "mlflow", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if settings.llm_provider == "fake":
        print("LLM_PROVIDER=fake: set a real LLM in .env for the answer eval.", file=sys.stderr)
        return 1

    print("Validating the eval set against the manuals (about a minute)...", flush=True)
    pages = load_reference_pages(settings.data_raw_dir, PageCache(settings.data_parsed_dir))
    questions = load_eval_set(args.questions, pages)
    store = IndexStore(settings.data_index_dir)
    cross_encoder = FastEmbedCrossEncoder(
        settings.reranker_model, cache_dir=str(settings.models_dir)
    )
    index = SearchIndex(store, make_dense_index(settings), cross_encoder)
    llm = make_llm(settings)
    pipeline = RagPipeline(index.retriever(MODE), llm, top_k=TOP_K, prices=settings.llm_prices)
    config = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "mode": MODE,
        "top_k": str(TOP_K),
        "reranker_model": settings.reranker_model,
        "max_answer_tokens": str(MAX_ANSWER_TOKENS),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
    }
    cache = AnswerCache(args.cache, config)
    # The judge is the same LLM (one server for the eval); its verdicts are only reused for
    # the same answers, judge model and judge prompt.
    judge_config = {
        "judge_provider": settings.llm_provider,
        "judge_model": settings.llm_model,
        "judge_max_tokens": str(JUDGE_MAX_TOKENS),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "answers_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12],
    }
    verdicts: list[JudgeRecord] | None = None
    try:
        records = generate_answers(pipeline.ask, questions, cache, progress=_print)
        if not args.no_judge:
            print("Judging the answers...", flush=True)
            answers = {r.id: r for r in records}
            judge_cache = JudgeCache(args.judge_cache, judge_config)
            verdicts = judge_answers(llm, questions, answers, judge_cache, progress=_print)
    except LlmError as error:
        print(
            f"Stopped: {error}. Answers and verdicts so far are saved in {args.cache.parent}; "
            "run again to continue (e.g. after a rate limit resets).",
            file=sys.stderr,
        )
        return 1

    commit = _git_commit()
    eval_sha = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    notes = [
        f"Generated by `make eval-answers` on {date.today().isoformat()} at commit `{commit}`.",
        f"Pipeline: `/ask` as served: `{MODE}` ({settings.reranker_model}), top {TOP_K} "
        f"chunks, LLM `{settings.llm_provider}` / `{settings.llm_model}`, no relevance gate.",
        f"Eval set: `{args.questions}` (sha256 `{eval_sha[:12]}…`), {len(questions)} questions.",
    ]
    if verdicts is not None:
        notes.append(
            f"Judge: `{settings.llm_model}`, prompt sha256 "
            f"`{judge_config['judge_prompt_sha256']}`. The judge is the model that wrote the "
            "answers, so it may favour them (self-preference)."
        )
    report = render_report(records, notes, verdicts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    tracker: Tracker = (
        NullTracker()
        if args.no_tracking
        else MlflowTracker(
            settings.mlflow_tracking_uri, "answer-eval", settings.mlflow_artifact_dir
        )
    )
    params = {**config, "git_commit": commit, "eval_set_sha256": eval_sha}
    metrics = _metrics(records)
    if verdicts is not None:
        params |= judge_config
        metrics |= _judge_metrics(verdicts)
    run_id = tracker.log_run("rag_" + settings.llm_model, params, metrics, {"answers.md": report})
    print(f"Wrote {args.out}" + (f" (MLflow run {run_id})" if run_id else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
