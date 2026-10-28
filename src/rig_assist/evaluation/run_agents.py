"""`make eval-agents`: run the scenarios through `/ask` and `/diagnose`, judge, report.

Both systems use the LLM from .env (one server for the whole run, as in the answer eval)
and the served retrieval. Runs and verdicts are cached in `data/eval/agents/` (manual text
inside, so outside Git) and the run resumes after an error. `/diagnose` runs are traced in
Langfuse when the keys are set. Report: `eval/results/agents.md`; MLflow experiment
`agent-eval`, one run per system.
"""

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

from rig_assist.agents import prompts as agent_prompts
from rig_assist.agents.graph import TroubleshootingGraph
from rig_assist.config import get_settings
from rig_assist.evaluation.agent_eval import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM_PROMPT,
    SYSTEMS,
    JudgeRecord,
    RunCache,
    Summary,
    System,
    SystemRun,
    VerdictCache,
    judge_run,
    run_ask,
    run_diagnose,
    summarize,
)
from rig_assist.evaluation.eval_set import load_reference_pages
from rig_assist.evaluation.run_retrieval import _git_commit
from rig_assist.evaluation.scenarios import Scenario, load_scenarios
from rig_assist.evaluation.tracking import MlflowTracker, NullTracker, Tracker
from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.store import IndexStore
from rig_assist.jobs import JobStore
from rig_assist.llm import LlmError, make_llm
from rig_assist.llm.usage import Prices
from rig_assist.rag import SYSTEM_PROMPT, RagPipeline
from rig_assist.retrieval.dense import make_dense_index
from rig_assist.retrieval.rerank import FastEmbedCrossEncoder
from rig_assist.retrieval.service import SearchIndex, SearchMode
from rig_assist.tracing import make_tracer

MODE: SearchMode = "hybrid_rerank"
TOP_K = 5
EVAL_RETRIES = 6
LABELS: dict[System, str] = {"ask": "/ask (single agent)", "diagnose": "/diagnose (multi-agent)"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def agent_prompts_sha() -> str:
    """Hash of every agent prompt: the `/diagnose` version a cached run was made with."""
    text = "".join(
        getattr(agent_prompts, name) for name in dir(agent_prompts) if name.endswith("_PROMPT")
    )
    return _sha(text)


def _ratio(count: int, n: int) -> str:
    return f"{count}/{n}"


def _num(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _summary_rows(summaries: Mapping[System, Summary]) -> list[str]:
    ask, diag = summaries["ask"], summaries["diagnose"]

    def row(label: str, a: str, d: str) -> str:
        return f"| {label} | {a} | {d} |"

    return [
        f"| metric | {LABELS['ask']} | {LABELS['diagnose']} |",
        "|---|---|---|",
        row(
            "Cause correct (judge 2/2)",
            _ratio(ask.cause_correct, ask.n),
            _ratio(diag.cause_correct, diag.n),
        ),
        row("Cause score, mean (0–2)", _num(ask.cause_mean), _num(diag.cause_mean)),
        row("Safety warnings, mean (0–2)", _num(ask.safety_mean), _num(diag.safety_mean)),
        row(
            "Cites an expected page",
            _ratio(ask.cites_expected_page, ask.n),
            _ratio(diag.cites_expected_page, diag.n),
        ),
        row(
            "Unsafe action endorsed",
            _ratio(ask.unsafe_endorsed, ask.unsafe_n),
            _ratio(diag.unsafe_endorsed, diag.unsafe_n),
        ),
        row(
            "Unsafe action flagged",
            _ratio(ask.unsafe_flagged, ask.unsafe_n),
            _ratio(diag.unsafe_flagged, diag.unsafe_n),
        ),
        row(
            "Unsafe action blocked (in `blocked_actions`)",
            "–",
            _ratio(diag.unsafe_blocked, diag.unsafe_n),
        ),
        row("Escalated to an engineer", "–", _ratio(diag.escalated, diag.n)),
        row("Failed runs", str(ask.failed), str(diag.failed)),
        row("Unparseable verdicts", str(ask.unparseable), str(diag.unparseable)),
        row(
            "Latency p50 / p95",
            f"{ask.p50_seconds:.1f} s / {ask.p95_seconds:.1f} s",
            f"{diag.p50_seconds:.1f} s / {diag.p95_seconds:.1f} s",
        ),
        row(
            "LLM calls per scenario, mean",
            f"{ask.mean_llm_calls:.1f}",
            f"{diag.mean_llm_calls:.1f}",
        ),
        row(
            "Tokens in / out (all scenarios)",
            f"{ask.input_tokens:,} / {ask.output_tokens:,}",
            f"{diag.input_tokens:,} / {diag.output_tokens:,}",
        ),
        row("Cost (all scenarios)", _cost(ask.cost_usd), _cost(diag.cost_usd)),
    ]


def _cost(value: float | None) -> str:
    return "– (prices not set)" if value is None else f"${value:.4f}"


def _cell(verdict: JudgeRecord) -> str:
    if verdict.outcome == "failed":
        return "failed"
    if verdict.outcome == "unparseable":
        return "?"
    text = f"{verdict.cause} / {verdict.safety}"
    if verdict.unsafe_endorsed:
        text += " ⚠ endorsed"
    elif verdict.unsafe_flagged:
        text += " ✓ flagged"
    return text


def render_report(
    scenarios: Sequence[Scenario],
    runs: Mapping[System, Sequence[SystemRun]],
    verdicts: Mapping[System, Sequence[JudgeRecord]],
    prices: Prices | None,
    notes: Sequence[str] = (),
) -> str:
    summaries = {s: summarize(scenarios, runs[s], verdicts[s], prices) for s in SYSTEMS}
    lines = ["# /ask vs /diagnose", "", *notes, "", "## Summary", ""]
    lines += _summary_rows(summaries)
    lines += [
        "",
        "## Per scenario",
        "",
        "Judge scores as cause / safety (0–2 each). ⚠ = the response endorsed the unsafe "
        "action, ✓ = it explicitly warned against or blocked it.",
        "",
        f"| id | kind | {LABELS['ask']} | {LABELS['diagnose']} "
        "| /diagnose blocked | /diagnose s | trace |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, scenario in enumerate(scenarios):
        diag = runs["diagnose"][i]
        blocked = "; ".join(diag.blocked_actions) or "–"
        trace = f"[trace]({diag.trace_url})" if diag.trace_url else "–"
        lines.append(
            f"| {scenario.id} | {scenario.kind} | {_cell(verdicts['ask'][i])} | "
            f"{_cell(verdicts['diagnose'][i])} | {blocked} | {diag.seconds:.1f} | {trace} |"
        )
    lines += ["", "## Judge reasons", ""]
    for i, scenario in enumerate(scenarios):
        lines.append(f"- **{scenario.id}** ({scenario.expected_cause})")
        for system in SYSTEMS:
            lines.append(f"  - {LABELS[system]}: {verdicts[system][i].reason}")
    return "\n".join(lines) + "\n"


def _metrics(summary: Summary) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(summary).items() if isinstance(v, int | float)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="/ask vs /diagnose over eval/scenarios.jsonl.")
    parser.add_argument("--scenarios", type=Path, default=Path("eval/scenarios.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/agents.md"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/eval/agents"))
    parser.add_argument("--only", action="append", help="scenario id to run (repeatable)")
    parser.add_argument("--no-tracking", action="store_true", help="don't log to MLflow")
    args = parser.parse_args(argv)
    settings = get_settings().model_copy(update={"llm_max_retries": EVAL_RETRIES})
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "alembic", "mlflow", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if settings.llm_provider == "fake":
        print("LLM_PROVIDER=fake: set a real LLM in .env for the agent eval.", file=sys.stderr)
        return 1

    print("Validating the scenarios against the manuals (about a minute)...", flush=True)
    pages = load_reference_pages(settings.data_raw_dir, PageCache(settings.data_parsed_dir))
    scenarios = load_scenarios(args.scenarios, pages)
    if args.only:
        scenarios = [s for s in scenarios if s.id in set(args.only)]
    cross_encoder = FastEmbedCrossEncoder(
        settings.reranker_model, cache_dir=str(settings.models_dir)
    )
    index = SearchIndex(
        IndexStore(settings.data_index_dir), make_dense_index(settings), cross_encoder
    )
    retriever = index.retriever(MODE)
    llm = make_llm(settings)
    tracer = make_tracer(settings)
    pipeline = RagPipeline(retriever, llm, top_k=TOP_K, prices=settings.llm_prices)
    graph = TroubleshootingGraph(llm, retriever, tracer=tracer)
    config = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_thinking": settings.llm_thinking,
        "mode": MODE,
        "top_k": str(TOP_K),
        "reranker_model": settings.reranker_model,
        "rag_prompt_sha256": _sha(SYSTEM_PROMPT),
        "agent_prompts_sha256": agent_prompts_sha(),
    }
    judge_config = {
        "judge_model": settings.llm_model,
        "judge_max_tokens": str(JUDGE_MAX_TOKENS),
        "judge_prompt_sha256": _sha(JUDGE_SYSTEM_PROMPT),
        "runs_config_sha256": _sha(json.dumps(config, sort_keys=True)),
    }
    run_cache = RunCache(args.cache_dir / "runs.jsonl", config)
    verdict_cache = VerdictCache(args.cache_dir / "verdicts.jsonl", judge_config)
    store = JobStore(args.cache_dir / "jobs.sqlite")
    runs: dict[System, list[SystemRun]] = {s: [] for s in SYSTEMS}
    verdicts: dict[System, list[JudgeRecord]] = {s: [] for s in SYSTEMS}
    try:
        done = run_cache.load()
        judged = verdict_cache.load()
        for scenario in scenarios:
            for system in SYSTEMS:
                run = done.get((scenario.id, system))
                if run is None:
                    if system == "ask":
                        run = run_ask(pipeline.ask, scenario, config)
                    else:
                        run = run_diagnose(
                            graph, store, scenario, config, settings.llm_prices, tracer
                        )
                    run_cache.append(run)
                print(
                    f"{scenario.id} {system}: {run.seconds:.1f} s"
                    + (f" FAILED {run.error}" if run.error else ""),
                    flush=True,
                )
                runs[system].append(run)
                verdict = judged.get((scenario.id, system))
                if verdict is None:
                    verdict = judge_run(llm, scenario, run, judge_config)
                    verdict_cache.append(verdict)
                verdicts[system].append(verdict)
    except LlmError as error:
        print(
            f"Stopped: {error}. Progress is saved in {args.cache_dir}; run again to continue.",
            file=sys.stderr,
        )
        return 1
    finally:
        tracer.flush()

    commit = _git_commit()
    scenarios_sha = hashlib.sha256(args.scenarios.read_bytes()).hexdigest()
    notes = [
        f"Generated by `make eval-agents` on {date.today().isoformat()} at commit `{commit}`.",
        f"Scenarios: `{args.scenarios}` (sha256 `{scenarios_sha[:12]}…`), {len(scenarios)} "
        f"({sum(s.unsafe_action is not None for s in scenarios)} unsafe temptations).",
        f"Both systems: LLM `{settings.llm_provider}` / `{settings.llm_model}` (thinking "
        f"`{settings.llm_thinking}`), retrieval "
        f"`{MODE}` ({settings.reranker_model}). `/ask`: top {TOP_K} chunks, one call. "
        "`/diagnose`: the LangGraph agents through the job runner, as served.",
        f"Judge: `{settings.llm_model}` (prompt sha256 `{judge_config['judge_prompt_sha256']}`), "
        "not told which system wrote the response. It is the model that wrote both, so "
        "self-preference affects both systems alike; verdicts were spot-checked by hand.",
    ]
    report = render_report(scenarios, runs, verdicts, settings.llm_prices, notes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    tracker: Tracker = (
        NullTracker()
        if args.no_tracking
        else MlflowTracker(settings.mlflow_tracking_uri, "agent-eval", settings.mlflow_artifact_dir)
    )
    params = {**config, **judge_config, "git_commit": commit, "scenarios_sha256": scenarios_sha}
    for system in SYSTEMS:
        summary = summarize(scenarios, runs[system], verdicts[system], settings.llm_prices)
        tracker.log_run(
            f"{system}_{settings.llm_model}",
            params | {"system": system},
            _metrics(summary),
            {"agents.md": report},
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
