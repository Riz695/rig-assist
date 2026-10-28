"""`make eval-safety`: blocking precision of the `/diagnose` safety node, run against run.

Reads `/diagnose` runs already cached by `make eval-agents` (no new runs here), judges every
blocked action with the block judge (cached in `<cache dir>/blocks.jsonl`), and writes a
side-by-side report. Each `--run LABEL=CACHE_DIR[:PROMPTS_SHA]` names one cached run;
the hash picks one prompt version when a cache holds several (`current` = the agent prompts
of the checked-out code). All runs in one report must be on the same scenario set.
MLflow experiment `safety-eval`, one run per label.
"""

import argparse
import hashlib
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from rig_assist.config import get_settings
from rig_assist.evaluation.agent_eval import JUDGE_MAX_TOKENS, SystemRun
from rig_assist.evaluation.block_judge import (
    BLOCK_JUDGE_PROMPT,
    BlockCache,
    BlockRecord,
    BlockSummary,
    judge_blocks,
    load_diagnose_runs,
    summarize_blocks,
)
from rig_assist.evaluation.eval_set import load_reference_pages
from rig_assist.evaluation.run_agents import EVAL_RETRIES, _sha, agent_prompts_sha
from rig_assist.evaluation.run_retrieval import _git_commit
from rig_assist.evaluation.scenarios import Scenario, load_scenarios
from rig_assist.evaluation.tracking import MlflowTracker, NullTracker, Tracker
from rig_assist.ingestion.cache import PageCache
from rig_assist.llm import LlmError, make_llm

Runs = Mapping[str, SystemRun]
Records = Mapping[tuple[str, str], BlockRecord]


@dataclass(frozen=True)
class RunSpec:
    label: str
    cache_dir: Path
    prompts_sha: str | None  # None: the cache must hold one prompt version


def parse_run(text: str, current_sha: str | None = None) -> RunSpec:
    """`LABEL=CACHE_DIR[:PROMPTS_SHA]`; the hash `current` becomes `current_sha`."""
    label, sep, rest = text.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"--run {text!r}: expected LABEL=CACHE_DIR[:PROMPTS_SHA]")
    cache_dir, colon, sha = rest.partition(":")
    if not cache_dir or (colon and not sha):
        raise ValueError(f"--run {text!r}: expected LABEL=CACHE_DIR[:PROMPTS_SHA]")
    if sha == "current":
        sha = current_sha or agent_prompts_sha()
    return RunSpec(label, Path(cache_dir), sha or None)


def _precision(summary: BlockSummary) -> str:
    judged = summary.judged_unsafe + summary.judged_safe
    if judged == 0:
        return "0/0 (–)"
    return f"{summary.judged_unsafe}/{judged} ({summary.judged_unsafe / judged:.0%})"


def _judgment(record: BlockRecord) -> str:
    if record.outcome == "unparseable":
        return "BLOCKED (judgment unparseable)"
    verdict = "judged unsafe" if record.unsafe_as_written else "judged safe"
    if record.is_the_temptation:
        verdict += ", the temptation"
    return f"BLOCKED ({verdict})"


def render_block_report(
    scenarios: Sequence[Scenario],
    runs: Mapping[str, Runs],
    records: Mapping[str, Records],
    notes: Sequence[str] = (),
) -> str:
    """Runs side by side (labels in the order given), then every block with its judgment."""
    labels = list(runs)
    summaries = {
        label: summarize_blocks(scenarios, runs[label], records[label]) for label in labels
    }

    def row(name: str, cell: Callable[[BlockSummary], str]) -> str:
        return f"| {name} | " + " | ".join(cell(summaries[label]) for label in labels) + " |"

    lines = ["# Safety node: blocking precision", "", *notes, "", "## Summary", ""]
    lines += [
        "| metric | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
        row("Blocks", lambda s: str(s.blocks)),
        row("Judged unsafe (precision)", _precision),
        row("Safe actions blocked", lambda s: str(s.judged_safe)),
        row("Temptation blocked (of n)", lambda s: f"{s.temptation_blocked}/{s.temptations}"),
        row("Failed runs", lambda s: str(s.failed)),
        row("Unparseable judgments", lambda s: str(s.unparseable)),
    ]
    lines += [
        "",
        "Precision = blocks judged unsafe as written / blocks judged. The judge is the model "
        "that wrote the work orders; every block is also audited by hand.",
        "",
        "## Every block",
    ]
    for scenario in scenarios:
        tempted = f" (tempted: {scenario.unsafe_action})" if scenario.unsafe_action else ""
        lines += ["", f"### {scenario.id}: {scenario.kind}{tempted}", ""]
        for label in labels:
            run = runs[label][scenario.id]
            if run.error is not None:
                lines.append(f"- **{label}**: failed: {run.error}")
            elif not run.blocked_actions:
                lines.append(f"- **{label}**: no blocks")
            else:
                lines.append(f"- **{label}**:")
                for action in run.blocked_actions:
                    record = records[label][(scenario.id, action)]
                    lines.append(f"  - {_judgment(record)}: {action}: {record.reason}")
    return "\n".join(lines) + "\n"


def _metrics(summary: BlockSummary) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(summary).items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blocking precision across cached /diagnose runs.")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CACHE_DIR[:PROMPTS_SHA]",
        help="a cached /diagnose run (repeatable); PROMPTS_SHA 'current' = this code's prompts",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-tracking", action="store_true", help="don't log to MLflow")
    args = parser.parse_args(argv)
    settings = get_settings().model_copy(update={"llm_max_retries": EVAL_RETRIES})
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "httpx2", "alembic", "mlflow", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        specs = [parse_run(text) for text in args.run]
    except ValueError as error:
        parser.error(str(error))
    if len({spec.label for spec in specs}) != len(specs):
        parser.error("--run labels must be unique")
    if settings.llm_provider == "fake":
        print("LLM_PROVIDER=fake: set a real LLM in .env for the block judge.", file=sys.stderr)
        return 1

    print("Validating the scenarios against the manuals (about a minute)...", flush=True)
    pages = load_reference_pages(settings.data_raw_dir, PageCache(settings.data_parsed_dir))
    scenarios = load_scenarios(args.scenarios, pages)
    llm = make_llm(settings)
    runs: dict[str, Runs] = {}
    records: dict[str, Records] = {}
    configs: dict[str, dict[str, str]] = {}
    try:
        for spec in specs:
            runs[spec.label] = load_diagnose_runs(spec.cache_dir / "runs.jsonl", spec.prompts_sha)
            shas = {r.config.get("agent_prompts_sha256", "") for r in runs[spec.label].values()}
            configs[spec.label] = {
                "judge_model": settings.llm_model,
                "judge_max_tokens": str(JUDGE_MAX_TOKENS),
                "block_judge_prompt_sha256": _sha(BLOCK_JUDGE_PROMPT),
                # the judge reads the whole work order: never reuse a judgment across versions
                "agent_prompts_sha256": ",".join(sorted(shas)),
            }
            cache = BlockCache(spec.cache_dir / "blocks.jsonl", configs[spec.label])
            records[spec.label] = judge_blocks(
                llm, scenarios, runs[spec.label], cache, configs[spec.label]
            )
            print(f"{spec.label}: {len(records[spec.label])} blocks judged", flush=True)
    except LlmError as error:
        print(f"Stopped: {error}. Judgments so far are saved; run again.", file=sys.stderr)
        return 1

    commit = _git_commit()
    scenarios_sha = hashlib.sha256(args.scenarios.read_bytes()).hexdigest()
    notes = [
        f"Generated by `make eval-safety` on {date.today().isoformat()} at commit `{commit}`.",
        f"Scenarios: `{args.scenarios}` (sha256 `{scenarios_sha[:12]}…`), {len(scenarios)} "
        f"({sum(s.unsafe_action is not None for s in scenarios)} unsafe temptations).",
        "Runs: "
        + "; ".join(
            f"**{spec.label}** = `{spec.cache_dir}` (agent prompts "
            f"`{configs[spec.label]['agent_prompts_sha256']}`)"
            for spec in specs
        )
        + ".",
        f"Block judge: `{settings.llm_model}` (prompt sha256 "
        f"`{_sha(BLOCK_JUDGE_PROMPT)}`), not told which run wrote the work order.",
    ]
    report = render_block_report(scenarios, runs, records, notes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    tracker: Tracker = (
        NullTracker()
        if args.no_tracking
        else MlflowTracker(
            settings.mlflow_tracking_uri, "safety-eval", settings.mlflow_artifact_dir
        )
    )
    for spec in specs:
        summary = summarize_blocks(scenarios, runs[spec.label], records[spec.label])
        params = configs[spec.label] | {
            "label": spec.label,
            "cache_dir": str(spec.cache_dir),
            "scenarios": str(args.scenarios),
            "scenarios_sha256": scenarios_sha,
            "labels": ",".join(s.label for s in specs),
            "git_commit": commit,
        }
        tracker.log_run(
            f"{spec.label}_{args.scenarios.stem}",
            params,
            _metrics(summary),
            {args.out.name: report},
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
