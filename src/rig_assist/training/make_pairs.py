"""Build the fine-tuning data: synthetic queries per chunk, then (query, chunk) pairs.

`make pairs` runs `main()`: the LLM from .env writes queries for every trainable chunk of the
main index (eval pages excluded), cached in `data/training/queries.jsonl` so a stopped run
resumes; then the pairs go to `data/training/pairs/{train,val}.jsonl` with `stats.json`.
Everything under `data/training/` holds manual text, so it stays out of Git.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rig_assist.config import get_settings
from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.ingestion.store import IndexStore
from rig_assist.llm import LlmError, make_llm
from rig_assist.training.pairs import (
    VAL_FRACTION,
    TrainingPair,
    build_pairs,
    eval_pages,
    trainable_chunks,
)
from rig_assist.training.queries import (
    QUERIES_PER_CHUNK,
    QUERY_MAX_TOKENS,
    QUERY_SYSTEM_PROMPT,
    QueryCache,
    generate_queries,
    queries_by_chunk,
    query_stats,
)


def write_pairs(pairs: Sequence[TrainingPair], out_dir: Path) -> dict[str, int]:
    """`train.jsonl` and `val.jsonl` in the given (shuffled) order; returns pairs per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for split in ("train", "val"):
        rows = [asdict(p) for p in pairs if p.split == split]
        counts[split] = len(rows)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (out_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
    return dict(counts)


def _print(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic queries + training pairs.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--workers", type=int, default=16, help="LLM requests in flight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = parser.parse_args(argv)
    settings = get_settings()
    if settings.llm_provider == "fake":
        print("LLM_PROVIDER=fake: set a real LLM in .env to write queries.", file=sys.stderr)
        return 1

    chunks = IndexStore(settings.data_index_dir).chunks()
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    excluded = eval_pages(questions)
    todo = trainable_chunks(chunks, excluded)
    _print(f"{len(chunks)} chunks, {len(excluded)} eval pages excluded, {len(todo)} trainable")

    config = {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "queries_per_chunk": str(QUERIES_PER_CHUNK),
        "max_tokens": str(QUERY_MAX_TOKENS),
        "prompt_sha256": hashlib.sha256(QUERY_SYSTEM_PROMPT.encode()).hexdigest()[:12],
    }
    cache = QueryCache(args.out_dir / "queries.jsonl", config)
    try:
        records = generate_queries(
            make_llm(settings), todo, cache, workers=args.workers, progress=_print
        )
    except LlmError as error:
        print(
            f"Stopped: {error}. Queries so far are cached; run again to continue.", file=sys.stderr
        )
        return 1

    pairs = build_pairs(
        chunks, queries_by_chunk(records), excluded, val_fraction=args.val_fraction, seed=args.seed
    )
    counts = write_pairs(pairs, args.out_dir / "pairs")
    stats = {
        "config": config,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "eval_pages_excluded": len(excluded),
        "trainable_chunks": len(todo),
        "queries": query_stats(list(records.values())),
        "pairs": counts,
        "chunks_with_pairs": len({p.chunk_id for p in pairs}),
    }
    (args.out_dir / "pairs" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    _print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
