"""Mine hard negatives for the training pairs (`make negatives`), locally, no LLM.

Reads `data/training/pairs/{train,val}.jsonl` (from `make pairs`), takes BM25 and the
served base embedder (EMBEDDING_MODEL, EMBEDDING_QUERY_PREFIX) over the trainable chunks,
and writes `data/training/mined/{train,val}.jsonl` with `stats.json`. Gitignored like the
pairs: it holds manual text.
"""

import argparse
import json
import sys
from pathlib import Path

from rig_assist.config import get_settings
from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.embedder import FastEmbedEmbedder
from rig_assist.training.negatives import (
    MAX_RELATIVE_SCORE,
    N_NEGATIVES,
    TOP_K,
    mine_negatives,
    mining_stats,
    write_mined,
)
from rig_assist.training.pairs import TrainingPair, eval_pages, trainable_chunks


def read_pairs(pairs_dir: Path) -> list[TrainingPair]:
    """train.jsonl then val.jsonl, each in its file's (shuffled) order."""
    return [
        TrainingPair(**json.loads(line))
        for split in ("train", "val")
        for line in (pairs_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hard negatives for the training pairs.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--training-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--n-negatives", type=int, default=N_NEGATIVES)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-relative-score", type=float, default=MAX_RELATIVE_SCORE)
    args = parser.parse_args(argv)
    settings = get_settings()

    pairs_dir = args.training_dir / "pairs"
    if not (pairs_dir / "train.jsonl").exists():
        print(f"No pairs in {pairs_dir}: run `make pairs` first.", file=sys.stderr)
        return 1
    pairs = read_pairs(pairs_dir)
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = trainable_chunks(IndexStore(settings.data_index_dir).chunks(), eval_pages(questions))
    print(f"{len(pairs)} pairs, {len(pool)} chunks in the pool", flush=True)

    embedder = FastEmbedEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        cache_dir=str(settings.models_dir),
    )
    mined = mine_negatives(
        pairs,
        pool,
        embedder,
        n_negatives=args.n_negatives,
        top_k=args.top_k,
        max_relative_score=args.max_relative_score,
    )
    out_dir = args.training_dir / "mined"
    stats = {
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "n_negatives": args.n_negatives,
        "top_k": args.top_k,
        "max_relative_score": args.max_relative_score,
        "rows": write_mined(mined, out_dir),
        **mining_stats(mined, args.n_negatives),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
