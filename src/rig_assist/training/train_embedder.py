"""Fine-tune the embedder on the mined pairs (`make train`).

Loss: MultipleNegativesRankingLoss (InfoNCE). In each batch a query must pick its own
chunk out of every chunk in the batch, plus each row's hard negative. The cached variant
(GradCache) computes it in mini-batches: a batch of 32 triplets at 512 tokens ran out of
memory on a 14 GB laptop; the loss and the negatives are the same, only slower. Rows come from
`make negatives` (`data/training/mined/`); a row with no negative is left out when hard
negatives are on, since every row of a triplet dataset needs all three columns.

The batch sampler is NO_DUPLICATES: a chunk has three queries, and if two of them shared a
batch, the chunk would be both the right answer and an in-batch "wrong" one.

Model selection uses the validation split only (MRR@10 of the synthetic validation queries
against every trainable chunk). The frozen eval set is not looked at here: it is the final
test (`eval/results/finetune.md`), and choosing a checkpoint on it would leak it.

torch and sentence-transformers live in the `train` dependency group and MLflow in `eval`,
so they are imported inside `fit` and `log_run`; the data preparation is plain Python.
A run writes a self-contained folder (`run.json` + `model/`); `log_run` records it in
MLflow, locally right after training, or later (`--log-run DIR`) for a run on a rented GPU.
"""

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rig_assist.config import get_settings
from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.ingestion.models import Chunk
from rig_assist.ingestion.store import IndexStore
from rig_assist.training.negatives import MinedPair
from rig_assist.training.pairs import eval_pages, trainable_chunks

BASE_MODEL = "BAAI/bge-small-en-v1.5"
REGISTERED_NAME = "rig-assist-embedder"


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = BASE_MODEL
    epochs: int = 1
    batch_size: int = 32  # in-batch negatives per query = batch_size - 1 (+ hard negatives)
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    seed: int = 0
    hard_negatives: bool = True
    query_prefix: str = ""  # the served config uses none (Phase 3)
    mini_batch_size: int = 8  # rows embedded at once by the cached loss: memory, not the loss
    max_seq_length: int = 512  # the longest chunk is 482 tokens; median 348

    def __post_init__(self) -> None:
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 (in-batch negatives)")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 1 <= self.mini_batch_size <= self.batch_size:
            raise ValueError("mini_batch_size must be between 1 and batch_size")

    def params(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def load_mined(mined_dir: Path) -> list[MinedPair]:
    """Rows of `train.jsonl` then `val.jsonl`."""
    return [
        MinedPair(**json.loads(line))
        for split in ("train", "val")
        for line in (mined_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def train_columns(
    rows: Sequence[MinedPair], hard_negatives: bool, query_prefix: str = ""
) -> dict[str, list[str]]:
    """Columns of the training dataset (train split only, in the given order)."""
    train = [r for r in rows if r.split == "train" and (r.negatives or not hard_negatives)]
    columns = {
        "anchor": [query_prefix + r.query for r in train],
        "positive": [r.positive for r in train],
    }
    if hard_negatives:
        columns["negative"] = [r.negatives[0] for r in train]
    return columns


def ir_eval_data(
    rows: Sequence[MinedPair], pool: Sequence[Chunk], query_prefix: str = ""
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """(queries, corpus, relevant) for validation: val queries against the whole pool."""
    corpus = {c.chunk_id: c.text for c in pool}
    val = [r for r in rows if r.split == "val"]
    missing = sorted({r.chunk_id for r in val} - set(corpus))
    if missing:
        raise ValueError(f"validation positives not in the pool: {', '.join(missing[:5])}")
    queries = {f"val-{i}": query_prefix + r.query for i, r in enumerate(val)}
    relevant = {f"val-{i}": {r.chunk_id} for i, r in enumerate(val)}
    return queries, corpus, relevant


def history_metrics(history: Sequence[Mapping[str, Any]]) -> list[tuple[str, float, int]]:
    """(name, value, step) for every number in the trainer's log, for MLflow's curves.

    MLflow metric names can't hold "@", so "mrr@10" becomes "mrr_at_10".
    """
    metrics = []
    for entry in history:
        if "step" not in entry:
            continue
        for key, value in entry.items():
            if key in ("step", "epoch") or isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                metrics.append((key.replace("@", "_at_"), float(value), int(entry["step"])))
    return metrics


def fit(
    config: TrainConfig,
    rows: Sequence[MinedPair],
    pool: Sequence[Chunk],
    out_dir: Path,
    max_steps: int = -1,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Train, keep the best epoch by validation MRR@10, save it to `out_dir / "model"`.

    Returns the validation metrics of the base model (`base_*`) and of the saved one, and
    the trainer's log history (loss every 10 steps, validation metrics every epoch).
    Runs on a GPU when torch sees one, else on the CPU.
    """
    import torch
    from datasets import Dataset
    from sentence_transformers.base.sampler import BatchSamplers
    from sentence_transformers.sentence_transformer import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )

    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length
    dataset = Dataset.from_dict(train_columns(rows, config.hard_negatives, config.query_prefix))
    queries, corpus, relevant = ir_eval_data(rows, pool, config.query_prefix)
    # dict[str, Any]: the evaluator's value type also allows images and audio (dict is invariant).
    query_inputs: dict[str, Any] = dict(queries)
    corpus_inputs: dict[str, Any] = dict(corpus)
    evaluator = InformationRetrievalEvaluator(
        query_inputs, corpus_inputs, relevant, name="val", batch_size=64, write_csv=False
    )
    base = {f"base_{k}": float(v) for k, v in evaluator(model).items()}

    full_run = max_steps < 0
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=max_steps,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_ratio,  # a float below 1 is a ratio in Transformers v5
        seed=config.seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch" if full_run else "no",
        save_strategy="epoch" if full_run else "no",
        save_total_limit=1,
        load_best_model_at_end=full_run,
        metric_for_best_model="eval_val_cosine_mrr@10",
        logging_steps=10,
        report_to="none",  # logged to MLflow afterwards (`log_run`), also after a remote run
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=config.mini_batch_size),
        evaluator=evaluator,
    )
    trainer.train()
    final = {k: float(v) for k, v in evaluator(model).items()}
    model.save_pretrained(str(out_dir / "model"))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)  # the best one is saved above
    return {**base, **final}, list(trainer.state.log_history)


def log_run(run_dir: Path, tracking_uri: str, artifact_root: Path, register: bool = True) -> str:
    """Record a finished run folder in MLflow, optionally register its model; returns the id.

    The folder holds `run.json` (name, config, data stats, metrics, log history) and
    `model/`, so a run trained on a rented GPU is logged the same way as a local one.
    """
    from rig_assist.evaluation.tracking import MlflowTracker

    MlflowTracker(tracking_uri, "embedder-training", artifact_root)  # experiment + backend
    import mlflow

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=record["name"]) as run:
        mlflow.log_params({**record["config"], **record["data"]})
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in record["metrics"].items()})
        mlflow.log_metric("train_seconds", record["train_seconds"])
        for name, value, step in history_metrics(record["history"]):
            mlflow.log_metric(name, value, step=step)
        mlflow.log_dict(record, "run.json")
        mlflow.log_artifacts(str(run_dir / "model"), "model")
    if not register:
        print(f"MLflow run {run.info.run_id} (not registered)")
        return str(run.info.run_id)
    # The files are registered directly: `mlflow.register_model("runs:/...")` in MLflow 3
    # wants a LoggedModel (a model flavor), and plain files are what fastembed will load.
    client = mlflow.MlflowClient()
    if not client.search_registered_models(filter_string=f"name = '{REGISTERED_NAME}'"):
        client.create_registered_model(REGISTERED_NAME)
    version = client.create_model_version(
        REGISTERED_NAME, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )
    print(f"MLflow run {run.info.run_id}, registered {REGISTERED_NAME} v{version.version}")
    return str(run.info.run_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the embedder on the mined pairs.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--mined-dir", type=Path, default=Path("data/training/mined"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: data/models/...")
    parser.add_argument("--name", default=None, help="run name (default: a timestamp)")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--mini-batch-size", type=int, default=TrainConfig.mini_batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="for a timing run")
    parser.add_argument("--no-tracking", action="store_true", help="train only (e.g. on a pod)")
    parser.add_argument("--log-run", type=Path, default=None, help="only log a run folder")
    parser.add_argument("--register", action="store_true", help="with --log-run: register it")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.log_run is not None:
        log_run(
            args.log_run,
            settings.mlflow_tracking_uri,
            settings.mlflow_artifact_dir,
            register=args.register,
        )
        return 0

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hard_negatives=not args.no_hard_negatives,
        query_prefix=settings.embedding_query_prefix,
    )
    if not (args.mined_dir / "train.jsonl").exists():
        print(f"No mined pairs in {args.mined_dir}: run `make negatives` first.", file=sys.stderr)
        return 1
    rows = load_mined(args.mined_dir)
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = trainable_chunks(IndexStore(settings.data_index_dir).chunks(), eval_pages(questions))
    name = args.name or time.strftime("bge-small-ft-%Y%m%d-%H%M%S")
    out_dir = args.out_dir or settings.models_dir / "finetuned" / name
    n_rows = len(train_columns(rows, config.hard_negatives)["anchor"])
    print(f"{n_rows} training rows, {len(pool)} chunks in the validation corpus -> {out_dir}")

    started = time.perf_counter()
    metrics, history = fit(config, rows, pool, out_dir, max_steps=args.max_steps)
    seconds = time.perf_counter() - started
    record = {
        "name": name,
        "config": config.params(),
        "data": {
            "train_rows": str(n_rows),
            "val_queries": str(sum(r.split == "val" for r in rows)),
            "val_corpus": str(len(pool)),
        },
        "mined_stats": json.loads((args.mined_dir / "stats.json").read_text()),
        "metrics": metrics,
        "train_seconds": round(seconds, 1),
        "history": history,
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**metrics, "train_seconds": record["train_seconds"]}, indent=2))

    if not args.no_tracking and args.max_steps < 0:
        log_run(out_dir, settings.mlflow_tracking_uri, settings.mlflow_artifact_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
