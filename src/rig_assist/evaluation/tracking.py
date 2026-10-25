"""Experiment tracking: where each eval run's params, metrics and files are recorded.

`Tracker` is the small interface the eval code needs; `MlflowTracker` implements it.
MLflow is imported inside the class (it lives in the optional `eval` dependency group),
so the rest of the package and the unit tests never need it installed.
"""

import os
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifacts: dict[str, str],
    ) -> str | None:
        """Record one run; returns its id if the tracker has ids."""
        ...


class NullTracker:
    """Records nothing (`--no-tracking`)."""

    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifacts: dict[str, str],
    ) -> str | None:
        return None


class MlflowTracker:
    """One MLflow run per call, in `experiment`, stored in a SQLite backend.

    SQLite has been MLflow's default backend since 3.7; the old `./mlruns` file store is
    legacy. Artifacts (text files here) go to `artifact_root`, a local folder.
    """

    def __init__(self, tracking_uri: str, experiment: str, artifact_root: Path) -> None:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # skip a log line about skills
        import mlflow

        if tracking_uri.startswith("sqlite:///"):
            Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            artifact_root.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(experiment, artifact_location=artifact_root.resolve().as_uri())
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifacts: dict[str, str],
    ) -> str | None:
        mlflow = self._mlflow
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for filename, text in artifacts.items():
                mlflow.log_text(text, filename)
        return str(run.info.run_id)
