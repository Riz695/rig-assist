"""MlflowTracker against a real (temporary) SQLite MLflow store.

MLflow is in the optional `eval` dependency group, so this is skipped where it isn't
installed (CI installs only the default groups). Run it with `uv run --group eval pytest`.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_mlflow_tracker_logs_params_metrics_and_artifacts(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from rig_assist.evaluation.tracking import MlflowTracker

    uri = f"sqlite:///{tmp_path}/db/mlflow.db"
    tracker = MlflowTracker(uri, "test-experiment", tmp_path / "artifacts")

    run_id = tracker.log_run(
        "bm25", {"mode": "bm25"}, {"all/hit_at_5": 0.75}, {"per_question.json": "[]"}
    )

    assert run_id is not None
    run = mlflow.MlflowClient(tracking_uri=uri).get_run(run_id)
    assert run.info.run_name == "bm25"
    assert run.data.params == {"mode": "bm25"}
    assert run.data.metrics == {"all/hit_at_5": 0.75}
    assert (tmp_path / "artifacts" / run_id / "artifacts" / "per_question.json").read_text() == "[]"
