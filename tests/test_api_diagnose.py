"""API contract tests for POST /diagnose and GET /jobs/{job_id}.

The real graph runs with a scripted FakeLlm against an indexed test manual (in-memory
Qdrant, fake embedder and cross-encoder). TestClient runs background tasks before the
POST returns, so the job is finished when the test reads it.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeCrossEncoder, KeywordEmbedder, build_pdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rig_assist.api.main import create_app
from rig_assist.config import Settings
from rig_assist.jobs import JobStore
from rig_assist.llm import FakeLlm
from rig_assist.retrieval.dense import DenseIndex

PUMP_MANUAL = build_pdf(
    [
        ["3.1 Mud Pumps", "Worn liners cause pressure loss. Check the liners every shift."],
        ["3.2 Pump safety", "Never open the fluid end while the pump is pressurised."],
    ]
)
DESCRIPTION = "Mud pump 2 losing pressure. Should I open the fluid end while it runs?"


def settings_for(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        data_index_dir=tmp_path / "index",
        data_parsed_dir=tmp_path / "parsed",
        job_db_path=tmp_path / "jobs.sqlite",
    )


def client_for(tmp_path: Path, replies: list[str], index: bool = True) -> TestClient:
    dense = DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())
    app = create_app(
        settings_for(tmp_path), dense=dense, cross_encoder=FakeCrossEncoder(), llm=FakeLlm(replies)
    )
    client = TestClient(app)
    if index:
        response = client.post(
            "/documents", files={"file": ("Pump_Manual.pdf", PUMP_MANUAL, "application/pdf")}
        )
        assert response.status_code == 201
    return client


def triage(open_questions: list[str] | None = None) -> str:
    return json.dumps(
        {
            "equipment": "Mud pump",
            "equipment_id": "MP-2",
            "symptoms": ["pressure loss"],
            "operating_context": None,
            "open_questions": open_questions or [],
        }
    )


PLAN_DONE = json.dumps({"queries": [], "done": True})
DIAGNOSIS = json.dumps(
    {
        "causes": [
            {
                "cause": "Worn liners",
                "likelihood": "high",
                "evidence": [{"doc_id": "pump-manual", "page": 1}],
            }
        ],
        "recommended_actions": ["Check the liners", "Open the fluid end while running"],
    }
)
SAFETY = json.dumps(
    {
        "warnings": ["Stop and bleed down the pump before opening the fluid end"],
        "blocked_action_numbers": [2],
        "other_blocked_actions": [],
        "hazard": False,
        "evidence": [{"doc_id": "pump-manual", "page": 2}],
    }
)
FULL = [triage(), PLAN_DONE, DIAGNOSIS, SAFETY]


def diagnose(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/diagnose", json=body)
    assert response.status_code == 202, response.text
    created: dict[str, Any] = response.json()
    return created


def test_diagnose_returns_202_with_a_job_id_and_a_status_url(tmp_path: Path) -> None:
    created = diagnose(client_for(tmp_path, FULL), description=DESCRIPTION)
    assert created["status"] == "queued"
    assert created["status_url"] == f"/jobs/{created['job_id']}"


def test_the_finished_job_returns_a_valid_work_order(tmp_path: Path) -> None:
    client = client_for(tmp_path, FULL)
    created = diagnose(client, description=DESCRIPTION, equipment_id="MP-2")
    job = client.get(created["status_url"]).json()
    assert job["status"] == "done"
    order = job["result"]
    assert order["equipment"] == "Mud pump (MP-2)"
    assert order["recommended_actions"] == ["Check the liners"]
    assert order["blocked_actions"] == ["Open the fluid end while running"]
    assert [(c["doc_id"], c["page"]) for c in order["citations"]] == [
        ("pump-manual", 1),
        ("pump-manual", 2),
    ]
    assert job["error"] is None and job["trace_url"] is None
    assert job["usage"]["llm_calls"] == 4 and job["usage"]["output_tokens"] > 0
    assert [s["node"] for s in job["steps"]] == [
        "triage",
        "retrieval",
        "diagnosis",
        "safety",
        "report",
    ]
    assert "disclaimer" in job


def test_missing_information_finishes_with_open_questions(tmp_path: Path) -> None:
    client = client_for(tmp_path, [triage(["Which pump: MP-1 or MP-2?"])])
    job = client.get(diagnose(client, description="pump noise")["status_url"]).json()
    assert job["status"] == "done"
    assert job["result"]["open_questions"] == ["Which pump: MP-1 or MP-2?"]
    assert [s["node"] for s in job["steps"]] == ["triage", "report"]


def test_a_failing_node_fails_the_job_with_the_error(tmp_path: Path) -> None:
    client = client_for(tmp_path, [triage(), PLAN_DONE, "not json", "still not json"])
    job = client.get(diagnose(client, description=DESCRIPTION)["status_url"]).json()
    assert job["status"] == "failed"
    assert job["result"] is None
    assert "Diagnosis" in job["error"]


def test_an_unknown_job_is_404(tmp_path: Path) -> None:
    response = client_for(tmp_path, [], index=False).get("/jobs/does-not-exist")
    assert response.status_code == 404


def test_nothing_indexed_is_409_and_creates_no_job(tmp_path: Path) -> None:
    client = client_for(tmp_path, FULL, index=False)
    response = client.post("/diagnose", json={"description": DESCRIPTION})
    assert response.status_code == 409
    assert JobStore(tmp_path / "jobs.sqlite").all() == []


@pytest.mark.parametrize(
    "body",
    [{}, {"description": ""}, {"description": "x" * 4001}, {"description": "ok", "extra": 1}],
)
def test_invalid_requests_are_422(tmp_path: Path, body: dict[str, Any]) -> None:
    response = client_for(tmp_path, [], index=False).post("/diagnose", json=body)
    assert response.status_code == 422


def test_jobs_left_running_by_a_restart_are_failed_at_startup(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create({"description": "x", "equipment_id": None})
    store.start(job.id)
    dense = DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())
    app = create_app(settings_for(tmp_path), dense=dense, cross_encoder=FakeCrossEncoder())
    with TestClient(app) as client:  # runs the startup hook, as a restarted server would
        response = client.get(f"/jobs/{job.id}").json()
    assert response["status"] == "failed"
    assert "restart" in response["error"]
