from pathlib import Path

import pytest
from conftest import FakeCrossEncoder, KeywordEmbedder, build_pdf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import rig_assist
from rig_assist.api.main import create_app
from rig_assist.config import Settings
from rig_assist.retrieval.dense import DenseIndex


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": rig_assist.__version__}


def test_health_is_documented_in_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Rig Assist API"
    assert schema["info"]["version"] == rig_assist.__version__
    assert "get" in schema["paths"]["/health"]
    assert "HealthResponse" in schema["components"]["schemas"]


def test_swagger_ui_is_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/does-not-exist").status_code == 404


# --- readiness: GET /ready --------------------------------------------------
# The platform routes traffic only to a ready API: index loaded, Qdrant answering with the
# collection. Real pieces: temp folders, an in-memory Qdrant, a real upload.

PUMP_MANUAL = build_pdf(
    [
        ["3.1 Mud Pumps", "Check the pump liners and valves every shift."],
        ["5.1 Blowout Preventer", "Close the annular preventer when a kick is detected."],
    ]
)


def ready_settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        data_index_dir=tmp_path / "index",
        data_parsed_dir=tmp_path / "parsed",
        job_db_path=tmp_path / "jobs.sqlite",
    )


def app_with(settings: Settings, dense: DenseIndex) -> TestClient:
    return TestClient(create_app(settings, dense=dense, cross_encoder=FakeCrossEncoder()))


def indexed(settings: Settings, dense: DenseIndex) -> TestClient:
    client = app_with(settings, dense)
    upload = ("Drilling_Manual.pdf", PUMP_MANUAL, "application/pdf")
    assert client.post("/documents", files={"file": upload}).status_code == 201
    return client


def memory_dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "rig_assist_chunks", KeywordEmbedder())


def test_ready_is_200_with_counts_when_index_and_vectors_exist(tmp_path: Path) -> None:
    response = indexed(ready_settings(tmp_path), memory_dense()).get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["chunks"] > 0 and body["vectors"] == body["chunks"]


def test_ready_is_503_when_nothing_is_indexed(tmp_path: Path) -> None:
    response = app_with(ready_settings(tmp_path), memory_dense()).get("/ready")

    assert response.status_code == 503
    assert "No documents are indexed" in response.json()["detail"]


def test_ready_is_503_when_the_collection_is_missing(tmp_path: Path) -> None:
    dense = memory_dense()
    client = indexed(ready_settings(tmp_path), dense)
    dense.client.delete_collection("rig_assist_chunks")  # e.g. the snapshot restore failed

    response = client.get("/ready")

    assert response.status_code == 503
    assert "rig_assist_chunks" in response.json()["detail"]


def test_ready_is_503_when_qdrant_is_unreachable(tmp_path: Path) -> None:
    settings = ready_settings(tmp_path)
    indexed(settings, memory_dense())  # BM25 index on disk
    nowhere = DenseIndex(
        QdrantClient(url="http://127.0.0.1:9", timeout=1), "rig_assist_chunks", KeywordEmbedder()
    )

    response = app_with(settings, nowhere).get("/ready")

    assert response.status_code == 503
    assert "Qdrant" in response.json()["detail"]


def test_ready_is_documented_in_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "get" in schema["paths"]["/ready"]
    assert "ReadyResponse" in schema["components"]["schemas"]
