"""API contract tests for /documents and /search (FastAPI TestClient, temp folders, no OCR)."""

from pathlib import Path
from typing import Any

import pytest
from conftest import build_pdf
from fastapi.testclient import TestClient
from httpx2 import Response

from rig_assist.api.main import create_app
from rig_assist.config import Settings

PUMP_MANUAL = build_pdf(
    [
        ["3.1 Mud Pumps", "Check the pump liners and valves every shift."],
        ["5.1 Blowout Preventer", "Close the annular preventer when a kick is detected."],
    ]
)
SCANNED = build_pdf([["Riser tension depends on water depth."], []])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        data_index_dir=tmp_path / "index",
        data_parsed_dir=tmp_path / "parsed",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def upload(
    client: TestClient, name: str, content: bytes, content_type: str = "application/pdf"
) -> Response:
    return client.post("/documents", files={"file": (name, content, content_type)})


def search(client: TestClient, **body: Any) -> Response:
    return client.post("/search", json=body)


# --- /documents ---------------------------------------------------------


def test_no_documents_at_first(client: TestClient) -> None:
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": []}


def test_upload_ingests_the_pdf_and_returns_201(client: TestClient, settings: Settings) -> None:
    response = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)

    assert response.status_code == 201
    body = response.json()
    assert body["doc_id"] == "drilling-manual"
    assert body["title"] == "Drilling Manual"
    assert body["filename"] == "Drilling_Manual.pdf"
    assert body["pages"] == 2
    assert body["chunks"] >= 2
    assert body["missing_pages"] == []
    assert (settings.data_raw_dir / "Drilling_Manual.pdf").read_bytes() == PUMP_MANUAL


def test_uploaded_documents_are_listed(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    docs = client.get("/documents").json()["documents"]
    assert [d["doc_id"] for d in docs] == ["drilling-manual"]


def test_uploading_the_same_file_again_replaces_it(client: TestClient) -> None:
    first = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL).json()
    second = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)

    assert second.status_code == 201
    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["chunks"] == first["chunks"]


def test_upload_reports_pages_without_text(client: TestClient) -> None:
    body = upload(client, "Riser_Notes.pdf", SCANNED).json()
    assert body["missing_pages"] == [2]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("notes.txt", b"plain text"),
        ("fake.pdf", b"this is not a pdf at all"),  # right name, wrong content
    ],
)
def test_upload_rejects_files_that_are_not_pdfs(
    client: TestClient, name: str, content: bytes
) -> None:
    response = upload(client, name, content, "text/plain")
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]
    assert client.get("/documents").json() == {"documents": []}


def test_upload_without_a_file_is_a_validation_error(client: TestClient) -> None:
    assert client.post("/documents").status_code == 422


def test_upload_cannot_write_outside_the_raw_folder(client: TestClient, settings: Settings) -> None:
    response = upload(client, "../../evil.pdf", PUMP_MANUAL)

    assert response.status_code == 201
    assert response.json()["doc_id"] == "evil"
    assert (settings.data_raw_dir / "evil.pdf").exists()
    assert not (settings.data_raw_dir.parent.parent / "evil.pdf").exists()


def test_upload_rejects_a_different_file_with_the_same_doc_id(client: TestClient) -> None:
    upload(client, "Drilling Manual.pdf", PUMP_MANUAL)
    response = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)

    assert response.status_code == 409
    assert "drilling-manual" in response.json()["detail"]


# --- /search ------------------------------------------------------------


def test_search_returns_ranked_chunks_with_page_numbers(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)

    response = search(client, query="annular preventer kick", top_k=5, mode="bm25")

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "annular preventer kick"
    assert body["mode"] == "bm25"
    top = body["results"][0]
    assert (top["doc_id"], top["page"]) == ("drilling-manual", 2)
    assert top["doc_title"] == "Drilling Manual"
    assert top["section"] == "5.1 Blowout Preventer"
    assert top["parser"] == "pdfplumber"
    assert "annular preventer" in top["text"]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_defaults_to_bm25_and_five_results(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    body = search(client, query="pump").json()
    assert body["mode"] == "bm25"
    assert len(body["results"]) <= 5


def test_search_respects_top_k(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert len(search(client, query="the pump preventer", top_k=1).json()["results"]) == 1


def test_search_with_no_match_returns_an_empty_list(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    response = search(client, query="helicopter")
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_on_an_empty_index_says_to_upload_first(client: TestClient) -> None:
    response = search(client, query="pump")
    assert response.status_code == 409
    assert "POST /documents" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "pump", "mode": "semantic"},  # unknown mode (dense etc. arrive in Phase 2)
        {"query": ""},  # empty query
        {"query": "pump", "top_k": 0},
        {"query": "pump", "top_k": 51},
        {"top_k": 5},  # no query
    ],
)
def test_search_validates_the_request(client: TestClient, body: dict[str, Any]) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert search(client, **body).status_code == 422


def test_search_sees_documents_uploaded_after_the_app_started(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert search(client, query="riser tension").json()["results"] == []
    upload(client, "Riser_Notes.pdf", SCANNED)
    assert search(client, query="riser tension").json()["results"][0]["doc_id"] == "riser-notes"


def test_app_loads_an_existing_index_at_startup(settings: Settings) -> None:
    upload(TestClient(create_app(settings)), "Drilling_Manual.pdf", PUMP_MANUAL)
    restarted = TestClient(create_app(settings))
    assert search(restarted, query="liners").json()["results"][0]["page"] == 1


# --- Swagger ------------------------------------------------------------


def test_swagger_documents_the_new_endpoints_with_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert {"/documents", "/search"} <= set(schema["paths"])
    search_request = schema["components"]["schemas"]["SearchRequest"]
    assert search_request.get("examples") or any(
        "examples" in prop for prop in search_request["properties"].values()
    )
