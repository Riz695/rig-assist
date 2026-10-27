"""API contract tests for /documents and /search.

FastAPI TestClient, temp folders, no OCR; Qdrant in-memory with a fake embedder and a
fake cross-encoder, so no server and no model downloads.
"""

from pathlib import Path
from typing import Any

import pytest
from conftest import FakeCrossEncoder, KeywordEmbedder, build_pdf
from fastapi.testclient import TestClient
from httpx2 import Response
from qdrant_client import QdrantClient

from rig_assist.api.main import create_app
from rig_assist.config import Settings
from rig_assist.retrieval.dense import DenseIndex

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
def dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())


def make_client(settings: Settings, dense: DenseIndex) -> TestClient:
    return TestClient(create_app(settings, dense=dense, cross_encoder=FakeCrossEncoder()))


def unreachable_qdrant() -> DenseIndex:
    """A dense index whose server refuses connections (nothing listens on port 1)."""
    client = QdrantClient(url="http://127.0.0.1:1", check_compatibility=False, timeout=2)
    return DenseIndex(client, "test_chunks", KeywordEmbedder())


@pytest.fixture
def client(settings: Settings, dense: DenseIndex) -> TestClient:
    return make_client(settings, dense)


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


def test_upload_writes_the_chunks_into_the_dense_index_too(
    client: TestClient, dense: DenseIndex
) -> None:
    body = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL).json()
    assert dense.count_document("drilling-manual") == body["chunks"]


def test_uploading_the_same_file_again_replaces_it(client: TestClient, dense: DenseIndex) -> None:
    first = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL).json()
    second = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)

    assert second.status_code == 201
    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["chunks"] == first["chunks"]
    assert dense.count() == first["chunks"]  # no duplicates in Qdrant either


def test_upload_with_qdrant_down_is_503_and_the_document_is_not_listed(
    settings: Settings,
) -> None:
    client = make_client(settings, unreachable_qdrant())
    response = upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert response.status_code == 503
    assert "docker compose up -d qdrant" in response.json()["detail"]
    assert client.get("/documents").json() == {"documents": []}


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


def test_search_defaults_to_hybrid_rerank_and_five_results(client: TestClient) -> None:
    # The most accurate mode on the eval set (eval/results/retrieval.md); the others are
    # one parameter away.
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    body = search(client, query="pump").json()
    assert body["mode"] == "hybrid_rerank"
    assert len(body["results"]) <= 5


def test_search_respects_top_k(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert len(search(client, query="the pump preventer", top_k=1).json()["results"]) == 1


def test_keyword_search_with_no_match_returns_an_empty_list(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    response = search(client, query="helicopter", mode="bm25")
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_default_search_returns_the_nearest_chunks_even_without_a_keyword_match(
    client: TestClient,
) -> None:
    # Dense retrieval always has nearest neighbours, so /search never says "nothing found";
    # deciding that the manuals don't answer a question is /ask's job (Phase 4).
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert search(client, query="helicopter").json()["results"] != []


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_search_on_an_empty_index_says_to_upload_first(client: TestClient, mode: str) -> None:
    response = search(client, query="pump", mode=mode)
    assert response.status_code == 409
    assert "POST /documents" in response.json()["detail"]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_relevant_page(client: TestClient, mode: str) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    response = search(client, query="mud pump liners", mode=mode, top_k=2)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert body["results"][0]["page"] == 1
    assert len(body["results"]) <= 2


def test_dense_mode_matches_meaning_where_bm25_finds_nothing(client: TestClient) -> None:
    # The manual says "kick", the query "kicks". BM25 matches exact tokens (no stemming),
    # so it finds nothing; the fake embedder, like a real one, puts both near "kick".
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    assert search(client, query="kicks", mode="bm25").json()["results"] == []
    dense = search(client, query="kicks", mode="dense").json()["results"]
    assert dense[0]["page"] == 2


def test_dense_score_is_reported_for_modes_that_use_dense_retrieval(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    bm25 = search(client, query="mud pump", mode="bm25").json()["results"][0]
    hybrid = search(client, query="mud pump", mode="hybrid").json()["results"][0]
    assert bm25["dense_score"] is None
    assert 0 < hybrid["dense_score"] <= 1.0 + 1e-6
    assert hybrid["score"] != hybrid["dense_score"]  # the RRF score


def test_hybrid_rerank_scores_with_the_cross_encoder(client: TestClient) -> None:
    upload(client, "Drilling_Manual.pdf", PUMP_MANUAL)
    top = search(client, query="close the annular preventer", mode="hybrid_rerank").json()
    # FakeCrossEncoder: the score is the number of query words found in the chunk.
    assert top["results"][0]["page"] == 2
    assert top["results"][0]["score"] == 4.0


@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_rerank"])
def test_dense_modes_with_qdrant_down_are_503_and_bm25_still_works(
    settings: Settings, dense: DenseIndex, mode: str
) -> None:
    upload(make_client(settings, dense), "Drilling_Manual.pdf", PUMP_MANUAL)
    client = make_client(settings, unreachable_qdrant())
    response = search(client, query="pump", mode=mode)
    assert response.status_code == 503
    assert "docker compose up -d qdrant" in response.json()["detail"]
    assert search(client, query="pump", mode="bm25").status_code == 200


@pytest.mark.parametrize("mode", ["dense", "hybrid", "hybrid_rerank"])
def test_dense_modes_say_so_when_only_bm25_was_built(settings: Settings, mode: str) -> None:
    # e.g. `make ingest --skip-dense`: the store has documents, Qdrant has none.
    upload(
        make_client(settings, DenseIndex(QdrantClient(":memory:"), "a", KeywordEmbedder())),
        "Drilling_Manual.pdf",
        PUMP_MANUAL,
    )
    empty_dense = DenseIndex(QdrantClient(":memory:"), "b", KeywordEmbedder())
    response = search(make_client(settings, empty_dense), query="pump", mode=mode)
    assert response.status_code == 409
    assert "make ingest" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "pump", "mode": "semantic"},  # unknown mode
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
    assert search(client, query="riser tension", mode="bm25").json()["results"] == []
    upload(client, "Riser_Notes.pdf", SCANNED)
    top = search(client, query="riser tension", mode="bm25").json()["results"][0]
    assert top["doc_id"] == "riser-notes"


def test_app_loads_an_existing_index_at_startup(settings: Settings, dense: DenseIndex) -> None:
    upload(make_client(settings, dense), "Drilling_Manual.pdf", PUMP_MANUAL)
    restarted = make_client(settings, dense)
    assert search(restarted, query="liners").json()["results"][0]["page"] == 1


# --- Swagger ------------------------------------------------------------


def test_swagger_documents_the_new_endpoints_with_examples(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert {"/documents", "/search"} <= set(schema["paths"])
    search_request = schema["components"]["schemas"]["SearchRequest"]
    assert search_request.get("examples") or any(
        "examples" in prop for prop in search_request["properties"].values()
    )


# --- GET /documents/{doc_id}/pages/{page} -------------------------------------


def page_image(client: TestClient, doc_id: str, page: int) -> Response:
    return client.get(f"/documents/{doc_id}/pages/{page}")


def test_a_page_comes_back_as_a_png_image(client: TestClient) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    response = page_image(client, "pump-manual", 1)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert "max-age" in response.headers["cache-control"]


def test_each_page_has_its_own_image(client: TestClient) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    assert (
        page_image(client, "pump-manual", 1).content != page_image(client, "pump-manual", 2).content
    )


def test_a_reuploaded_manual_shows_its_new_pages(client: TestClient) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    before = page_image(client, "pump-manual", 1).content
    upload(client, "Pump_Manual.pdf", build_pdf([["Revised 3.1", "Replace liners weekly."]]))
    assert page_image(client, "pump-manual", 1).content != before


def test_an_unknown_document_is_404(client: TestClient) -> None:
    response = page_image(client, "no-such-manual", 1)
    assert response.status_code == 404
    assert "no-such-manual" in response.json()["detail"]


def test_a_page_past_the_end_is_404(client: TestClient) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    response = page_image(client, "pump-manual", 3)
    assert response.status_code == 404
    assert "2 pages" in response.json()["detail"]


@pytest.mark.parametrize("page", ["0", "-1", "two"])
def test_an_invalid_page_number_is_422(client: TestClient, page: str) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    assert client.get(f"/documents/pump-manual/pages/{page}").status_code == 422


def test_a_pdf_removed_from_disk_is_404(client: TestClient, settings: Settings) -> None:
    upload(client, "Pump_Manual.pdf", PUMP_MANUAL)
    (settings.data_raw_dir / "Pump_Manual.pdf").unlink()
    response = page_image(client, "pump-manual", 1)
    assert response.status_code == 404
    assert "PDF" in response.json()["detail"]
