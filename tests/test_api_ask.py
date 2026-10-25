"""API contract tests for POST /ask.

Same setup as the /search tests (temp folders, in-memory Qdrant, fake embedder and
cross-encoder), plus a scripted FakeLlm: offline, free, deterministic.
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
from rig_assist.llm import FAKE_REPLY, FakeLlm
from rig_assist.rag import NOT_FOUND
from rig_assist.retrieval.dense import DenseIndex

PUMP_MANUAL = build_pdf(
    [
        ["3.1 Mud Pumps", "Check the pump liners and valves every shift."],
        ["5.1 Blowout Preventer", "Close the annular preventer when a kick is detected."],
    ]
)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_raw_dir=tmp_path / "raw",
        data_index_dir=tmp_path / "index",
        data_parsed_dir=tmp_path / "parsed",
        **overrides,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())


def make_client(settings: Settings, dense: DenseIndex, llm: FakeLlm | None = None) -> TestClient:
    app = create_app(settings, dense=dense, cross_encoder=FakeCrossEncoder(), llm=llm)
    return TestClient(app)


def with_manual(settings: Settings, dense: DenseIndex, llm: FakeLlm | None = None) -> TestClient:
    client = make_client(settings, dense, llm)
    response = client.post(
        "/documents", files={"file": ("Drilling_Manual.pdf", PUMP_MANUAL, "application/pdf")}
    )
    assert response.status_code == 201
    return client


def ask(client: TestClient, **body: Any) -> Response:
    return client.post("/ask", json=body)


# --- Answers ------------------------------------------------------------


def test_ask_returns_the_answer_with_checked_citations(
    settings: Settings, dense: DenseIndex
) -> None:
    answer = "Close the annular preventer [drilling-manual p.2]."
    llm = FakeLlm(replies=[answer])
    client = with_manual(settings, dense, llm)

    response = ask(client, question="What do I do when a kick is detected?")

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What do I do when a kick is detected?"
    assert body["answer"] == answer
    assert body["found"] is True
    assert body["refusal_reason"] is None
    assert body["citations"] == [
        {
            "doc_id": "drilling-manual",
            "doc_title": "Drilling Manual",
            "page": 2,
            "snippet": "5.1 Blowout Preventer Close the annular preventer when a kick is detected.",
        }
    ]
    assert len(llm.calls) == 1


def test_ask_uses_hybrid_rerank_and_five_chunks_by_default(
    settings: Settings, dense: DenseIndex
) -> None:
    llm = FakeLlm(replies=["Yes [drilling-manual p.1]."])
    body = ask(with_manual(settings, dense, llm), question="pump liners").json()
    assert body["mode"] == "hybrid_rerank"
    assert body["top_k"] == 5


def test_ask_passes_mode_and_top_k_to_retrieval(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=["Check them every shift [drilling-manual p.1]."])
    client = with_manual(settings, dense, llm)
    body = ask(client, question="pump liners", mode="bm25", top_k=1).json()
    assert (body["mode"], body["top_k"], body["found"]) == ("bm25", 1, True)
    prompt = llm.calls[0][1].content
    assert "[drilling-manual p.1]" in prompt
    assert "[drilling-manual p.2]" not in prompt


def test_the_default_fake_llm_gives_a_cited_answer(settings: Settings, dense: DenseIndex) -> None:
    # LLM_PROVIDER=fake (the default) must give a working demo offline.
    client = with_manual(settings, dense)  # no llm injected: built from settings
    body = ask(client, question="pump liners", mode="bm25").json()
    assert body["found"] is True
    assert body["answer"].startswith(FAKE_REPLY)
    assert body["usage"]["model"] == "fake"


# --- "Not found" --------------------------------------------------------


def test_an_invented_citation_is_refused(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=["Every 14 days [drilling-manual p.9]."])
    body = ask(with_manual(settings, dense, llm), question="pump liners").json()
    assert body["found"] is False
    assert body["answer"] == NOT_FOUND
    assert body["refusal_reason"] == "invalid_citation"
    assert body["citations"] == []


def test_the_llm_saying_not_found_is_passed_on(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=[NOT_FOUND])
    body = ask(with_manual(settings, dense, llm), question="helicopter refuelling").json()
    assert (body["found"], body["refusal_reason"]) == (False, "llm_not_found")


def test_no_retrieval_results_is_not_found_without_an_llm_call(
    settings: Settings, dense: DenseIndex
) -> None:
    llm = FakeLlm(replies=[])
    body = ask(with_manual(settings, dense, llm), question="helicopter", mode="bm25").json()
    assert (body["found"], body["refusal_reason"]) == (False, "no_results")
    assert body["answer"] == NOT_FOUND
    assert llm.calls == []
    assert body["usage"]["llm_calls"] == 0


# --- Usage, cost, latency, disclaimer -----------------------------------


def test_usage_and_latency_are_reported(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=["Every shift [drilling-manual p.1]."])
    body = ask(with_manual(settings, dense, llm), question="pump liners").json()
    usage = body["usage"]
    assert usage["llm_calls"] == 1
    assert usage["model"] == "fake"
    assert usage["output_tokens"] == 4  # FakeLlm counts words
    assert usage["input_tokens"] == sum(len(m.content.split()) for m in llm.calls[0])
    assert usage["cost_usd"] is None  # no prices configured
    latency = body["latency"]
    assert latency["total_s"] >= latency["retrieval_s"] >= 0
    assert latency["llm_s"] >= 0


def test_cost_is_estimated_when_prices_are_set(tmp_path: Path, dense: DenseIndex) -> None:
    settings = make_settings(tmp_path, llm_input_price_per_1m=1.0, llm_output_price_per_1m=2.0)
    llm = FakeLlm(replies=["Every shift [drilling-manual p.1]."])
    usage = ask(with_manual(settings, dense, llm), question="pump liners").json()["usage"]
    expected = (usage["input_tokens"] * 1.0 + usage["output_tokens"] * 2.0) / 1e6
    assert usage["cost_usd"] == pytest.approx(expected)


def test_every_answer_carries_the_safety_disclaimer(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=["Every shift [drilling-manual p.1]."])
    body = ask(with_manual(settings, dense, llm), question="pump liners").json()
    assert "not a safety-certified system" in body["disclaimer"].lower()


# --- Errors -------------------------------------------------------------


def test_llm_failure_is_502_with_the_reason(settings: Settings, dense: DenseIndex) -> None:
    llm = FakeLlm(replies=[])  # raises LlmError on the first call
    response = ask(with_manual(settings, dense, llm), question="pump liners")
    assert response.status_code == 502
    assert "LLM" in response.json()["detail"]


def test_ask_on_an_empty_index_says_to_upload_first(settings: Settings, dense: DenseIndex) -> None:
    response = ask(make_client(settings, dense, FakeLlm(replies=[])), question="pump")
    assert response.status_code == 409
    assert "POST /documents" in response.json()["detail"]


def test_ask_with_qdrant_down_is_503_and_bm25_still_works(
    settings: Settings, dense: DenseIndex
) -> None:
    with_manual(settings, dense)
    down = QdrantClient(url="http://127.0.0.1:1", check_compatibility=False, timeout=2)
    client = make_client(settings, DenseIndex(down, "test_chunks", KeywordEmbedder()))
    response = ask(client, question="pump liners")
    assert response.status_code == 503
    assert "docker compose up -d qdrant" in response.json()["detail"]
    assert ask(client, question="pump liners", mode="bm25").status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},  # no question
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "pump", "mode": "semantic"},
        {"question": "pump", "top_k": 0},
        {"question": "pump", "top_k": 21},
    ],
)
def test_ask_validates_the_request(
    settings: Settings, dense: DenseIndex, body: dict[str, Any]
) -> None:
    assert ask(with_manual(settings, dense), **body).status_code == 422


def test_swagger_documents_ask_with_an_example(settings: Settings, dense: DenseIndex) -> None:
    schema = make_client(settings, dense).get("/openapi.json").json()
    assert "/ask" in schema["paths"]
    assert schema["components"]["schemas"]["AskRequest"].get("examples")
