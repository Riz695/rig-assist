from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from conftest import KeywordEmbedder
from qdrant_client import QdrantClient

import rig_assist.ingestion.__main__ as cli
from rig_assist.config import Settings, get_settings
from rig_assist.ingestion.__main__ import main
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.dense import DenseIndex


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run in an empty folder (no .env) with a clean environment and fresh settings."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_main_ingests_the_raw_folder_and_prints_a_summary(
    workdir: Path, make_pdf: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_pdf([["Mud pump liner wear is checked weekly"], []], name="Drilling_Manual.pdf")
    pdf.rename(workdir / "data" / "raw" / pdf.name)

    assert main(["--skip-dense"]) == 0

    out = capsys.readouterr().out
    assert "drilling-manual" in out
    assert "not searchable" in out  # page 2 has no text and OCR is off
    assert [d.doc_id for d in IndexStore(workdir / "data" / "index").documents()] == [
        "drilling-manual"
    ]


def test_main_fails_cleanly_when_the_raw_folder_is_missing(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workdir / "data" / "raw").rmdir()
    assert main(["--skip-dense"]) == 1
    assert "Ingestion failed" in capsys.readouterr().err


def test_main_fills_the_dense_index_too(
    workdir: Path, make_pdf: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf([["Mud pump liner wear is checked weekly"]], name="Drilling_Manual.pdf")
    pdf.rename(workdir / "data" / "raw" / pdf.name)
    dense = DenseIndex(QdrantClient(":memory:"), "cli", KeywordEmbedder())
    monkeypatch.setattr(cli, "make_dense_index", lambda settings: dense)

    assert main([]) == 0
    assert dense.count_document("drilling-manual") == 1


def test_main_explains_how_to_start_qdrant_when_it_is_down(
    workdir: Path,
    make_pdf: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pdf = make_pdf([["Mud pump liner wear is checked weekly"]], name="Drilling_Manual.pdf")
    pdf.rename(workdir / "data" / "raw" / pdf.name)
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")  # nothing listens on port 1
    get_settings.cache_clear()

    assert main([]) == 1
    err = capsys.readouterr().err
    assert "Qdrant" in err and "docker compose up -d qdrant" in err
