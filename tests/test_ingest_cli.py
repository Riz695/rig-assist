from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from rig_assist.config import Settings, get_settings
from rig_assist.ingestion.__main__ import main
from rig_assist.ingestion.store import IndexStore


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

    assert main([]) == 0

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
    assert main([]) == 1
    assert "Ingestion failed" in capsys.readouterr().err
