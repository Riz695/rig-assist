from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import FakeOcr, KeywordEmbedder, no_sleep
from qdrant_client import QdrantClient

from rig_assist.ingestion.models import Chunk
from rig_assist.ingestion.pipeline import ingest_directory, ingest_file, make_doc_id, make_title
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.dense import DenseIndex

MakePdf = Callable[..., Path]

PUMP_PAGE = ["3.1 Mud Pumps", "Check the liners and valves every shift."]
BOP_PAGE = ["5.1 Blowout Preventer", "Close the annular preventer on a kick."]


@pytest.mark.parametrize(
    ("filename", "doc_id"),
    [
        ("Drilling_Manual.pdf", "drilling-manual"),
        ("Offshore manual.pdf", "offshore-manual"),
        ("cds-guide-feb21.pdf", "cds-guide-feb21"),
        ("Occupational Safety and Health for Oil.pdf", "occupational-safety-and-health-for-oil"),
        (
            "An-Introduction-To-Oilfield-Drilling-Operations.pdf",
            "an-introduction-to-oilfield-drilling-operations",
        ),
        ("  Pump  (rev 2).PDF", "pump-rev-2"),
        ("MarineOffshoreLogOpsManual.pdf", "marine-offshore-log-ops-manual"),
        ("API_RP54Guide.pdf", "api-rp54-guide"),
    ],
)
def test_make_doc_id_turns_file_names_into_slugs(filename: str, doc_id: str) -> None:
    assert make_doc_id(filename) == doc_id


def test_make_doc_id_rejects_names_without_letters_or_digits() -> None:
    with pytest.raises(ValueError):
        make_doc_id("???.pdf")


@pytest.mark.parametrize(
    ("filename", "title"),
    [
        ("Drilling_Manual.pdf", "Drilling Manual"),
        (
            "An-Introduction-To-Oilfield-Drilling-Operations.pdf",
            "An Introduction To Oilfield Drilling Operations",
        ),
        ("Offshore manual.pdf", "Offshore manual"),
        ("MarineOffshoreLogOpsManual.pdf", "Marine Offshore Log Ops Manual"),
    ],
)
def test_make_title_is_the_readable_file_name(filename: str, title: str) -> None:
    assert make_title(filename) == title


def test_ingest_file_stores_chunks_and_reports_missing_pages(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    store = IndexStore(tmp_path / "index")
    path = make_pdf([PUMP_PAGE, [], BOP_PAGE], name="Drilling_Manual.pdf")

    info = ingest_file(path, store)

    assert info.doc_id == "drilling-manual"
    assert info.title == "Drilling Manual"
    assert info.filename == "Drilling_Manual.pdf"
    assert info.pages == 3
    assert info.missing_pages == [2]
    assert info.chunks == len(store.chunks()) > 0
    assert store.documents() == [info]
    chunk = store.chunks()[0]
    assert (chunk.doc_id, chunk.doc_title, chunk.page) == ("drilling-manual", "Drilling Manual", 1)
    assert chunk.section == "3.1 Mud Pumps"


def test_ingesting_the_same_file_twice_does_not_duplicate_chunks(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    store = IndexStore(tmp_path / "index")
    path = make_pdf([PUMP_PAGE, BOP_PAGE])
    ingest_file(path, store)
    first = store.chunks()
    ingest_file(path, store)
    assert store.chunks() == first


def test_ingest_file_with_ocr_fills_scanned_pages(make_pdf: MakePdf, tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index")
    ocr = FakeOcr()

    info = ingest_file(
        make_pdf([PUMP_PAGE, []]), store, ocr_mode="missing", ocr_client=ocr, sleep=no_sleep
    )

    assert ocr.calls == 1
    assert info.missing_pages == []
    assert {(c.page, c.parser) for c in store.chunks()} == {(1, "pdfplumber"), (2, "olmocr")}


def test_chunk_size_is_passed_through(make_pdf: MakePdf, tmp_path: Path) -> None:
    long_page = [f"Line {i}: inspect the drawworks brake band and linkage." for i in range(60)]
    small = ingest_file(
        make_pdf([long_page]), IndexStore(tmp_path / "a"), chunk_size=60, chunk_overlap=0
    )
    large = ingest_file(
        make_pdf([long_page]), IndexStore(tmp_path / "b"), chunk_size=500, chunk_overlap=0
    )
    assert small.chunks > large.chunks


def test_ingest_directory_ingests_every_pdf(make_pdf: MakePdf, tmp_path: Path) -> None:
    make_pdf([PUMP_PAGE], name="Pump Manual.pdf")
    make_pdf([BOP_PAGE], name="BOP_Guide.pdf")
    (tmp_path / "notes.txt").write_text("not a manual")
    store = IndexStore(tmp_path / "index")

    infos = ingest_directory(tmp_path, store)

    assert [i.doc_id for i in infos] == ["bop-guide", "pump-manual"]
    assert [d.doc_id for d in store.documents()] == ["bop-guide", "pump-manual"]


def test_ingest_directory_rejects_two_files_with_the_same_doc_id_before_writing(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    make_pdf([PUMP_PAGE], name="pump manual.pdf")
    make_pdf([BOP_PAGE], name="pump_manual.pdf")
    store = IndexStore(tmp_path / "index")

    with pytest.raises(ValueError, match="pump-manual"):
        ingest_directory(tmp_path, store)
    assert store.documents() == []


def test_ingest_directory_with_no_pdfs_returns_empty(tmp_path: Path) -> None:
    assert ingest_directory(tmp_path, IndexStore(tmp_path / "index")) == []


# --- both indexes: BM25 store + dense (Qdrant) -----------------------------


@pytest.fixture
def dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "test_chunks", KeywordEmbedder())


def test_ingest_file_writes_the_same_chunks_to_both_indexes(
    make_pdf: MakePdf, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "index")
    info = ingest_file(make_pdf([PUMP_PAGE, BOP_PAGE]), store, dense=dense)

    assert dense.count_document(info.doc_id) == info.chunks == len(store.chunks())
    assert dense.search("pump", k=1)[0].chunk == store.chunks()[0]


def test_reingesting_the_same_document_leaves_both_counts_unchanged(
    make_pdf: MakePdf, tmp_path: Path, dense: DenseIndex
) -> None:
    """PLAN.md Phase 2 test: idempotent ingestion across both indexes."""
    store = IndexStore(tmp_path / "index")
    path = make_pdf([PUMP_PAGE, BOP_PAGE])
    ingest_file(path, store, dense=dense)
    first = (len(store.chunks()), dense.count())

    ingest_file(path, store, dense=dense)
    ingest_file(path, store, dense=dense)

    assert (len(store.chunks()), dense.count()) == first


def test_reingesting_a_shorter_version_shrinks_both_indexes(
    make_pdf: MakePdf, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "index")
    ingest_file(make_pdf([PUMP_PAGE, BOP_PAGE], name="Manual.pdf"), store, dense=dense)
    info = ingest_file(make_pdf([PUMP_PAGE], name="Manual.pdf"), store, dense=dense)

    assert info.chunks == len(store.chunks()) == dense.count() == 1


def test_ingest_directory_fills_both_indexes(
    make_pdf: MakePdf, tmp_path: Path, dense: DenseIndex
) -> None:
    make_pdf([PUMP_PAGE], name="Pump Manual.pdf")
    make_pdf([BOP_PAGE], name="BOP_Guide.pdf")
    store = IndexStore(tmp_path / "index")

    ingest_directory(tmp_path, store, dense=dense)

    assert dense.count() == len(store.chunks())
    assert dense.count_document("bop-guide") == 1


class FailingDense:
    """A dense index whose writes fail (e.g. Qdrant went down mid-run)."""

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        raise ConnectionError("Qdrant unreachable")


def test_a_document_is_only_listed_once_both_indexes_are_written(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    store = IndexStore(tmp_path / "index")
    with pytest.raises(ConnectionError):
        ingest_file(make_pdf([PUMP_PAGE]), store, dense=FailingDense())  # type: ignore[arg-type]
    assert store.documents() == []
