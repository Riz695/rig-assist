import json
from pathlib import Path

import pytest

from rig_assist.ingestion.models import Chunk
from rig_assist.ingestion.store import DocumentInfo, IndexStore


def make_info(doc_id: str, chunks: int = 2, missing: list[int] | None = None) -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id,
        title=doc_id.replace("-", " ").title(),
        filename=f"{doc_id}.pdf",
        file_sha256="ab" * 32,
        pages=3,
        chunks=chunks,
        missing_pages=missing or [],
    )


def make_chunks(doc_id: str, n: int = 2) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_id}::p{i + 1}::c0",
            doc_id=doc_id,
            doc_title=doc_id.title(),
            page=i + 1,
            section="3.1 Well Control" if i == 0 else None,
            parser="pdfplumber",
            text=f"Chunk {i} of {doc_id}: close the annular preventer",
        )
        for i in range(n)
    ]


def test_empty_store_has_no_documents_or_chunks(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index")
    assert store.documents() == []
    assert store.chunks() == []


def test_put_then_read_back(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.put(make_info("drilling-manual", missing=[4]), make_chunks("drilling-manual"))

    assert store.documents() == [make_info("drilling-manual", missing=[4])]
    assert store.chunks() == make_chunks("drilling-manual")


def test_store_survives_a_restart(tmp_path: Path) -> None:
    IndexStore(tmp_path).put(make_info("drilling-manual"), make_chunks("drilling-manual"))
    reopened = IndexStore(tmp_path)
    assert reopened.documents() == [make_info("drilling-manual")]
    assert reopened.chunks() == make_chunks("drilling-manual")


def test_putting_a_document_again_replaces_it_without_duplicates(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.put(make_info("drilling-manual", chunks=2), make_chunks("drilling-manual", 2))
    store.put(make_info("drilling-manual", chunks=3), make_chunks("drilling-manual", 3))

    assert [d.chunks for d in store.documents()] == [3]
    assert len(store.chunks()) == 3


def test_documents_and_chunks_come_back_in_doc_id_order(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.put(make_info("offshore-manual"), make_chunks("offshore-manual"))
    store.put(make_info("cds-guide"), make_chunks("cds-guide"))

    assert [d.doc_id for d in store.documents()] == ["cds-guide", "offshore-manual"]
    assert [c.doc_id for c in store.chunks()] == ["cds-guide"] * 2 + ["offshore-manual"] * 2


def test_files_are_plain_json_not_pickle(tmp_path: Path) -> None:
    IndexStore(tmp_path).put(make_info("drilling-manual"), make_chunks("drilling-manual"))

    info = json.loads((tmp_path / "documents" / "drilling-manual.json").read_text("utf-8"))
    assert info["title"] == "Drilling Manual"
    lines = (tmp_path / "chunks" / "drilling-manual.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["chunk_id"] for line in lines] == [
        "drilling-manual::p1::c0",
        "drilling-manual::p2::c0",
    ]


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "", "Upper", "with space"])
def test_unsafe_doc_ids_are_rejected(tmp_path: Path, bad_id: str) -> None:
    """doc_id becomes a file name; it must never point outside the store."""
    with pytest.raises(ValueError, match="doc_id"):
        IndexStore(tmp_path).put(make_info(bad_id), [])


def test_one_document_by_id(tmp_path: Path) -> None:
    store = IndexStore(tmp_path)
    store.put(make_info("drilling-manual"), make_chunks("drilling-manual"))
    assert store.document("drilling-manual") == make_info("drilling-manual")
    assert store.document("no-such-manual") is None


@pytest.mark.parametrize("doc_id", ["..", "../index", "Drilling_Manual", ""])
def test_a_doc_id_that_is_not_a_slug_is_never_looked_up(tmp_path: Path, doc_id: str) -> None:
    store = IndexStore(tmp_path / "index")
    (tmp_path / "index").mkdir()
    (tmp_path / "x.json").write_text("{}")  # would be reachable through a bad id
    assert store.document(doc_id) is None
