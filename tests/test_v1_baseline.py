"""The v1 baseline: v1's parser, token-window chunker and retrieval settings, ported.

The chunking and parsing tests are v1's own (tests/test_chunking.py, tests/test_parsing.py
in offshore-drilling-rag), adapted to v2's ParsedPage/Chunk types, so the port is shown
to behave like the original.
"""

from pathlib import Path

import pytest
import tiktoken
from conftest import RAW_MANUALS_DIR, FakeOcr, KeywordEmbedder, build_pdf, no_sleep
from qdrant_client import QdrantClient

from rig_assist.ingestion.models import Chunk, ParsedPage
from rig_assist.ingestion.parsing import parse_document
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.dense import DenseIndex
from rig_assist.retrieval.hybrid import HybridRetriever
from rig_assist.v1_baseline.chunking import chunk_pages, chunk_text
from rig_assist.v1_baseline.parsing import (
    is_real_table,
    parse_pdf,
    strip_repeated_lines,
    table_to_markdown,
)
from rig_assist.v1_baseline.pipeline import (
    V1_CONFIGS,
    build_v1_index,
    make_v1_retriever,
    v1_collection,
    v1_store_dir,
)

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def make_text(n_words: int) -> str:
    return " ".join(f"drill{i}" for i in range(n_words))


def pages_of(*texts: str, doc_id: str = "drilling-manual") -> list[ParsedPage]:
    return [ParsedPage(doc_id, n, text, "v1-pypdf") for n, text in enumerate(texts, start=1)]


# --- chunk_text (v1's token windows) ------------------------------------


def test_chunk_text_returns_nothing_for_empty_or_whitespace() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_keeps_short_text_as_one_chunk() -> None:
    assert chunk_text("A short sentence about blowout preventers.") == [
        "A short sentence about blowout preventers."
    ]


def test_chunk_text_never_exceeds_size() -> None:
    chunks = chunk_text(make_text(2000), size=500, overlap=50)
    assert len(chunks) > 1
    assert all(token_count(c) <= 500 for c in chunks)


def test_chunk_text_overlaps_consecutive_chunks_by_exact_token_count() -> None:
    chunks = chunk_text(make_text(2000), size=500, overlap=50)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert ENCODING.encode(previous)[-50:] == ENCODING.encode(current)[:50]


def test_chunk_text_does_not_drop_the_tail() -> None:
    chunks = chunk_text(make_text(2000), size=500, overlap=50)
    assert chunks[0].startswith("drill0")
    assert chunks[-1].endswith("drill1999")


def test_chunk_text_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", size=50, overlap=50)


def test_chunk_text_cuts_mid_line_unlike_v2s_chunker() -> None:
    # v1 counts raw tokens, so a table row can be split across chunks (v2 never does).
    row = "| Pump | Liner | 6.5 in | 7500 psi |"
    text = "\n".join([row] * 200)
    first = chunk_text(text, size=100, overlap=10)[0]
    assert not first.endswith("|")


# --- chunk_pages ----------------------------------------------------------


def test_chunk_pages_attaches_metadata_and_deterministic_ids() -> None:
    chunks = chunk_pages(pages_of(make_text(30), make_text(1200)), doc_title="Drilling Manual")

    assert all(isinstance(c, Chunk) for c in chunks)
    first = chunks[0]
    assert (first.doc_id, first.doc_title, first.page) == ("drilling-manual", "Drilling Manual", 1)
    assert first.chunk_id == "drilling-manual::p1::c0"
    assert first.section is None  # v1 detected no sections
    assert first.parser == "v1-pypdf"
    page_two = [c for c in chunks if c.page == 2]
    assert len(page_two) > 1
    assert [c.chunk_id for c in page_two] == [
        f"drilling-manual::p2::c{i}" for i in range(len(page_two))
    ]


def test_chunk_pages_never_mixes_text_from_two_pages() -> None:
    for chunk in chunk_pages(pages_of("alpha " * 40, "omega " * 40), doc_title="X"):
        assert ("alpha" in chunk.text) != ("omega" in chunk.text)


def test_chunk_pages_skips_empty_pages() -> None:
    chunks = chunk_pages(pages_of("", "   ", "real content"), doc_title="X")
    assert [c.page for c in chunks] == [3]


def test_chunk_pages_keeps_each_pages_parser() -> None:
    pages = [
        ParsedPage("d", 1, "text layer page", "pdfplumber"),
        ParsedPage("d", 2, "scanned page", "olmocr"),
    ]
    assert [c.parser for c in chunk_pages(pages, doc_title="D")] == ["pdfplumber", "olmocr"]


# --- v1 parsing: tables -------------------------------------------------


def test_table_to_markdown_builds_header_separator_and_rows() -> None:
    rows: list[list[str | None]] = [["Zone", "Class"], ["1", "A"], ["2", "B"]]
    assert table_to_markdown(rows) == "| Zone | Class |\n| --- | --- |\n| 1 | A |\n| 2 | B |"


def test_table_to_markdown_turns_none_cells_into_empty_strings() -> None:
    assert table_to_markdown([["Zone", "Class"], ["1", None]]).splitlines()[2] == "| 1 |  |"


def test_table_to_markdown_escapes_pipes_inside_cells() -> None:
    assert table_to_markdown([["Item"], ["a|b"]]).splitlines()[2] == r"| a\|b |"


def test_table_to_markdown_flattens_newlines_inside_cells() -> None:
    rows: list[list[str | None]] = [["Item"], ["line one\nline two"]]
    assert table_to_markdown(rows).splitlines()[2] == "| line one line two |"


def test_table_to_markdown_returns_empty_string_for_no_rows() -> None:
    assert table_to_markdown([]) == ""


def test_is_real_table_accepts_multi_column_table_with_content() -> None:
    assert is_real_table([["Zone", "Class"], ["1", "A"]])


def test_is_real_table_rejects_single_column_title_box() -> None:
    assert not is_real_table([[""], ["Section 1.0 Prerequisite Material 4/68"], [""]])


def test_is_real_table_rejects_table_with_only_empty_cells() -> None:
    assert not is_real_table([["", None], [None, ""]])


def test_is_real_table_rejects_no_rows() -> None:
    assert not is_real_table([])


# --- v1 parsing: repeated header/footer lines ---------------------------

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]


def numbered(lines_per_page: list[list[str]]) -> list[tuple[int, str]]:
    return [(i, "\n".join(lines)) for i, lines in enumerate(lines_per_page, start=1)]


def test_strip_repeated_lines_removes_header_found_on_most_pages() -> None:
    header = "Weatherford confidential header"
    result = strip_repeated_lines(numbered([[header, w] for w in WORDS]))
    assert [text for _, text in result] == WORDS


def test_strip_repeated_lines_keeps_page_numbers() -> None:
    result = strip_repeated_lines(numbered([["header", w] for w in WORDS]))
    assert [n for n, _ in result] == list(range(1, len(WORDS) + 1))


def test_strip_repeated_lines_keeps_lines_that_are_not_widespread() -> None:
    pages = numbered([["header", w] for w in WORDS])
    pages[3] = (4, "header\nunique warning text")
    result = dict(strip_repeated_lines(pages))
    assert result[4] == "unique warning text"
    assert "unique" not in result[1]


def test_strip_repeated_lines_leaves_short_documents_alone() -> None:
    pages = numbered([["header", "a"], ["header", "b"]])
    assert strip_repeated_lines(pages) == pages


def test_strip_repeated_lines_ignores_blank_lines() -> None:
    assert [t for _, t in strip_repeated_lines(numbered([["", w, ""] for w in WORDS]))] == WORDS


def test_strip_repeated_lines_removes_footer_whose_page_number_changes() -> None:
    pages = numbered(
        [[f"{w} text", f"For training only    Page {n} of 205"] for n, w in enumerate(WORDS, 1)]
    )
    assert [text for _, text in strip_repeated_lines(pages)] == [f"{w} text" for w in WORDS]


def test_strip_repeated_lines_keeps_numeric_lines_that_are_not_widespread() -> None:
    pages = numbered([["common header", f"filler {c}"] for c in "abcdefghij"])
    pages[2] = (3, "common header\nPressure is 5 psi")
    pages[6] = (7, "common header\nPressure is 9 psi")
    result = dict(strip_repeated_lines(pages))
    assert result[3] == "Pressure is 5 psi"
    assert result[7] == "Pressure is 9 psi"


# --- v1 parsing: whole PDF ----------------------------------------------


def test_parse_pdf_returns_every_page_numbered_from_one_with_pypdf_text(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(build_pdf([["Check the pump liners."], [], ["Close the BOP."]]))

    pages = parse_pdf(path, doc_id="manual")

    assert [p.page for p in pages] == [1, 2, 3]
    assert {p.doc_id for p in pages} == {"manual"}
    assert {p.parser for p in pages} == {"v1-pypdf"}
    assert "pump liners" in pages[0].text
    assert pages[1].text == ""  # a scanned page: v1 has no OCR, so it stays empty


@pytest.mark.integration
def test_parse_pdf_on_a_real_manual_finds_a_markdown_table() -> None:
    path = RAW_MANUALS_DIR / "cds-guide-feb21.pdf"
    if not path.exists():
        pytest.skip("cds-guide-feb21.pdf not in data/raw")
    pages = parse_pdf(path, doc_id="cds-guide-feb21")
    assert pages[0].page == 1
    assert any("| --- |" in p.text for p in pages)


# --- the two configurations ---------------------------------------------


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "Drilling_Manual.pdf").write_bytes(
        build_pdf(
            [
                ["3.1 Mud Pumps", "Check the pump liners and valves every shift."],
                [],  # scanned page
                ["5.1 Blowout Preventer", "Close the annular preventer when a kick is detected."],
            ]
        )
    )
    return raw


@pytest.fixture
def dense() -> DenseIndex:
    return DenseIndex(QdrantClient(":memory:"), "v1_test", KeywordEmbedder())


def v2_pages_with_ocr(path: Path, doc_id: str) -> list[ParsedPage]:
    """v2's parser with OCR on scanned pages (fake OCR client, no network)."""
    return parse_document(path, doc_id, "missing", FakeOcr(), sleep=no_sleep).pages


def test_the_configs_are_v1_end_to_end_and_v1_retrieval_on_v2_pages() -> None:
    assert V1_CONFIGS == ("v1_baseline", "v1_retrieval_on_v2_pages")


def test_v1_baseline_uses_v1s_parser_and_skips_the_scanned_page(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "v1")
    [info] = build_v1_index(raw_dir, "v1_baseline", store, dense)

    chunks = store.chunks()
    assert info.doc_id == "drilling-manual"
    assert info.pages == 3
    assert info.missing_pages == [2]
    assert sorted({c.page for c in chunks}) == [1, 3]
    assert {c.parser for c in chunks} == {"v1-pypdf"}
    assert info.chunks == len(chunks) == dense.count_document("drilling-manual")


def test_v1_retrieval_on_v2_pages_chunks_v2s_parsed_pages_including_ocr(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "v1-on-v2")
    [info] = build_v1_index(
        raw_dir, "v1_retrieval_on_v2_pages", store, dense, parse_v2=v2_pages_with_ocr
    )

    chunks = store.chunks()
    assert info.missing_pages == []
    assert sorted({c.page for c in chunks}) == [1, 2, 3]
    assert {c.parser for c in chunks if c.page == 2} == {"olmocr"}
    assert all(c.section is None for c in chunks)  # still v1's chunker


def test_v1_retrieval_on_v2_pages_needs_the_v2_parser(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    with pytest.raises(ValueError, match="parse_v2"):
        build_v1_index(raw_dir, "v1_retrieval_on_v2_pages", IndexStore(tmp_path / "x"), dense)


def test_an_unknown_config_raises(raw_dir: Path, tmp_path: Path, dense: DenseIndex) -> None:
    with pytest.raises(ValueError, match="v2_baseline"):
        build_v1_index(raw_dir, "v2_baseline", IndexStore(tmp_path / "x"), dense)  # type: ignore[arg-type]


def test_rebuilding_leaves_chunk_counts_unchanged_in_both_indexes(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "v1")
    build_v1_index(raw_dir, "v1_baseline", store, dense)
    first = (len(store.chunks()), dense.count())
    build_v1_index(raw_dir, "v1_baseline", store, dense)
    assert (len(store.chunks()), dense.count()) == first


def test_each_config_has_its_own_store_folder_and_qdrant_collection(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    dirs = {v1_store_dir(index_dir, config) for config in V1_CONFIGS}
    collections = {v1_collection("rig_assist_chunks", config) for config in V1_CONFIGS}
    assert len(dirs) == len(collections) == 2
    # Never the main index's own folders or collection.
    assert all(d.parent != index_dir and index_dir in d.parents for d in dirs)
    assert "rig_assist_chunks" not in collections


# --- the retriever ------------------------------------------------------


def test_v1_retriever_is_hybrid_with_v1s_settings_and_okapi_bm25(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "v1")
    build_v1_index(raw_dir, "v1_baseline", store, dense)

    retriever = make_v1_retriever(store, dense)

    assert isinstance(retriever, HybridRetriever)
    assert (retriever.k_each, retriever.rrf_k) == (20, 60)
    assert retriever.dense is dense
    assert retriever.bm25.idf == "okapi"  # type: ignore[attr-defined]


def test_v1_retriever_finds_the_page_and_keeps_the_dense_score(
    raw_dir: Path, tmp_path: Path, dense: DenseIndex
) -> None:
    store = IndexStore(tmp_path / "v1")
    build_v1_index(raw_dir, "v1_baseline", store, dense)

    top = make_v1_retriever(store, dense).search("mud pump liners", k=5)[0]

    assert top.chunk.page == 1
    assert top.dense_score is not None
