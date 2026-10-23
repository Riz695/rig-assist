from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import RAW_MANUALS_DIR

from rig_assist.ingestion.parsers.text_layer import (
    extract_text_layer,
    has_text_layer,
    is_real_table,
    table_to_markdown,
)

MakePdf = Callable[[list[list[str]]], Path]

# --- table_to_markdown / is_real_table (ported from v1) -----------------


def test_table_to_markdown_builds_header_separator_and_rows() -> None:
    rows = [["Zone", "Class"], ["1", "A"], ["2", "B"]]
    assert table_to_markdown(rows) == "| Zone | Class |\n| --- | --- |\n| 1 | A |\n| 2 | B |"


def test_table_to_markdown_turns_none_cells_into_empty_strings() -> None:
    assert table_to_markdown([["Zone", "Class"], ["1", None]]).splitlines()[2] == "| 1 |  |"


def test_table_to_markdown_escapes_pipes_inside_cells() -> None:
    assert table_to_markdown([["Item"], ["a|b"]]).splitlines()[2] == r"| a\|b |"


def test_table_to_markdown_flattens_newlines_inside_cells() -> None:
    rows = [["Item"], ["line one\nline two"]]
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


# --- has_text_layer -----------------------------------------------------


@pytest.mark.parametrize("text", ["", "   \n ", "35/68", "- . -"])
def test_has_text_layer_is_false_for_empty_or_near_empty_pages(text: str) -> None:
    assert not has_text_layer(text)


def test_has_text_layer_is_true_for_real_text() -> None:
    assert has_text_layer("Section 7.0 Causes of Kicks 35/68")


# --- extract_text_layer -------------------------------------------------


def test_extract_text_layer_returns_one_page_per_pdf_page_numbered_from_one(
    make_pdf: MakePdf,
) -> None:
    path = make_pdf([["Mud pump liner wear"], ["Check the relief valve"], ["Annular BOP"]])
    pages = extract_text_layer(path, doc_id="drilling-manual")

    assert [p.page for p in pages] == [1, 2, 3]
    assert all(p.doc_id == "drilling-manual" for p in pages)
    assert all(p.parser == "pdfplumber" for p in pages)
    assert "relief valve" in pages[1].text


def test_extract_text_layer_keeps_pages_without_text_as_empty(make_pdf: MakePdf) -> None:
    pages = extract_text_layer(make_pdf([["Page one text"], [], ["Page three text"]]), "d")

    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert not has_text_layer(pages[1].text)


def test_extract_text_layer_rejects_a_file_that_is_not_a_pdf(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_text("this is not a pdf")
    with pytest.raises(ValueError, match=r"notes\.pdf"):
        extract_text_layer(path, "d")


# --- real manuals (skipped when data/raw is absent, e.g. in CI) ---------


def _manual(name: str) -> Path:
    path = RAW_MANUALS_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not in data/raw")
    return path


@pytest.mark.integration
def test_real_manual_tables_come_out_as_markdown() -> None:
    pages = extract_text_layer(_manual("Drilling_Manual.pdf"), "drilling-manual")
    assert any("| --- |" in p.text for p in pages)


@pytest.mark.integration
def test_scanned_manual_has_no_text_layer() -> None:
    pages = extract_text_layer(_manual("MarineOffshoreLogOpsManual.pdf"), "marine-ops")
    assert len(pages) == 146
    assert not any(has_text_layer(p.text) for p in pages)
