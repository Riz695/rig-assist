import pytest

from rig_assist.ingestion.cleaning import (
    collapse_dot_leaders,
    html_tables_to_markdown,
    strip_repeated_lines,
)
from rig_assist.ingestion.models import ParsedPage

WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
]


def make_pages(lines_per_page: list[list[str]], parser: str = "pdfplumber") -> list[ParsedPage]:
    return [
        ParsedPage(doc_id="d", page=i, text="\n".join(lines), parser=parser)
        for i, lines in enumerate(lines_per_page, start=1)
    ]


def texts(pages: list[ParsedPage]) -> list[str]:
    return [p.text for p in pages]


def test_removes_header_found_on_most_pages() -> None:
    header = "Weatherford confidential header"
    result = strip_repeated_lines(make_pages([[header, w] for w in WORDS]))
    assert texts(result) == WORDS


def test_keeps_page_numbers_doc_id_and_parser() -> None:
    pages = make_pages([["header", w] for w in WORDS], parser="olmocr")
    result = strip_repeated_lines(pages)
    assert [p.page for p in result] == list(range(1, 11))
    assert {(p.doc_id, p.parser) for p in result} == {("d", "olmocr")}


def test_keeps_lines_that_are_not_widespread() -> None:
    lines = [["header", w] for w in WORDS]
    lines[3] = ["header", "unique warning text"]
    result = strip_repeated_lines(make_pages(lines))
    assert result[3].text == "unique warning text"
    assert "unique" not in result[0].text


def test_removes_footer_whose_page_number_changes() -> None:
    lines = [[f"{w} text", f"For training only    Page {n} of 205"] for n, w in enumerate(WORDS, 1)]
    assert texts(strip_repeated_lines(make_pages(lines))) == [f"{w} text" for w in WORDS]


def test_keeps_numeric_lines_that_are_not_widespread() -> None:
    lines = [["common header", f"filler {w}"] for w in WORDS]
    lines[2] = ["common header", "Pressure is 5 psi"]
    lines[6] = ["common header", "Pressure is 9 psi"]
    result = strip_repeated_lines(make_pages(lines))
    assert result[2].text == "Pressure is 5 psi"
    assert result[6].text == "Pressure is 9 psi"


def test_leaves_short_documents_alone() -> None:
    pages = make_pages([["header", "a"], ["header", "b"]])
    assert strip_repeated_lines(pages) == pages


def test_ignores_blank_lines() -> None:
    assert texts(strip_repeated_lines(make_pages([["", w, ""] for w in WORDS]))) == WORDS


def test_empty_pages_do_not_lower_the_threshold() -> None:
    """A scanned manual's empty pages shouldn't make a header look rare (or common)."""
    lines = [["header", w] for w in WORDS[:6]] + [[] for _ in range(6)]
    result = strip_repeated_lines(make_pages(lines))
    assert texts(result)[:6] == WORDS[:6]
    assert texts(result)[6:] == [""] * 6


# --- dot leaders (tables of contents) -----------------------------------
# A run of dots costs a few tokens in cl100k but one token per dot in the embedding
# model's WordPiece tokenizer, so TOC chunks blew past its 512-token limit.


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1 INTRODUCTION......................7", "1 INTRODUCTION … 7"),
        ("Section 1 Classification.............3", "Section 1 Classification … 3"),
        ("7.11 Pressure Equipment . . . . . . . . . . 38", "7.11 Pressure Equipment … 38"),
        ("(IRC) .............................205", "(IRC) … 205"),
    ],
)
def test_collapse_dot_leaders(line: str, expected: str) -> None:
    assert collapse_dot_leaders(line) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Pressure is 3.5 psi.",
        "Wait... then close the valve.",
        "See sections 4.4.1 and 4.4.2.",
        "U.S.A. regulations",
    ],
)
def test_collapse_dot_leaders_keeps_ordinary_text(text: str) -> None:
    assert collapse_dot_leaders(text) == text


def test_strip_repeated_lines_is_unaffected_by_leader_collapsing() -> None:
    pages = make_pages([["Contents", f"{w} section........{i}"] for i, w in enumerate(WORDS)])
    assert all("…" not in p.text for p in strip_repeated_lines(pages))


# --- OCR tables: HTML -> Markdown ---------------------------------------------------
# olmOCR writes tables as HTML, one row per line. In bge's tokenizer every tag costs
# tokens, so blank forms overflowed its 512-token limit. Markdown keeps the content and
# matches the pdfplumber parser's tables.


def test_html_table_becomes_the_same_markdown_as_pdfplumber_tables() -> None:
    html = (
        "<table>\n<tr><th>Pump</th><th>Max pressure</th></tr>\n"
        "<tr><td>P-101</td><td>350 bar</td></tr>\n</table>"
    )
    assert html_tables_to_markdown(html) == (
        "| Pump | Max pressure |\n| --- | --- |\n| P-101 | 350 bar |"
    )


def test_text_around_tables_is_kept() -> None:
    text = "Before\n<table><tr><td>a</td><td>b</td></tr></table>\nAfter"
    assert html_tables_to_markdown(text) == "Before\n| a | b |\n| --- | --- |\nAfter"


def test_several_tables_on_one_page() -> None:
    text = "<table><tr><td>a</td></tr></table>\nmid\n<table><tr><td>b</td></tr></table>"
    assert html_tables_to_markdown(text) == "| a |\n| --- |\nmid\n| b |\n| --- |"


def test_inline_formatting_and_entities_inside_cells() -> None:
    html = (
        "<table><tr><td><b>Flow</b><br>rate</td><td>&gt; 20 m<sup>3</sup>/h</td></tr>"
        "<tr><td><ul><li>one</li><li>two</li></ul></td><td>a | b</td></tr></table>"
    )
    assert html_tables_to_markdown(html) == (
        "| Flow rate | > 20 m3/h |\n| --- | --- |\n| one two | a \\| b |"
    )


def test_unescaped_less_than_in_a_cell_is_kept_as_text() -> None:
    html = "<table><tr><td>Wind</td><td><20 knots</td><td><finished goods</td></tr></table>"
    assert html_tables_to_markdown(html) == (
        "| Wind | <20 knots | <finished goods |\n| --- | --- | --- |"
    )


def test_colspan_pads_empty_cells_so_columns_stay_aligned() -> None:
    html = (
        '<table><tr><th colspan="2">Valve</th><th>State</th></tr>'
        "<tr><td>V1</td><td>gate</td><td>open</td></tr></table>"
    )
    assert html_tables_to_markdown(html).splitlines()[0] == "| Valve |  | State |"


def test_completely_empty_rows_are_dropped_but_numbered_blank_rows_stay() -> None:
    html = (
        "<table><tr><th>No.</th><th>Date</th></tr>"
        "<tr><td></td><td> </td></tr>"
        "<tr><td>1</td><td></td></tr></table>"
    )
    assert html_tables_to_markdown(html) == "| No. | Date |\n| --- | --- |\n| 1 |  |"


def test_a_table_with_no_text_at_all_disappears() -> None:
    assert html_tables_to_markdown("Form\n<table><tr><td></td></tr></table>") == "Form\n"


@pytest.mark.parametrize(
    "text",
    ["No tables here.", "Pressure < 20 bar and > 5 bar", "| already | markdown |\n| --- | --- |"],
)
def test_text_without_html_tables_is_unchanged(text: str) -> None:
    assert html_tables_to_markdown(text) == text
