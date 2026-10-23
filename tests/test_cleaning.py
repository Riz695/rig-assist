from rig_assist.ingestion.cleaning import strip_repeated_lines
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
