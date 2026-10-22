import pytest
import tiktoken

from rig_assist.ingestion.chunking import chunk_pages, chunk_text, detect_section
from rig_assist.ingestion.models import Chunk, ParsedPage

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def make_lines(n_lines: int, words_per_line: int = 8) -> str:
    """Numbered lines, so tests can check that nothing is lost or reordered."""
    return "\n".join(
        " ".join(f"line{i}word{j}" for j in range(words_per_line)) for i in range(n_lines)
    )


def make_page(page: int, text: str, doc_id: str = "drilling-manual") -> ParsedPage:
    return ParsedPage(doc_id=doc_id, page=page, text=text, parser="pdfplumber")


# --- chunk_text ---------------------------------------------------------


def test_chunk_text_returns_nothing_for_empty_or_whitespace() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_keeps_short_text_as_one_chunk() -> None:
    assert chunk_text("Close the annular preventer.\nCheck the gauge.") == [
        "Close the annular preventer.\nCheck the gauge."
    ]


def test_chunk_text_never_exceeds_size() -> None:
    chunks = chunk_text(make_lines(400), size=200, overlap=20)
    assert len(chunks) > 1
    assert all(token_count(c) <= 200 for c in chunks)


def test_chunk_text_breaks_only_between_lines() -> None:
    text = make_lines(400)
    original_lines = set(text.splitlines())
    for chunk in chunk_text(text, size=200, overlap=20):
        assert all(line in original_lines for line in chunk.splitlines())


def test_chunk_text_keeps_every_line_in_order() -> None:
    text = make_lines(400)
    chunks = chunk_text(text, size=200, overlap=20)
    seen: list[str] = []
    for chunk in chunks:
        seen.extend(line for line in chunk.splitlines() if line not in seen)
    assert seen == text.splitlines()


def test_chunk_text_overlaps_with_trailing_lines_of_previous_chunk() -> None:
    chunks = chunk_text(make_lines(400), size=200, overlap=40)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_line = previous.splitlines()[-1]
        assert current.splitlines()[0] in previous.splitlines()
        assert last_line in current.splitlines()
        shared = [line for line in current.splitlines() if line in previous.splitlines()]
        assert token_count("\n".join(shared)) <= 40


def test_chunk_text_does_not_split_table_rows() -> None:
    header = "| Rig size | Maximum drilling depth (ft) |\n| --- | --- |"
    rows = "\n".join(f"| Rig class {i} heavy duty | {i * 1000} - {i * 1500} |" for i in range(80))
    text = f"Land rigs classified by drilling depth\n\n{header}\n{rows}"
    for chunk in chunk_text(text, size=150, overlap=20):
        for line in chunk.splitlines():
            if line.startswith("|"):
                assert line.endswith("|"), f"row split mid-way: {line!r}"


def test_chunk_text_splits_a_single_line_longer_than_size() -> None:
    long_line = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(long_line, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(token_count(c) <= 100 for c in chunks)
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word999")


def test_chunk_text_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", size=50, overlap=50)


# --- detect_section -----------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "11.4.1 Land rigs classified by drilling depth",
            "11.4.1 Land rigs classified by drilling depth",
        ),
        ("4 .5.1 Incident Reporting", "4.5.1 Incident Reporting"),
        ("9.2 Redundancy", "9.2 Redundancy"),
        ("## Mud Pump Maintenance", "Mud Pump Maintenance"),
        ("Section 5.0", "Section 5.0"),
        ("Chapter 2 Drilling Systems", "Chapter 2 Drilling Systems"),
    ],
)
def test_detect_section_recognises_headings(line: str, expected: str) -> None:
    assert detect_section(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",
        "The valve is either a ball type or a flapper device.",
        # Numbered clause that is a full sentence, not a heading.
        "6 .4.6 To maintain occupation of confined spaces, the atmosphere within the space shall "
        "be monitored continuously",
        "1.2 The pump is checked daily.",
        # Table-of-contents entry with dot leaders.
        "21.1 WHAT IS UNDER-BALANCED DRILLING?........................158",
        "3.5 psi is the maximum",
        "| 1.5 | Medium duty |",
    ],
)
def test_detect_section_rejects_non_headings(line: str) -> None:
    assert detect_section(line) is None


# --- chunk_pages --------------------------------------------------------


def test_chunk_pages_attaches_metadata_and_deterministic_ids() -> None:
    pages = [make_page(1, make_lines(3)), make_page(2, make_lines(300))]
    chunks = chunk_pages(pages, doc_title="Drilling Manual", size=200, overlap=20)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.doc_id == "drilling-manual" for c in chunks)
    assert all(c.doc_title == "Drilling Manual" for c in chunks)
    assert all(c.parser == "pdfplumber" for c in chunks)
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == "drilling-manual::p1::c0"
    page_two = [c for c in chunks if c.page == 2]
    assert len(page_two) > 1
    assert [c.chunk_id for c in page_two] == [
        f"drilling-manual::p2::c{i}" for i in range(len(page_two))
    ]


def test_chunk_pages_never_mixes_text_from_two_pages() -> None:
    pages = [make_page(1, "alpha\n" * 40), make_page(2, "omega\n" * 40)]
    for chunk in chunk_pages(pages, doc_title="x", size=30, overlap=5):
        assert ("alpha" in chunk.text) != ("omega" in chunk.text)


def test_chunk_pages_skips_empty_pages() -> None:
    pages = [make_page(1, ""), make_page(2, "  \n "), make_page(3, "real content")]
    assert [c.page for c in chunk_pages(pages, doc_title="x")] == [3]


def test_chunk_pages_produces_no_empty_chunks() -> None:
    pages = [make_page(i, f"\n\n{make_lines(50)}\n\n\n") for i in range(1, 4)]
    assert all(c.text.strip() for c in chunk_pages(pages, doc_title="x", size=100, overlap=10))


def test_chunk_ids_are_unique_across_a_document() -> None:
    pages = [make_page(i, make_lines(200)) for i in range(1, 6)]
    ids = [c.chunk_id for c in chunk_pages(pages, doc_title="x", size=200, overlap=20)]
    assert len(ids) == len(set(ids))


def test_chunk_pages_sets_section_from_heading_and_carries_it_to_later_pages() -> None:
    pages = [
        make_page(1, "Introduction text without a heading."),
        make_page(2, "9.2 Redundancy\nThrusters must be redundant."),
        make_page(3, "More about thrusters on the next page."),
    ]
    sections = {c.page: c.section for c in chunk_pages(pages, doc_title="x")}
    assert sections == {1: None, 2: "9.2 Redundancy", 3: "9.2 Redundancy"}


def test_chunk_pages_uses_the_section_in_effect_at_the_start_of_each_chunk() -> None:
    first = "\n".join(f"Pump detail line {i} about liners and valves." for i in range(30))
    second = "\n".join(f"Mud detail line {i} about weight and viscosity." for i in range(30))
    text = f"3.1 Mud Pumps\n{first}\n3.2 Drilling Fluids\n{second}"
    chunks = chunk_pages([make_page(1, text)], doc_title="x", size=120, overlap=0)

    assert chunks[0].section == "3.1 Mud Pumps"
    for chunk in chunks:
        first_line = chunk.text.splitlines()[0]
        if first_line.startswith("Mud detail"):
            assert chunk.section == "3.2 Drilling Fluids"
        if first_line.startswith("Pump detail"):
            assert chunk.section == "3.1 Mud Pumps"
