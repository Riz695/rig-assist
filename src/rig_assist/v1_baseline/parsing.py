"""v1's PDF parser (`src/rag/parsing.py`): pypdf for text, pdfplumber for tables.

Each page = pypdf's text, then any real tables as Markdown appended below it (so table
text can appear twice). Lines repeated on most pages are dropped. No OCR: a scanned page
comes back as "" and v1's chunker skips it.
"""

import re
from collections import Counter
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page
from pypdf import PageObject, PdfReader

from rig_assist.ingestion.models import ParsedPage

PARSER_NAME = "v1-pypdf"


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return cell.replace("\n", " ").replace("|", r"\|").strip()


def table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def is_real_table(rows: list[list[str | None]]) -> bool:
    """Real tables have 2+ columns and some text; title boxes and empties do not."""
    if not any(len(row) >= 2 for row in rows):
        return False
    return any(_clean_cell(cell) for row in rows for cell in row)


def extract_page(plumber_page: Page, pypdf_page: PageObject) -> str:
    text = (pypdf_page.extract_text() or "").strip()
    tables = [table_to_markdown(t) for t in plumber_page.extract_tables() if is_real_table(t)]
    return "\n\n".join([text, *tables]).strip()


def _line_key(line: str) -> str:
    """Normalise digits and spacing so 'Page 2 of 205' and 'Page 90 of 205' match."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def strip_repeated_lines(
    pages: list[tuple[int, str]], min_fraction: float = 0.5, min_pages: int = 5
) -> list[tuple[int, str]]:
    """Drop lines repeated on most pages (running headers/footers), keep page numbers."""
    if len(pages) < min_pages:
        return pages
    seen_on: Counter[str] = Counter()
    for _, text in pages:
        seen_on.update({_line_key(ln) for ln in text.splitlines() if ln.strip()})
    boilerplate = {key for key, n in seen_on.items() if n / len(pages) >= min_fraction}
    return [
        (
            number,
            "\n".join(ln for ln in text.splitlines() if _line_key(ln) not in boilerplate).strip(),
        )
        for number, text in pages
    ]


def parse_pdf(path: Path, doc_id: str) -> list[ParsedPage]:
    """Every page of the PDF, 1-based, including empty ones (text "")."""
    reader = PdfReader(path)
    with pdfplumber.open(path) as pdf:
        pages = [
            (number, extract_page(plumber_page, pypdf_page))
            for number, (plumber_page, pypdf_page) in enumerate(
                zip(pdf.pages, reader.pages, strict=True), start=1
            )
        ]
    return [
        ParsedPage(doc_id, number, text, PARSER_NAME)
        for number, text in strip_repeated_lines(pages)
    ]
