"""Read the text layer embedded in a PDF with pdfplumber, tables included as Markdown.

Ported from v1 (`src/rag/parsing.py`). v1 read text with pypdf and tables with
pdfplumber; here pdfplumber does both, so there's one PDF library instead of two.
Scanned pages have no text layer: they come back as empty text, and the caller
decides whether to OCR them (see `has_text_layer`).
"""

from collections.abc import Sequence
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page
from pdfplumber.utils.exceptions import PdfminerException

from rig_assist.ingestion.models import ParsedPage

PARSER_NAME = "pdfplumber"
# A page needs at least this many letters/digits to count as having a text layer.
# Lower than any real page ("Section 7.0 Causes of Kicks"), higher than a lone "35/68".
MIN_TEXT_CHARS = 10

# Sequence (read-only), not list: lets callers pass list[list[str]] as well (mypy variance).
Row = Sequence[str | None]


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return cell.replace("\n", " ").replace("|", r"\|").strip()


def table_to_markdown(rows: Sequence[Row]) -> str:
    """Render table rows as a Markdown table, one row per line (the chunker relies on it)."""
    if not rows:
        return ""
    header, *body = [[_clean_cell(c) for c in row] for row in rows]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def is_real_table(rows: Sequence[Row]) -> bool:
    """Real tables have 2+ columns and some text; title boxes and empty frames don't."""
    if not any(len(row) >= 2 for row in rows):
        return False
    return any(_clean_cell(cell) for row in rows for cell in row)


def has_text_layer(text: str) -> bool:
    """True if the page has real extractable text, False for scans and near-empty pages."""
    return sum(ch.isalnum() for ch in text) >= MIN_TEXT_CHARS


def _page_text(page: Page) -> str:
    text = (page.extract_text() or "").strip()
    tables = [table_to_markdown(t) for t in page.extract_tables() if is_real_table(t)]
    return "\n\n".join([text, *tables]).strip()


def extract_text_layer(path: Path, doc_id: str) -> list[ParsedPage]:
    """One ParsedPage per PDF page, numbered from 1. Pages with no text get text ""."""
    try:
        with pdfplumber.open(path) as pdf:
            return [
                ParsedPage(doc_id=doc_id, page=number, text=_page_text(page), parser=PARSER_NAME)
                for number, page in enumerate(pdf.pages, start=1)
            ]
    except PdfminerException as error:
        raise ValueError(f"{path.name} is not a readable PDF: {error}") from error
