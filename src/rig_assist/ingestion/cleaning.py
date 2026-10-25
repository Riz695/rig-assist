"""Clean parsed pages before chunking.

Ported from v1 (`strip_repeated_lines` in `src/rag/parsing.py`). It runs on the
output of any parser, so OCR'd pages lose their running headers too. OCR tables are
rewritten from HTML to Markdown here, after the cache, so the cache keeps the raw output.
"""

import html
import re
from collections import Counter
from dataclasses import replace

from rig_assist.ingestion.models import ParsedPage
from rig_assist.ingestion.parsers.text_layer import table_to_markdown

# 4+ dots, optionally spaced (". . . ."), plus surrounding spaces: a table-of-contents leader.
# [ \t] not \s, so a leader never swallows a line break.
_DOT_LEADER = re.compile(r"[ \t]*(?:\.[ \t]?){4,}[ \t]*")


def collapse_dot_leaders(text: str) -> str:
    """'INTRODUCTION........7' → 'INTRODUCTION … 7'.

    Dot leaders carry no meaning but cost one token per dot in the embedding model's
    WordPiece tokenizer (few in cl100k), so table-of-contents chunks overflowed its
    512-token limit and were cut. Three dots ("wait...") and decimals are untouched.
    """
    return _DOT_LEADER.sub(" … ", text)


_TABLE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL = re.compile(r"<t([dh])\b([^>]*)>(.*?)</t\1>", re.IGNORECASE | re.DOTALL)
_COLSPAN = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
# Only the formatting tags olmOCR puts inside cells. A bare "<" in a cell ("<20 knots")
# is text, so a general HTML parser (which would read "<finished" as a tag) isn't used.
_INLINE_TAG = re.compile(r"</?(?:b|i|u|em|strong|sup|sub|span)\b[^>]*>", re.I)  # "m<sup>3</sup>"
_BREAK_TAG = re.compile(r"</?(?:br|p|ul|ol|li)\b[^>]*>", re.I)  # "Flow<br>rate"


def _cell_text(raw: str) -> str:
    text = _BREAK_TAG.sub(" ", _INLINE_TAG.sub("", raw))
    return " ".join(html.unescape(text).split())


def _html_table(match: re.Match[str]) -> str:
    rows = []
    for row in _ROW.findall(match.group(1)):
        cells: list[str] = []
        for _, attrs, raw in _CELL.findall(row):
            span = _COLSPAN.search(attrs)
            cells += [_cell_text(raw)] + [""] * (int(span.group(1)) - 1 if span else 0)
        if any(cells):
            rows.append(cells)
    # rowspan is ignored: the cell appears in its first row only.
    return table_to_markdown(rows)


def html_tables_to_markdown(text: str) -> str:
    """Rewrite olmOCR's HTML tables as Markdown tables, like the pdfplumber parser's.

    In bge's WordPiece tokenizer each `<td>`/`</td>` costs several tokens, so blank OCR'd
    forms reached 770 tokens (14 of them text) and were cut at 512. Rows with no text are
    dropped; a table with no text at all disappears. Text outside tables is unchanged.
    """
    return _TABLE.sub(_html_table, text)


def _line_key(line: str) -> str:
    """Normalise digits and spacing so "Page 2 of 205" and "Page 90 of 205" match."""
    return re.sub(r"\d+", "#", " ".join(line.split()))


def strip_repeated_lines(
    pages: list[ParsedPage], min_fraction: float = 0.5, min_pages: int = 5
) -> list[ParsedPage]:
    """Drop lines found on at least `min_fraction` of pages (running headers/footers).

    Only pages with text count towards the fraction, so a document with many blank
    or scanned pages doesn't hide its headers. Documents with fewer than `min_pages`
    pages with text are left alone: too few pages to tell a header from content.
    """
    with_text = [p for p in pages if p.text.strip()]
    if len(with_text) < min_pages:
        return pages
    seen_on: Counter[str] = Counter()
    for page in with_text:
        seen_on.update({_line_key(line) for line in page.text.splitlines() if line.strip()})
    boilerplate = {key for key, n in seen_on.items() if n / len(with_text) >= min_fraction}
    return [
        replace(
            page,
            text="\n".join(
                line for line in page.text.splitlines() if _line_key(line) not in boilerplate
            ).strip(),
        )
        for page in pages
    ]
