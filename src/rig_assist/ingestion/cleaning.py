"""Clean parsed pages before chunking.

Ported from v1 (`strip_repeated_lines` in `src/rag/parsing.py`). It runs on the
output of any parser, so OCR'd pages lose their running headers too.
"""

import re
from collections import Counter
from dataclasses import replace

from rig_assist.ingestion.models import ParsedPage


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
