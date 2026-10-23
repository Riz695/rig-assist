"""Split parsed pages into overlapping, token-limited chunks.

Ported from v1 (`src/rag/chunking.py`) with one change: v1 cut the token stream every
`size` tokens, which can split a word or a table row in half. Here chunks are packed
from whole lines, so a Markdown table row (one line) stays intact. Only a single line
longer than `size` is cut by tokens.

Sizes are counted in `cl100k_base` tokens, as in v1, so v1's settings can be reproduced.
"""

import re

import tiktoken

from rig_assist.ingestion.models import Chunk, ParsedPage

_ENCODING = tiktoken.get_encoding("cl100k_base")

# "## Heading" (olmOCR and other Markdown output).
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*#*$")
# "11.4.1 Land rigs ..." or "4 .5.1 Incident Reporting" (stray space in one manual's text layer).
# At least two number levels: a single "3 Close the valve" is far more often a list item.
_NUMBERED_HEADING = re.compile(r"^(?P<number>\d+(?:\s?\.\d+)+)\.?\s+(?P<title>[A-Z].*)$")
# "Section 5.0", "Chapter 2 Drilling Systems", "APPENDIX 3 ...", "Appendix A".
_NAMED_HEADING = re.compile(
    r"^(?:Chapter|CHAPTER|Section|SECTION|Appendix|APPENDIX)\s+(?:\d[\d.]*|[A-Z])\b.*$"
)
_MAX_HEADING_CHARS = 80


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def detect_section(line: str) -> str | None:
    """Return the heading text if `line` looks like a section heading, else None.

    Deliberately conservative: a missed heading only leaves `section` empty, while a
    false one would label chunks with a sentence fragment.
    """
    line = line.strip()
    if not line or len(line) > _MAX_HEADING_CHARS or "..." in line or ". ." in line:
        return None
    if match := _MARKDOWN_HEADING.match(line):
        return match["title"]
    if line.endswith((".", ",", ";", ":")):
        return None  # a sentence, not a heading
    if match := _NUMBERED_HEADING.match(line):
        return f"{match['number'].replace(' ', '')} {match['title']}"
    if _NAMED_HEADING.match(line):
        return line
    return None


def _split_long_line(line: str, size: int, overlap: int) -> list[str]:
    """Cut one over-long line into token windows (v1's method)."""
    tokens = _ENCODING.encode(line)
    step = size - overlap
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        pieces.append(_ENCODING.decode(tokens[start : start + size]))
        if start + size >= len(tokens):
            break
    return pieces


def _to_units(text: str, size: int, overlap: int) -> list[str]:
    """Non-empty lines of `text`, with any line longer than `size` cut into pieces."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _token_len(line) > size:
            units.extend(_split_long_line(line, size, overlap))
        else:
            units.append(line)
    return units


def _pack(unit_lengths: list[int], size: int, overlap: int) -> list[tuple[int, int]]:
    """Group consecutive units into chunks; return (start, end) index ranges, end exclusive.

    A chunk holds as many whole units as fit in `size` tokens. The next chunk starts by
    repeating the previous chunk's trailing units, as many as fit in `overlap` tokens.
    Each unit costs its own length plus one token for the newline that joins it.
    """
    ranges: list[tuple[int, int]] = []
    start, used = 0, 0
    for i, length in enumerate(unit_lengths):
        cost = length + 1
        if i > start and used + cost > size:
            ranges.append((start, i))
            # Step back over trailing units while they fit in the overlap budget...
            new_start, carried = i, 0
            while new_start - 1 > start and carried + unit_lengths[new_start - 1] + 1 <= overlap:
                new_start -= 1
                carried += unit_lengths[new_start] + 1
            # ...then drop carried units if the current one still wouldn't fit.
            while carried + cost > size and new_start < i:
                carried -= unit_lengths[new_start] + 1
                new_start += 1
            start, used = new_start, carried
        used += cost
    if start < len(unit_lengths):
        ranges.append((start, len(unit_lengths)))
    return ranges


def _chunk_units(text: str, size: int, overlap: int) -> tuple[list[str], list[tuple[int, int]]]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    units = _to_units(text, size, overlap)
    return units, _pack([_token_len(u) for u in units], size, overlap)


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split `text` into chunks of at most `size` tokens, breaking only between lines."""
    units, ranges = _chunk_units(text, size, overlap)
    return ["\n".join(units[start:end]) for start, end in ranges]


def chunk_pages(
    pages: list[ParsedPage], doc_title: str, size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Chunk each page separately (a chunk never spans two pages, so citations stay exact).

    `section` is the heading in effect at a chunk's first line. Headings carry over to
    later pages until a new one appears.
    """
    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        units, ranges = _chunk_units(page.text, size, overlap)
        # Section in effect at each unit: the latest heading at or before it.
        section_at: list[str | None] = []
        for unit in units:
            section = detect_section(unit) or section
            section_at.append(section)
        for index, (start, end) in enumerate(ranges):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}::p{page.page}::c{index}",
                    doc_id=page.doc_id,
                    doc_title=doc_title,
                    page=page.page,
                    section=section_at[start],
                    parser=page.parser,
                    text="\n".join(units[start:end]),
                )
            )
    return chunks
