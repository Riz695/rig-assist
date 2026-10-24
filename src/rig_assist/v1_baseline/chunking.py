"""v1's chunker (`src/rag/chunking.py`): fixed windows of cl100k tokens per page.

Unlike v2's line-aware chunker, windows can end mid-line or mid-word (a table row may be
split), and no section is detected. Kept exactly, because it is part of what's measured.
"""

import tiktoken

from rig_assist.ingestion.models import Chunk, ParsedPage

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    tokens = _ENCODING.encode(text.strip())
    if not tokens:
        return []
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        chunks.append(_ENCODING.decode(tokens[start : start + size]))
        if start + size >= len(tokens):
            break
    return chunks


def chunk_pages(
    pages: list[ParsedPage], doc_title: str, size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Chunk each page on its own; empty pages give no chunks."""
    return [
        Chunk(
            chunk_id=f"{page.doc_id}::p{page.page}::c{index}",
            doc_id=page.doc_id,
            doc_title=doc_title,
            page=page.page,
            section=None,
            parser=page.parser,
            text=piece,
        )
        for page in pages
        for index, piece in enumerate(chunk_text(page.text, size, overlap))
    ]
