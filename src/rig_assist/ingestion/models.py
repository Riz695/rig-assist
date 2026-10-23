"""Data passed between ingestion steps: parser → ParsedPage → chunker → Chunk → indexes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPage:
    """The text of one PDF page, as produced by one parser."""

    doc_id: str
    page: int  # 1-based, as printed in PDF viewers and citations
    text: str
    parser: str  # e.g. "pdfplumber" or "olmocr"


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of one page, with the metadata needed to cite it."""

    chunk_id: str  # "<doc_id>::p<page>::c<index>", stable across re-ingestion
    doc_id: str
    doc_title: str
    page: int
    section: str | None  # nearest heading above the chunk, when one was detected
    parser: str
    text: str
