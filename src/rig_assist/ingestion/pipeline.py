"""Ingestion: PDF → pages (text layer / OCR) → chunks → index store.

Used by `make ingest` (every PDF in DATA_RAW_DIR) and by the upload endpoint (one file).
"""

import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.chunking import chunk_pages
from rig_assist.ingestion.parsers.olmocr import OcrClient
from rig_assist.ingestion.parsing import OcrMode, parse_document
from rig_assist.ingestion.store import DocumentInfo, IndexStore

logger = logging.getLogger(__name__)


def make_doc_id(filename: str) -> str:
    """'Offshore manual.pdf' → 'offshore-manual'. Stable, URL- and file-name-safe."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    if not slug:
        raise ValueError(f"can't make a doc_id from {filename!r}: no letters or digits")
    return slug


def make_title(filename: str) -> str:
    """'Drilling_Manual.pdf' → 'Drilling Manual'. PDF metadata titles are often empty or junk."""
    return " ".join(re.sub(r"[_-]+", " ", Path(filename).stem).split())


def ingest_file(
    path: Path,
    store: IndexStore,
    ocr_mode: OcrMode = "off",
    ocr_client: OcrClient | None = None,
    cache: PageCache | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    sleep: Callable[[float], None] = time.sleep,
) -> DocumentInfo:
    """Parse, chunk and store one PDF, replacing any earlier version with the same doc_id."""
    doc_id, title = make_doc_id(path.name), make_title(path.name)
    result = parse_document(path, doc_id, ocr_mode, ocr_client, cache, sleep=sleep)
    chunks = chunk_pages(result.pages, doc_title=title, size=chunk_size, overlap=chunk_overlap)
    info = DocumentInfo(
        doc_id=doc_id,
        title=title,
        filename=path.name,
        file_sha256=file_sha256(path),
        pages=len(result.pages) + len(result.missing_pages),
        chunks=len(chunks),
        missing_pages=result.missing_pages,
    )
    store.put(info, chunks)
    if result.ocr_pages or result.ocr_cached_pages:
        logger.info(
            "%s: OCR %d new pages (%d in / %d out tokens), %d from cache",
            path.name,
            result.ocr_pages,
            result.input_tokens,
            result.output_tokens,
            result.ocr_cached_pages,
        )
    return info


def ingest_directory(
    raw_dir: Path,
    store: IndexStore,
    ocr_mode: OcrMode = "off",
    ocr_client: OcrClient | None = None,
    cache: PageCache | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    sleep: Callable[[float], None] = time.sleep,
) -> list[DocumentInfo]:
    """Ingest every *.pdf in `raw_dir` (not subfolders). Returns them sorted by doc_id."""
    pdfs = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    # Check every name first: failing on the 6th file after 5 were written would leave
    # a half-updated index.
    ids = {path: make_doc_id(path.name) for path in pdfs}
    clashes = sorted(doc_id for doc_id, n in Counter(ids.values()).items() if n > 1)
    if clashes:
        raise ValueError(f"several files map to the same doc_id: {', '.join(clashes)}")

    infos = []
    for path in sorted(pdfs, key=ids.__getitem__):
        logger.info("Ingesting %s", path.name)
        infos.append(
            ingest_file(path, store, ocr_mode, ocr_client, cache, chunk_size, chunk_overlap, sleep)
        )
    return infos
