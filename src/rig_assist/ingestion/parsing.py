"""Parse a PDF into pages, choosing per page between the text layer and OCR.

OCR modes (`OCR_MODE`):
- "off":     text layer only. Pages without text are listed in `missing_pages`.
- "missing": OCR only the pages without a text layer (scans). The normal setting.
- "all":     OCR every page, for the pdfplumber vs olmOCR comparison (Phase 3).
"""

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.cleaning import collapse_dot_leaders, strip_repeated_lines
from rig_assist.ingestion.models import ParsedPage
from rig_assist.ingestion.parsers import olmocr
from rig_assist.ingestion.parsers.olmocr import (
    OcrClient,
    OcrFailedError,
    OcrPageUnreadableError,
    OcrResult,
)
from rig_assist.ingestion.parsers.text_layer import extract_text_layer, has_text_layer

logger = logging.getLogger(__name__)

OcrMode = Literal["off", "missing", "all"]

# PDFium (the renderer) is not thread-safe, so pages are rendered one at a time.
# Rendering takes ~0.1 s against seconds for OCR, so the lock costs almost nothing.
_render_lock = threading.Lock()


@dataclass(frozen=True)
class ParseResult:
    pages: list[ParsedPage]  # pages with text, in page order
    missing_pages: list[int]  # pages left without text: reported, never silently dropped
    ocr_pages: int  # pages sent to the OCR API in this run
    ocr_cached_pages: int  # pages whose OCR came from the cache
    input_tokens: int  # OCR usage in this run (cached pages cost nothing)
    output_tokens: int


def _ocr_page(
    path: Path,
    page: int,
    client: OcrClient,
    cache: PageCache | None,
    file_hash: str,
    max_attempts: int,
    sleep: Callable[[float], None],
) -> OcrResult | None:
    """Render, OCR (with retries) and cache one page. Runs in a worker thread.

    Returns None if the model could never finish the page (reported as missing).
    """
    logger.info("OCR %s page %d", path.name, page)
    with _render_lock:
        png = olmocr.render_page_png(path, page)
    try:
        result = olmocr.ocr_with_retries(client, png, max_attempts, sleep)
    except OcrPageUnreadableError as error:
        logger.warning("%s page %d: %s; reported as missing", path.name, page, error)
        return None
    except OcrFailedError as error:
        raise OcrFailedError(f"{path.name} page {page}: {error}") from error
    if cache:
        cache.put(file_hash, page, olmocr.PARSER_NAME, result)  # one file per page: thread-safe
    return result


def _ocr_pages(
    path: Path,
    pages: list[int],
    client: OcrClient,
    cache: PageCache | None,
    file_hash: str,
    max_attempts: int,
    sleep: Callable[[float], None],
    concurrency: int,
) -> dict[int, OcrResult]:
    """OCR `pages` with up to `concurrency` requests in flight. Stops at the first failure."""
    results: dict[int, OcrResult] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_ocr_page, path, page, client, cache, file_hash, max_attempts, sleep): page
            for page in pages
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:  # None: unreadable page, left out (reported missing)
                    results[futures[future]] = result
        except BaseException:
            # Don't start queued pages; those already running finish and are cached.
            pool.shutdown(cancel_futures=True)
            raise
    return results


def parse_document(
    path: Path,
    doc_id: str,
    ocr_mode: OcrMode = "off",
    ocr_client: OcrClient | None = None,
    cache: PageCache | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    ocr_concurrency: int = 1,
) -> ParseResult:
    if ocr_mode != "off" and ocr_client is None:
        raise ValueError(f"ocr_mode={ocr_mode!r} needs an OCR client")
    if ocr_concurrency < 1:
        raise ValueError("ocr_concurrency must be at least 1")

    text_pages = extract_text_layer(path, doc_id)

    # 1. Which pages need OCR, and which of those are already cached.
    needs_ocr = [
        p.page
        for p in text_pages
        if ocr_mode == "all" or (ocr_mode == "missing" and not has_text_layer(p.text))
    ]
    ocr_results: dict[int, OcrResult] = {}
    new_results: dict[int, OcrResult] = {}
    if needs_ocr:
        assert ocr_client is not None  # checked above; narrows the type for mypy
        file_hash = file_sha256(path)
        to_run = []
        for page in needs_ocr:
            cached = cache.get(file_hash, page, olmocr.PARSER_NAME) if cache else None
            if cached is not None:
                ocr_results[page] = cached
            else:
                to_run.append(page)
        # 2. OCR the rest, several pages at a time.
        new_results = _ocr_pages(
            path, to_run, ocr_client, cache, file_hash, max_attempts, sleep, ocr_concurrency
        )
        ocr_results.update(new_results)

    # 3. Assemble in page order.
    pages: list[ParsedPage] = []
    missing: list[int] = []
    for text_page in text_pages:
        if text_page.page in ocr_results:
            text = ocr_results[text_page.page].text
            if has_text_layer(text):
                pages.append(ParsedPage(doc_id, text_page.page, text, parser=olmocr.PARSER_NAME))
            else:
                missing.append(text_page.page)
        elif has_text_layer(text_page.text):
            pages.append(text_page)
        else:
            missing.append(text_page.page)

    if missing:
        logger.warning("%s: no text on pages %s", path.name, missing)
    cleaned = [replace(page, text=collapse_dot_leaders(page.text)) for page in pages]
    return ParseResult(
        pages=strip_repeated_lines(cleaned),
        missing_pages=missing,
        ocr_pages=len(new_results),
        ocr_cached_pages=len(ocr_results) - len(new_results),
        input_tokens=sum(r.input_tokens for r in new_results.values()),
        output_tokens=sum(r.output_tokens for r in new_results.values()),
    )
