"""Parse a PDF into pages, choosing per page between the text layer and OCR.

OCR modes (`OCR_MODE`):
- "off":     text layer only. Pages without text are listed in `missing_pages`.
- "missing": OCR only the pages without a text layer (scans). The normal setting.
- "all":     OCR every page, for the pdfplumber vs olmOCR comparison (Phase 3).
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.cleaning import strip_repeated_lines
from rig_assist.ingestion.models import ParsedPage
from rig_assist.ingestion.parsers import olmocr
from rig_assist.ingestion.parsers.olmocr import OcrClient, OcrFailedError
from rig_assist.ingestion.parsers.text_layer import extract_text_layer, has_text_layer

logger = logging.getLogger(__name__)

OcrMode = Literal["off", "missing", "all"]


@dataclass(frozen=True)
class ParseResult:
    pages: list[ParsedPage]  # pages with text, in page order
    missing_pages: list[int]  # pages left without text: reported, never silently dropped
    ocr_pages: int  # pages sent to the OCR API in this run
    ocr_cached_pages: int  # pages whose OCR came from the cache
    input_tokens: int  # OCR usage in this run (cached pages cost nothing)
    output_tokens: int


def parse_document(
    path: Path,
    doc_id: str,
    ocr_mode: OcrMode = "off",
    ocr_client: OcrClient | None = None,
    cache: PageCache | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> ParseResult:
    if ocr_mode != "off" and ocr_client is None:
        raise ValueError(f"ocr_mode={ocr_mode!r} needs an OCR client")

    file_hash: str | None = None  # computed only if a page needs OCR
    pages: list[ParsedPage] = []
    missing: list[int] = []
    ocr_pages = cached_pages = input_tokens = output_tokens = 0

    for text_page in extract_text_layer(path, doc_id):
        needs_ocr = ocr_mode == "all" or (
            ocr_mode == "missing" and not has_text_layer(text_page.text)
        )
        if not needs_ocr:
            if has_text_layer(text_page.text):
                pages.append(text_page)
            else:
                missing.append(text_page.page)
            continue

        assert ocr_client is not None  # checked above; narrows the type for mypy
        if file_hash is None:
            file_hash = file_sha256(path)
        result = cache.get(file_hash, text_page.page, olmocr.PARSER_NAME) if cache else None
        if result is not None:
            cached_pages += 1
        else:
            logger.info("OCR %s page %d", path.name, text_page.page)
            png = olmocr.render_page_png(path, text_page.page)
            try:
                result = olmocr.ocr_with_retries(ocr_client, png, max_attempts, sleep)
            except OcrFailedError as error:
                raise OcrFailedError(f"{path.name} page {text_page.page}: {error}") from error
            if cache:
                cache.put(file_hash, text_page.page, olmocr.PARSER_NAME, result)
            ocr_pages += 1
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens

        if has_text_layer(result.text):
            pages.append(ParsedPage(doc_id, text_page.page, result.text, parser=olmocr.PARSER_NAME))
        else:
            missing.append(text_page.page)

    if missing:
        logger.warning("%s: no text on pages %s", path.name, missing)
    return ParseResult(
        pages=strip_repeated_lines(pages),
        missing_pages=missing,
        ocr_pages=ocr_pages,
        ocr_cached_pages=cached_pages,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
