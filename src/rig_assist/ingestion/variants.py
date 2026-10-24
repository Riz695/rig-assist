"""Index variants: the same manuals indexed another way, for the Phase 3 parser comparison.

`olmocr_all` uses olmOCR's text for every page, including pages that have a text layer
(the main index uses pdfplumber there and olmOCR only for pages without one). Each
variant has its own chunk store (`<DATA_INDEX_DIR>/variants/<variant>/`) and Qdrant
collection (`<QDRANT_COLLECTION>__<variant>`), so it never touches the main index.

Variants are built from the OCR cache only: `CacheOnlyOcrClient` stops the build at the
first uncached page instead of calling (and paying for) the OCR server.
"""

from pathlib import Path
from typing import Literal, get_args

from rig_assist.ingestion.parsers.olmocr import OcrFailedError, OcrResult
from rig_assist.ingestion.parsing import OcrMode

IndexVariant = Literal["olmocr_all"]
INDEX_VARIANTS: tuple[IndexVariant, ...] = get_args(IndexVariant)
VARIANT_OCR_MODE: dict[IndexVariant, OcrMode] = {"olmocr_all": "all"}


def variant_store_dir(index_dir: Path, variant: IndexVariant) -> Path:
    return index_dir / "variants" / variant


def variant_collection(base_collection: str, variant: IndexVariant) -> str:
    return f"{base_collection}__{variant}"


class CacheOnlyOcrClient:
    """An OCR client that is only asked for pages missing from the cache, and refuses."""

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        # Not an OcrError, so it isn't retried.
        raise OcrFailedError(
            "not in the OCR cache; variant builds never call the OCR server "
            "(run `make ingest` with OCR_MODE=all first)"
        )
