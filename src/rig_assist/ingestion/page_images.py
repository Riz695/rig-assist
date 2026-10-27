"""Render one page of a manual to PNG, so a citation can show the real page.

pypdfium2 (Google's PDFium, already used by pdfplumber) rasterises the page at a fixed
width. Rendered pages are cached in memory by (file, file SHA-256, page, width): the hash
is part of the key, so re-uploading a changed manual never serves an old image.

PDFium is not thread-safe, and FastAPI runs sync routes in a thread pool, so rendering
holds a lock. A cache hit doesn't take the lock.
"""

import io
import threading
from functools import lru_cache
from pathlib import Path

import pypdfium2 as pdfium

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PAGE_IMAGE_WIDTH = 1000  # px: readable on a laptop screen, ~100-300 KB per page

_pdfium_lock = threading.Lock()


@lru_cache(maxsize=32)  # ~32 pages x ~200 KB: a few MB
def render_page_png(
    pdf_path: Path, file_sha256: str, page: int, width: int = PAGE_IMAGE_WIDTH
) -> bytes:
    """PNG bytes of 1-based `page`, `width` pixels wide. `file_sha256` is only a cache key
    (the file's version). Raises IndexError when the page isn't in the PDF."""
    with _pdfium_lock:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            if not 1 <= page <= len(pdf):
                raise IndexError(f"page {page} is not in the PDF ({len(pdf)} pages)")
            pdf_page = pdf[page - 1]
            image = pdf_page.render(scale=width / pdf_page.get_width()).to_pil()
        finally:
            pdf.close()
    if image.width != width:  # scale rounds; make the width exact
        image = image.resize((width, round(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
