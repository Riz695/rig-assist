"""Shared test helpers."""

import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from rig_assist.ingestion.parsers.olmocr import OcrError, OcrResult

RAW_MANUALS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def build_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal valid PDF: one page per entry, one text line per string.

    An empty list gives a page with no text layer, like a scanned page.
    Written by hand so tests need no PDF-writing library and no binary fixtures.
    """
    n = len(pages)
    font_id = 3
    page_ids = [4 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % p for p in page_ids), n),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for page_id, lines in zip(page_ids, pages, strict=True):
        ops = [b"BT /F1 12 Tf 72 720 Td 14 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(b"(%s) Tj T*" % escaped.encode("latin-1"))
        ops.append(b"ET")
        stream = b"\n".join(ops) if lines else b""
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (font_id, page_id + 1)
        )
        objects[page_id + 1] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (obj_id, objects[obj_id])
    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for obj_id in range(1, size):
        out += b"%010d 00000 n \n" % offsets[obj_id]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at)
    return bytes(out)


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[[list[list[str]]], Path]:
    """Write a generated PDF to a temp file and return its path."""

    def _make(pages: list[list[str]], name: str = "manual.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(build_pdf(pages))
        return path

    return _make


class FakeOcr:
    """Stands in for the OCR API: no network, deterministic, records every call."""

    def __init__(self, text: str = "Text read from the page image by OCR", fail: bool = False):
        self.text = text
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()  # pages may be OCR'd from several threads

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        assert png.startswith(b"\x89PNG"), "OCR must receive a rendered PNG"
        with self._lock:
            self.calls += 1
            call = self.calls
        if self.fail:
            raise OcrError("503 service unavailable")
        return OcrResult(text=f"{self.text} ({call})", input_tokens=2000, output_tokens=500)


def no_sleep(_: float) -> None:
    pass


class KeywordEmbedder:
    """Fake embedder: one dimension per keyword, so similarity is predictable.

    "Check the mud pump" → mostly the 'pump' and 'mud' directions. A query about pumps is
    always nearest to the pump chunk, whatever a real model would do.
    """

    KEYWORDS = ("pump", "bop", "riser", "crane", "mud", "kick")

    def __init__(self, query_prefix: str = "") -> None:
        self.query_prefix = query_prefix
        self.passage_calls = 0
        self.queries: list[str] = []

    @property
    def dim(self) -> int:
        return len(self.KEYWORDS) + 1

    def _vector(self, text: str) -> list[float]:
        words = text.lower()
        vector = [float(words.count(keyword)) for keyword in self.KEYWORDS]
        vector.append(0.1)  # never all zeros (cosine is undefined for a zero vector)
        norm = sum(x * x for x in vector) ** 0.5
        return [x / norm for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(self.query_prefix + text)
