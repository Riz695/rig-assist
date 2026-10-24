import contextlib
import hashlib
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import FakeOcr, no_sleep

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.parsers.olmocr import (
    OcrError,
    OcrFailedError,
    OcrIncompleteError,
    OcrResult,
    render_page_png,
)
from rig_assist.ingestion.parsing import parse_document

MakePdf = Callable[[list[list[str]]], Path]

TEXT_PAGE = ["Mud pump liner wear is checked weekly"]
SCANNED_PAGE: list[str] = []


@pytest.fixture
def mixed_pdf(make_pdf: MakePdf) -> Path:
    """Page 1 has text, page 2 is 'scanned', page 3 has text."""
    return make_pdf([TEXT_PAGE, SCANNED_PAGE, ["Annular preventer closing pressure"]])


def test_ocr_off_reports_pages_without_text_instead_of_dropping_them_silently(
    mixed_pdf: Path,
) -> None:
    result = parse_document(mixed_pdf, doc_id="d", ocr_mode="off")

    assert [p.page for p in result.pages] == [1, 3]
    assert result.missing_pages == [2]
    assert {p.parser for p in result.pages} == {"pdfplumber"}


def test_ocr_missing_sends_only_pages_without_text_to_ocr(mixed_pdf: Path) -> None:
    ocr = FakeOcr()
    result = parse_document(mixed_pdf, "d", ocr_mode="missing", ocr_client=ocr, sleep=no_sleep)

    assert ocr.calls == 1
    assert [(p.page, p.parser) for p in result.pages] == [
        (1, "pdfplumber"),
        (2, "olmocr"),
        (3, "pdfplumber"),
    ]
    assert result.pages[1].text.startswith("Text read from the page image")
    assert result.missing_pages == []


def test_ocr_all_sends_every_page_to_ocr(mixed_pdf: Path) -> None:
    ocr = FakeOcr()
    result = parse_document(mixed_pdf, "d", ocr_mode="all", ocr_client=ocr, sleep=no_sleep)

    assert ocr.calls == 3
    assert [p.page for p in result.pages] == [1, 2, 3]
    assert {p.parser for p in result.pages} == {"olmocr"}


def test_pages_keep_doc_id_and_page_numbers(mixed_pdf: Path) -> None:
    result = parse_document(mixed_pdf, "drilling-manual", "missing", FakeOcr(), sleep=no_sleep)
    assert [(p.doc_id, p.page) for p in result.pages] == [
        ("drilling-manual", 1),
        ("drilling-manual", 2),
        ("drilling-manual", 3),
    ]


def test_cached_pages_are_not_sent_to_ocr_again(mixed_pdf: Path, tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "parsed")
    first_ocr, second_ocr = FakeOcr(), FakeOcr()

    first = parse_document(mixed_pdf, "d", "all", first_ocr, cache=cache, sleep=no_sleep)
    second = parse_document(mixed_pdf, "d", "all", second_ocr, cache=cache, sleep=no_sleep)

    assert first_ocr.calls == 3
    assert second_ocr.calls == 0
    assert second.pages == first.pages


def test_usage_counts_only_new_ocr_calls(mixed_pdf: Path, tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "parsed")
    first = parse_document(mixed_pdf, "d", "all", FakeOcr(), cache=cache, sleep=no_sleep)
    second = parse_document(mixed_pdf, "d", "all", FakeOcr(), cache=cache, sleep=no_sleep)

    assert (first.ocr_pages, first.ocr_cached_pages) == (3, 0)
    assert (first.input_tokens, first.output_tokens) == (6000, 1500)
    assert (second.ocr_pages, second.ocr_cached_pages) == (0, 3)
    assert (second.input_tokens, second.output_tokens) == (0, 0)


def test_ocr_failure_names_the_file_and_page(mixed_pdf: Path) -> None:
    with pytest.raises(OcrFailedError, match=r"manual\.pdf page 2.*503"):
        parse_document(mixed_pdf, "d", "missing", FakeOcr(fail=True), sleep=no_sleep)


def test_pages_ocrd_before_a_failure_stay_cached(make_pdf: MakePdf, tmp_path: Path) -> None:
    path = make_pdf([SCANNED_PAGE, SCANNED_PAGE])
    cache = PageCache(tmp_path / "parsed")

    class FailsOnSecondPage(FakeOcr):
        def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
            if self.calls >= 1:
                raise OcrError("timeout")
            return super().ocr_page(png, attempt)

    with pytest.raises(OcrFailedError):
        parse_document(path, "d", "missing", FailsOnSecondPage(), cache=cache, sleep=no_sleep)

    retry = FakeOcr()
    parse_document(path, "d", "missing", retry, cache=cache, sleep=no_sleep)
    assert retry.calls == 1, "page 1 came from the cache, only page 2 was OCR'd"


def test_page_where_ocr_finds_no_text_is_reported_missing(make_pdf: MakePdf) -> None:
    class BlankOcr(FakeOcr):
        def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
            return OcrResult(text="", input_tokens=2000, output_tokens=10)

    result = parse_document(make_pdf([TEXT_PAGE, SCANNED_PAGE]), "d", "missing", BlankOcr())
    assert [p.page for p in result.pages] == [1]
    assert result.missing_pages == [2]


def test_ocr_mode_without_a_client_is_rejected(mixed_pdf: Path) -> None:
    with pytest.raises(ValueError, match="OCR client"):
        parse_document(mixed_pdf, "d", ocr_mode="missing", ocr_client=None)


def test_running_headers_are_stripped(make_pdf: MakePdf) -> None:
    words = ["anchor", "winch", "mooring", "thruster", "ballast", "riser"]
    pages = [["Marine Operations Manual UKCS-MAL-001", f"Section about the {w}"] for w in words]
    result = parse_document(make_pdf(pages), "d", ocr_mode="off")
    assert all("UKCS-MAL-001" not in p.text for p in result.pages)


# --- concurrency ------------------------------------------------------------


class PageEchoOcr:
    """Answer depends only on the image, so results can't depend on thread timing.

    Records how many calls were in flight at once. With `meet=2`, the first two calls
    wait for each other at a barrier: that only succeeds if they run at the same time,
    which proves concurrency without relying on sleeps.
    """

    def __init__(self, meet: int = 0) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._calls = 0
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(meet) if meet else None

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        with self._lock:
            self._calls += 1
            first_calls = self._barrier is not None and self._calls <= self._barrier.parties
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if first_calls and self._barrier is not None:
            # Both calls must be in flight to pass. If they ran one at a time, the wait
            # times out, max_in_flight stays 1 and the test fails.
            with contextlib.suppress(threading.BrokenBarrierError):
                self._barrier.wait(timeout=5)
        with self._lock:
            self.in_flight -= 1
        digest = hashlib.sha256(png).hexdigest()[:12]
        return OcrResult(text=f"Text of page image {digest}", input_tokens=10, output_tokens=5)


@pytest.fixture
def four_pages(make_pdf: MakePdf) -> Path:
    return make_pdf([[f"Page about the {w} system"] for w in ("mud", "bop", "riser", "crane")])


def test_concurrent_ocr_gives_the_same_pages_in_the_same_order(four_pages: Path) -> None:
    one = parse_document(four_pages, "d", "all", PageEchoOcr(), ocr_concurrency=1)
    four = parse_document(four_pages, "d", "all", PageEchoOcr(), ocr_concurrency=4)

    assert four.pages == one.pages
    assert [p.page for p in four.pages] == [1, 2, 3, 4]
    assert len({p.text for p in four.pages}) == 4, "each page got its own OCR text"
    assert (four.ocr_pages, four.input_tokens) == (4, 40)


def test_concurrency_sends_several_pages_at_once(four_pages: Path) -> None:
    parallel = PageEchoOcr(meet=2)
    parse_document(four_pages, "d", "all", parallel, ocr_concurrency=4)
    assert parallel.max_in_flight >= 2


def test_concurrency_one_sends_one_page_at_a_time(four_pages: Path) -> None:
    sequential = PageEchoOcr()
    parse_document(four_pages, "d", "all", sequential, ocr_concurrency=1)
    assert sequential.max_in_flight == 1


def test_concurrent_ocr_failure_still_names_the_page(make_pdf: MakePdf) -> None:
    path = make_pdf([SCANNED_PAGE, SCANNED_PAGE, SCANNED_PAGE])
    with pytest.raises(OcrFailedError, match=r"manual\.pdf page \d.*503"):
        parse_document(path, "d", "missing", FakeOcr(fail=True), sleep=no_sleep, ocr_concurrency=3)


def test_concurrent_ocr_caches_every_page(four_pages: Path, tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "parsed")
    parse_document(four_pages, "d", "all", PageEchoOcr(), cache=cache, ocr_concurrency=4)

    again = FakeOcr()
    parse_document(four_pages, "d", "all", again, cache=cache, ocr_concurrency=4)
    assert again.calls == 0


def test_concurrency_below_one_is_rejected(four_pages: Path) -> None:
    with pytest.raises(ValueError, match="ocr_concurrency"):
        parse_document(four_pages, "d", "all", PageEchoOcr(), ocr_concurrency=0)


def test_a_page_ocr_cannot_finish_is_reported_missing_and_the_rest_continue(
    make_pdf: MakePdf, caplog: pytest.LogCaptureFixture
) -> None:
    """One looping page (e.g. a blank form) must not stop a 146-page run."""
    # Under 10 letters/digits: no usable text layer, but its image differs from a blank page.
    blank_form = ["FORM"]

    class LoopsOnBlankForm(FakeOcr):
        def __init__(self) -> None:
            super().__init__()
            self.form_png: bytes | None = None

        def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
            if png == self.form_png:
                raise OcrIncompleteError("incomplete output (finish_reason=length)")
            return super().ocr_page(png, attempt)

    path = make_pdf([SCANNED_PAGE, blank_form, SCANNED_PAGE])
    ocr = LoopsOnBlankForm()
    ocr.form_png = render_page_png(path, 2)

    result = parse_document(path, "d", "missing", ocr, sleep=no_sleep)

    assert [p.page for p in result.pages] == [1, 3]
    assert result.missing_pages == [2]
    assert "page 2" in caplog.text and "length" in caplog.text


def test_unreadable_page_falls_back_to_its_text_layer_when_it_has_one(make_pdf: MakePdf) -> None:
    class AlwaysLoops(FakeOcr):
        def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
            raise OcrIncompleteError("incomplete output (finish_reason=length)")

    result = parse_document(make_pdf([TEXT_PAGE]), "d", "all", AlwaysLoops(), sleep=no_sleep)

    assert [(p.page, p.parser) for p in result.pages] == [(1, "pdfplumber")]
    assert result.missing_pages == []


def test_parsed_pages_have_dot_leaders_collapsed(make_pdf: MakePdf) -> None:
    toc = ["Table of Contents", "1 INTRODUCTION......................7", "2 DRILLING........12"]
    result = parse_document(make_pdf([toc]), "d", ocr_mode="off")
    assert "1 INTRODUCTION … 7" in result.pages[0].text
    assert "...." not in result.pages[0].text


def test_ocr_html_tables_become_markdown_but_the_cache_keeps_the_raw_output(
    make_pdf: MakePdf, tmp_path: Path
) -> None:
    table = "<table><tr><th>Pump</th><th>Bar</th></tr><tr><td>P-101</td><td>350</td></tr></table>"
    cache = PageCache(tmp_path / "parsed")
    path = make_pdf([SCANNED_PAGE])

    result = parse_document(path, "d", "missing", FakeOcr(text=table), cache=cache, sleep=no_sleep)

    assert "| Pump | Bar |\n| --- | --- |\n| P-101 | 350 |" in result.pages[0].text
    assert "<table>" not in result.pages[0].text
    cached = cache.get(file_sha256(path), 1, "olmocr")
    assert cached is not None and "<table>" in cached.text
