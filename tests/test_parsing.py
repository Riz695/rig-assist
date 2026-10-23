from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import FakeOcr, no_sleep

from rig_assist.ingestion.cache import PageCache
from rig_assist.ingestion.parsers.olmocr import OcrError, OcrFailedError, OcrResult
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
