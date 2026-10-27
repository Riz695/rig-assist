"""Rendering a manual page to PNG (pypdfium2), for the UI's citation viewer."""

import io
from pathlib import Path

import pytest
from conftest import build_pdf
from PIL import Image

from rig_assist.ingestion.page_images import PNG_SIGNATURE, render_page_png

TWO_PAGES = build_pdf([["3.1 Mud Pumps", "Check the liners."], ["5.1 BOP", "Close the annular."]])


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "Pump_Manual.pdf"
    path.write_bytes(TWO_PAGES)
    return path


def test_a_page_renders_as_a_png_of_the_requested_width(pdf: Path) -> None:
    png = render_page_png(pdf, "sha-1", page=1, width=600)
    assert png.startswith(PNG_SIGNATURE)
    image = Image.open(io.BytesIO(png))
    assert image.width == 600
    # US Letter keeps its aspect ratio (PDFium rounds the height up: within a pixel)
    assert abs(image.height - 600 * 792 / 612) <= 1


def test_pages_are_1_based_and_distinct(pdf: Path) -> None:
    assert render_page_png(pdf, "sha-1", 1) != render_page_png(pdf, "sha-1", 2)


@pytest.mark.parametrize("page", [0, 3])
def test_a_page_outside_the_pdf_is_an_index_error(pdf: Path, page: int) -> None:
    with pytest.raises(IndexError, match="page"):
        render_page_png(pdf, "sha-1", page)


def test_the_same_file_version_is_rendered_once(pdf: Path) -> None:
    render_page_png.cache_clear()
    first = render_page_png(pdf, "sha-1", 1)
    pdf.write_bytes(b"not a pdf any more")  # a cache hit never reopens the file
    assert render_page_png(pdf, "sha-1", 1) == first
    assert render_page_png.cache_info().hits == 1
