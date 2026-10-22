"""OCR with olmOCR 2 through any OpenAI-compatible endpoint.

The request mirrors the official olmocr pipeline (v0.4.27, `olmocr/pipeline.py` and
`olmocr/prompts/prompts.py`): the page rendered as a PNG with its longest side at
1288 px, the "v4 YAML" prompt, up to 8000 output tokens, and a temperature that rises
on retries. The model answers with YAML front matter (language, rotation, is_table...)
followed by the page as Markdown; tables come back as HTML.

Switching provider (Parasail, Cirrascale, a rented GPU running vLLM) only changes
`base_url`, `api_key` and `model`.
"""

import base64
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx2
import openai
import pypdfium2 as pdfium

PARSER_NAME = "olmocr"
TARGET_LONGEST_SIDE = 1288
MAX_TOKENS = 8000
# From olmocr: a higher temperature on retries helps the model escape repetition loops.
TEMPERATURE_BY_ATTEMPT = (0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)
# Verbatim from olmocr's build_no_anchoring_v4_yaml_prompt(), typo included: the model
# was trained on this exact text.
PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were reading it "
    "naturally. Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values for "
    "the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram "
    "parameters."
)


@dataclass(frozen=True)
class OcrResult:
    text: str
    input_tokens: int
    output_tokens: int


class OcrError(Exception):
    """One OCR attempt failed (HTTP error, timeout, truncated output). Worth retrying."""


class OcrFailedError(Exception):
    """OCR failed on every attempt. Stops ingestion with a clear message."""


class OcrClient(Protocol):
    """Anything that turns a page image into text. Real: OlmOcrClient. Tests: a fake."""

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult: ...


def strip_front_matter(content: str) -> str:
    """Drop olmOCR's leading YAML block ("---\\n...\\n---") and return the page text."""
    if content.startswith("---\n"):
        end = content.find("\n---", 4)
        if end != -1:
            return content[end + len("\n---") :].strip()
    return content.strip()


class OlmOcrClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
        http_client: httpx2.Client | None = None,  # tests pass a mock transport here
    ) -> None:
        self.model = model
        # max_retries=0: ocr_with_retries owns retrying (with a rising temperature),
        # otherwise every failure would be retried 3 × 3 times.
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",  # a local vLLM server accepts any key
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        temperature = TEMPERATURE_BY_ATTEMPT[min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
        except openai.APIStatusError as error:
            raise OcrError(f"HTTP {error.status_code}: {error.message}") from error
        except openai.APIError as error:  # connection errors and timeouts
            raise OcrError(f"{type(error).__name__}: {error}") from error

        choice = response.choices[0]
        if choice.finish_reason != "stop":
            # "length" means the page was cut off at MAX_TOKENS: don't keep half a page.
            raise OcrError(f"incomplete output (finish_reason={choice.finish_reason})")
        usage = response.usage
        return OcrResult(
            text=strip_front_matter(choice.message.content or ""),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


def ocr_with_retries(
    client: OcrClient,
    png: bytes,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    base_delay: float = 2.0,
) -> OcrResult:
    """Call the OCR client up to `max_attempts` times, waiting 2 s, 4 s, ... in between."""
    last_error: OcrError | None = None
    for attempt in range(max_attempts):
        try:
            return client.ocr_page(png, attempt)
        except OcrError as error:
            last_error = error
            if attempt < max_attempts - 1:
                sleep(base_delay * 2**attempt)
    raise OcrFailedError(f"OCR failed after {max_attempts} attempts: {last_error}")


def render_page_png(path: Path, page: int, longest_side: int = TARGET_LONGEST_SIDE) -> bytes:
    """Render one page (1-based) to PNG bytes, scaled so its longest side is `longest_side` px."""
    pdf = pdfium.PdfDocument(path)
    try:
        if not 1 <= page <= len(pdf):
            raise ValueError(f"{path.name} has {len(pdf)} pages, no page {page}")
        pdf_page = pdf[page - 1]
        scale = longest_side / max(pdf_page.get_size())  # PDF sizes are in points (1/72 in)
        image = pdf_page.render(scale=scale).to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        pdf.close()
