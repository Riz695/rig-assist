import base64
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from openai import DefaultHttpx2Client
from PIL import Image

from rig_assist.ingestion.parsers.olmocr import (
    MAX_TOKENS,
    PROMPT,
    OcrError,
    OcrFailedError,
    OcrIncompleteError,
    OcrPageUnreadableError,
    OcrResult,
    OlmOcrClient,
    ocr_with_retries,
    render_page_png,
    strip_front_matter,
)

MakePdf = Callable[[list[list[str]]], Path]

FRONT_MATTER = (
    "---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\n"
    "is_table: False\nis_diagram: False\n---\n"
)


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    """A chat-completions response body, as an OpenAI-compatible server returns it."""
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "allenai/olmOCR-2-7B-1025",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1900, "completion_tokens": 600, "total_tokens": 2500},
    }


def make_client(
    respond: Callable[[httpx2.Request], httpx2.Response],
) -> tuple[OlmOcrClient, list[dict[str, Any]]]:
    """A real OlmOcrClient whose HTTP calls go to `respond` instead of the network."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return respond(request)

    client = OlmOcrClient(
        base_url="https://ocr.example/v1",
        api_key="test-key",
        model="allenai/olmOCR-2-7B-1025",
        http_client=DefaultHttpx2Client(transport=httpx2.MockTransport(handler)),
    )
    return client, sent


PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


# --- strip_front_matter -------------------------------------------------


def test_strip_front_matter_returns_the_page_text() -> None:
    assert strip_front_matter(FRONT_MATTER + "# Mooring\nCheck anchor chains.") == (
        "# Mooring\nCheck anchor chains."
    )


def test_strip_front_matter_keeps_text_without_front_matter() -> None:
    assert strip_front_matter("Plain page text\n") == "Plain page text"


def test_strip_front_matter_of_a_blank_page_is_empty() -> None:
    assert strip_front_matter(FRONT_MATTER) == ""


# --- OlmOcrClient -------------------------------------------------------


def test_client_sends_the_olmocr_request_format() -> None:
    client, sent = make_client(lambda _: httpx2.Response(200, json=completion(FRONT_MATTER + "x")))
    client.ocr_page(PNG, attempt=0)

    body = sent[0]
    assert body["model"] == "allenai/olmOCR-2-7B-1025"
    assert body["max_tokens"] == MAX_TOKENS
    assert body["temperature"] == pytest.approx(0.1)
    text_part, image_part = body["messages"][0]["content"]
    assert text_part == {"type": "text", "text": PROMPT}
    assert image_part["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(
        PNG
    ).decode("ascii")


def test_client_raises_temperature_on_later_attempts() -> None:
    client, sent = make_client(lambda _: httpx2.Response(200, json=completion("x")))
    client.ocr_page(PNG, attempt=0)
    client.ocr_page(PNG, attempt=3)
    assert sent[1]["temperature"] > sent[0]["temperature"]


def test_client_returns_text_and_token_usage() -> None:
    client, _ = make_client(
        lambda _: httpx2.Response(200, json=completion(FRONT_MATTER + "Anchor handling"))
    )
    assert client.ocr_page(PNG, attempt=0) == OcrResult(
        text="Anchor handling", input_tokens=1900, output_tokens=600
    )


def test_client_turns_http_errors_into_ocr_error() -> None:
    client, sent = make_client(lambda _: httpx2.Response(503, json={"error": {"message": "busy"}}))
    with pytest.raises(OcrError, match="503"):
        client.ocr_page(PNG, attempt=0)
    assert len(sent) == 1, "the SDK's own retries must be off: retrying is ocr_with_retries' job"


def test_client_rejects_output_cut_off_by_the_token_limit() -> None:
    client, _ = make_client(lambda _: httpx2.Response(200, json=completion("partial", "length")))
    with pytest.raises(OcrIncompleteError, match="length"):
        client.ocr_page(PNG, attempt=0)


def test_incomplete_output_is_still_an_ocr_error_so_it_is_retried() -> None:
    assert issubclass(OcrIncompleteError, OcrError)


# --- ocr_with_retries ---------------------------------------------------


class FlakyClient:
    """Fails `failures` times, then succeeds. Records the attempt numbers it was given."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts: list[int] = []

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        self.attempts.append(attempt)
        if len(self.attempts) <= self.failures:
            raise OcrError(f"server busy (call {len(self.attempts)})")
        return OcrResult(text="ok", input_tokens=1, output_tokens=1)


def test_retries_until_success_and_passes_the_attempt_number() -> None:
    client = FlakyClient(failures=2)
    delays: list[float] = []

    result = ocr_with_retries(client, PNG, max_attempts=3, sleep=delays.append)

    assert result.text == "ok"
    assert client.attempts == [0, 1, 2]
    assert len(delays) == 2
    assert delays[1] > delays[0], "backoff should grow"


def test_gives_up_after_max_attempts_with_a_clear_error() -> None:
    client = FlakyClient(failures=10)
    with pytest.raises(OcrFailedError, match=r"3 attempts.*call 3"):
        ocr_with_retries(client, PNG, max_attempts=3, sleep=lambda _: None)
    assert len(client.attempts) == 3


def test_does_not_sleep_after_the_last_attempt() -> None:
    delays: list[float] = []
    with pytest.raises(OcrFailedError):
        ocr_with_retries(FlakyClient(failures=10), PNG, max_attempts=2, sleep=delays.append)
    assert len(delays) == 1


# --- render_page_png ----------------------------------------------------


def test_render_page_png_scales_the_longest_side(make_pdf: MakePdf) -> None:
    path = make_pdf([["first page"], ["second page"]])
    png = render_page_png(path, page=2, longest_side=1288)

    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG"
    assert max(image.size) == 1288


def test_render_page_png_rejects_a_page_out_of_range(make_pdf: MakePdf) -> None:
    with pytest.raises(ValueError, match="page 5"):
        render_page_png(make_pdf([["only page"]]), page=5)


class ScriptedClient:
    """Raises the given errors in order, then succeeds."""

    def __init__(self, errors: list[OcrError]) -> None:
        self.errors = errors
        self.calls = 0

    def ocr_page(self, png: bytes, attempt: int) -> OcrResult:
        self.calls += 1
        if self.calls <= len(self.errors):
            raise self.errors[self.calls - 1]
        return OcrResult(text="ok", input_tokens=1, output_tokens=1)


def test_a_page_that_never_completes_is_reported_as_unreadable() -> None:
    """e.g. a blank form where the model repeats empty table rows until the token limit."""
    client = ScriptedClient([OcrIncompleteError("finish_reason=length")] * 3)
    with pytest.raises(OcrPageUnreadableError, match="length"):
        ocr_with_retries(client, PNG, max_attempts=3, sleep=lambda _: None)


def test_unreadable_is_a_kind_of_failure() -> None:
    assert issubclass(OcrPageUnreadableError, OcrFailedError)


def test_server_errors_are_never_reported_as_unreadable() -> None:
    """If any attempt failed for a server reason, the problem may not be the page."""
    client = ScriptedClient(
        [OcrError("HTTP 503"), OcrIncompleteError("length"), OcrIncompleteError("length")]
    )
    with pytest.raises(OcrFailedError) as excinfo:
        ocr_with_retries(client, PNG, max_attempts=3, sleep=lambda _: None)
    assert not isinstance(excinfo.value, OcrPageUnreadableError)
