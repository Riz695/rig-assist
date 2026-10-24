"""The eval set (`eval/questions.jsonl`): schema, validator, and reference page text.

Every answerable question names a manual and its PDF pages (1-based positions, not
printed page numbers) and quotes evidence copied from one of those pages. The validator
checks the quote really is on that page, so a question can't point at the wrong page
by mistake. Reference text = the text layer, or the cached OCR output for scanned pages.
"""

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, model_validator

from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.parsers.olmocr import PARSER_NAME as OCR_PARSER
from rig_assist.ingestion.parsers.text_layer import extract_text_layer, has_text_layer
from rig_assist.ingestion.pipeline import make_doc_id

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
QuestionType = Literal["factual", "troubleshooting", "procedure", "table", "unanswerable"]
Difficulty = Literal["easy", "medium", "hard"]

# doc_id -> page texts; index 0 is page 1.
ReferencePages = Mapping[str, Sequence[str]]


class EvalQuestion(BaseModel):
    """One line of `eval/questions.jsonl`. Unknown fields are rejected (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyStr
    question: NonEmptyStr
    expected_doc_id: str | None
    expected_pages: list[int]
    reference_answer: NonEmptyStr
    evidence_quote: str | None
    type: QuestionType
    difficulty: Difficulty

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"

    @model_validator(mode="after")
    def _check_consistency(self) -> "EvalQuestion":
        problems = []
        if self.answerable:
            if not self.expected_doc_id:
                problems.append("answerable questions need expected_doc_id")
            if not self.expected_pages:
                problems.append("answerable questions need expected_pages")
            if not (self.evidence_quote or "").strip():
                problems.append("answerable questions need an evidence_quote")
        elif self.expected_doc_id is not None or self.expected_pages or self.evidence_quote:
            problems.append(
                "unanswerable questions need expected_doc_id null, expected_pages [] "
                "and evidence_quote null"
            )
        if len(set(self.expected_pages)) != len(self.expected_pages):
            problems.append("expected_pages has duplicates")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class EvalSetError(ValueError):
    """The eval set has problems; `errors` lists all of them."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("invalid eval set:\n" + "\n".join(errors))
        self.errors = errors


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quote_on_page(quote: str, page_text: str) -> bool:
    """True if `quote` appears in `page_text`, ignoring case, spacing and line breaks.

    PDF text wraps lines wherever the layout did, so a quote copied as one line must
    still match. Words and punctuation must match exactly.
    """
    return _normalize(quote) in _normalize(page_text)


def _check_references(question: EvalQuestion, pages: ReferencePages) -> list[str]:
    """Doc exists, pages in range, quote on one of the pages. Answerable questions only."""
    doc_id = question.expected_doc_id
    if doc_id is None:
        return []
    if doc_id not in pages:
        return [f"unknown expected_doc_id {doc_id!r}"]
    doc_pages = pages[doc_id]
    out_of_range = [p for p in question.expected_pages if not 1 <= p <= len(doc_pages)]
    if out_of_range:
        return [
            f"page {p} out of range for {doc_id} (pages 1..{len(doc_pages)})" for p in out_of_range
        ]
    quote = question.evidence_quote or ""
    if not any(quote_on_page(quote, doc_pages[p - 1]) for p in question.expected_pages):
        return [f"evidence_quote not found on {doc_id} pages {question.expected_pages}"]
    return []


def _parse_lines(
    lines: Iterable[str], pages: ReferencePages
) -> tuple[list[EvalQuestion], list[str]]:
    questions: list[EvalQuestion] = []
    errors: list[str] = []
    ids: Counter[str] = Counter()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: not valid JSON ({error.msg})")
            continue
        if not isinstance(raw, dict):
            errors.append(f"line {number}: expected a JSON object")
            continue
        where = f"line {number} ({raw.get('id', '?')})"
        if isinstance(raw.get("id"), str):
            ids[raw["id"]] += 1
        try:
            question = EvalQuestion.model_validate(raw)
        except ValidationError as error:
            for item in error.errors():
                field = ".".join(str(part) for part in item["loc"])
                errors.append(f"{where}: {field + ': ' if field else ''}{item['msg']}")
            continue
        errors.extend(f"{where}: {problem}" for problem in _check_references(question, pages))
        questions.append(question)
    errors.extend(f"duplicate id {qid!r} ({n} times)" for qid, n in ids.items() if n > 1)
    return questions, errors


def validate_eval_set(lines: Iterable[str], pages: ReferencePages) -> list[str]:
    """Every problem in the eval set, or [] if it is valid."""
    return _parse_lines(lines, pages)[1]


def load_eval_set(path: Path, pages: ReferencePages) -> list[EvalQuestion]:
    """Read and validate a questions.jsonl; raises EvalSetError listing every problem."""
    questions, errors = _parse_lines(path.read_text(encoding="utf-8").splitlines(), pages)
    if errors:
        raise EvalSetError(errors)
    return questions


def load_reference_pages(raw_dir: Path, cache: PageCache) -> dict[str, list[str]]:
    """Reference text of every page of every PDF in `raw_dir`, keyed by doc_id.

    A page with a text layer uses it; a scanned page uses its cached OCR output. A page
    with neither keeps its (near-empty) text, so page numbering stays right.
    """
    reference: dict[str, list[str]] = {}
    for path in sorted(raw_dir.glob("*.pdf")):
        doc_id = make_doc_id(path.name)
        file_hash: str | None = None
        texts = []
        for page in extract_text_layer(path, doc_id):
            text = page.text
            if not has_text_layer(text):
                file_hash = file_hash or file_sha256(path)
                cached = cache.get(file_hash, page.page, OCR_PARSER)
                if cached is not None:
                    text = cached.text
            texts.append(text)
        reference[doc_id] = texts
    return reference
