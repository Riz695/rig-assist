"""Eval-set validator: schema, references to real pages, and evidence quotes.

The validator reports every problem at once (with the line and question id), so a
50-question file can be fixed in one pass instead of one error per run.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rig_assist.evaluation.eval_set import (
    EvalSetError,
    load_eval_set,
    load_reference_pages,
    quote_on_page,
    validate_eval_set,
)
from rig_assist.ingestion.cache import PageCache, file_sha256
from rig_assist.ingestion.parsers.olmocr import PARSER_NAME as OCR_PARSER
from rig_assist.ingestion.parsers.olmocr import OcrResult

# doc_id -> page texts; index 0 is page 1.
CORPUS: dict[str, list[str]] = {
    "drilling-manual": [
        "Contents",
        "Close the annular preventer before\nopening the choke line.",
        "| Size | Torque |\n| --- | --- |\n| 5 in | 30 kN·m |",
    ],
    "crane-manual": ["Never exceed the safe working load (SWL) of the crane."],
}


def answerable(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "q001",
        "question": "What must be shut before the choke line is opened?",
        "expected_doc_id": "drilling-manual",
        "expected_pages": [2],
        "reference_answer": "The annular preventer.",
        "evidence_quote": "Close the annular preventer before opening the choke line.",
        "type": "procedure",
        "difficulty": "easy",
        "wording": "manual",
    }
    entry.update(overrides)
    return entry


def unanswerable(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "q050",
        "question": "What is the rig's helideck fuel capacity?",
        "expected_doc_id": None,
        "expected_pages": [],
        "reference_answer": "Not found in the manuals.",
        "evidence_quote": None,
        "type": "unanswerable",
        "difficulty": "medium",
        "wording": "paraphrased",
    }
    entry.update(overrides)
    return entry


def lines(*entries: dict[str, Any]) -> list[str]:
    return [json.dumps(entry) for entry in entries]


def errors_for(*entries: dict[str, Any]) -> list[str]:
    return validate_eval_set(lines(*entries), CORPUS)


# --- valid entries -------------------------------------------------------


def test_valid_answerable_and_unanswerable_entries_pass() -> None:
    assert errors_for(answerable(), unanswerable()) == []


def test_several_expected_pages_pass_when_the_quote_is_on_one_of_them() -> None:
    assert errors_for(answerable(expected_pages=[2, 3])) == []


def test_quote_from_a_table_passes() -> None:
    entry = answerable(type="table", expected_pages=[3], evidence_quote="| 5 in | 30 kN·m |")
    assert errors_for(entry) == []


def test_blank_lines_are_ignored() -> None:
    assert validate_eval_set([json.dumps(answerable()), "", "   "], CORPUS) == []


# --- schema --------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["id", "question", "expected_pages", "reference_answer", "type", "difficulty", "wording"],
)
def test_missing_field_is_reported(field: str) -> None:
    entry = answerable()
    del entry[field]
    errors = errors_for(entry)
    assert len(errors) == 1
    assert field in errors[0]


def test_unknown_type_and_difficulty_are_reported() -> None:
    errors = errors_for(answerable(type="trivia", difficulty="extreme"))
    assert any("type" in e for e in errors)
    assert any("difficulty" in e for e in errors)


def test_unknown_wording_is_reported() -> None:
    # "paraphrased" = worded differently from the manual (synonyms, technician language),
    # so BM25 can't match on the manual's own words. Reported as its own subset.
    errors = errors_for(answerable(wording="copied"))
    assert any("wording" in e for e in errors)


def test_unknown_extra_field_is_reported() -> None:
    # Catches typos like "expected_page" that would otherwise be silently ignored.
    errors = errors_for(answerable(expected_page=[2]))
    assert any("expected_page" in e for e in errors)


def test_empty_question_is_reported() -> None:
    assert any("question" in e for e in errors_for(answerable(question="  ")))


def test_line_that_is_not_json_is_reported_with_its_line_number() -> None:
    errors = validate_eval_set([json.dumps(answerable()), "{not json"], CORPUS)
    assert len(errors) == 1
    assert "line 2" in errors[0]


def test_errors_name_the_question_id() -> None:
    errors = errors_for(answerable(id="q007", type="trivia"))
    assert "q007" in errors[0]


def test_duplicate_ids_are_reported() -> None:
    errors = errors_for(answerable(id="q001"), unanswerable(id="q001"))
    assert len(errors) == 1
    assert "duplicate" in errors[0]
    assert "q001" in errors[0]


# --- references to the corpus --------------------------------------------


def test_unknown_doc_id_is_reported() -> None:
    errors = errors_for(answerable(expected_doc_id="no-such-manual"))
    assert len(errors) == 1
    assert "no-such-manual" in errors[0]


@pytest.mark.parametrize("page", [0, 4, -1])
def test_page_out_of_range_is_reported(page: int) -> None:
    # drilling-manual has 3 pages, numbered 1..3 (PDF positions, not printed numbers).
    errors = errors_for(answerable(expected_pages=[page]))
    assert any("page" in e and str(page) in e for e in errors)


def test_answerable_question_needs_expected_pages() -> None:
    assert any("expected_pages" in e for e in errors_for(answerable(expected_pages=[])))


def test_answerable_question_needs_a_doc_id() -> None:
    assert any("expected_doc_id" in e for e in errors_for(answerable(expected_doc_id=None)))


def test_duplicate_expected_pages_are_reported() -> None:
    assert any("expected_pages" in e for e in errors_for(answerable(expected_pages=[2, 2])))


# --- evidence quotes -----------------------------------------------------


def test_answerable_question_needs_an_evidence_quote() -> None:
    for quote in (None, "", "   "):
        assert any("evidence_quote" in e for e in errors_for(answerable(evidence_quote=quote)))


def test_quote_not_on_the_stated_page_is_reported() -> None:
    # The text is real, but on page 1 of the crane manual, not page 2 of this one.
    errors = errors_for(answerable(evidence_quote="safe working load"))
    assert len(errors) == 1
    assert "evidence_quote" in errors[0]


def test_invented_quote_is_reported() -> None:
    errors = errors_for(answerable(evidence_quote="Open the choke line first."))
    assert any("evidence_quote" in e for e in errors)


def test_quote_matching_ignores_line_breaks_spacing_and_case() -> None:
    assert quote_on_page(
        "close the ANNULAR   preventer before opening",
        "Close the annular\npreventer before opening the choke line.",
    )


def test_quote_matching_does_not_ignore_words() -> None:
    assert not quote_on_page("close the preventer", "Close the annular preventer")


# --- unanswerable entries ------------------------------------------------


def test_unanswerable_question_must_not_point_at_a_page() -> None:
    errors = errors_for(unanswerable(expected_doc_id="drilling-manual", expected_pages=[2]))
    assert any("unanswerable" in e for e in errors)


def test_unanswerable_question_must_not_have_an_evidence_quote() -> None:
    errors = errors_for(unanswerable(evidence_quote="Contents"))
    assert any("unanswerable" in e for e in errors)


# --- load_eval_set -------------------------------------------------------


def test_load_eval_set_returns_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(lines(answerable(), unanswerable())) + "\n", encoding="utf-8")
    questions = load_eval_set(path, CORPUS)
    assert [q.id for q in questions] == ["q001", "q050"]
    assert questions[0].expected_pages == [2]
    assert questions[1].answerable is False
    assert questions[0].answerable is True


def test_load_eval_set_raises_with_every_error(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    bad = lines(answerable(id="q1", type="trivia"), answerable(id="q2", expected_pages=[9]))
    path.write_text("\n".join(bad), encoding="utf-8")
    with pytest.raises(EvalSetError) as excinfo:
        load_eval_set(path, CORPUS)
    assert len(excinfo.value.errors) == 2
    assert "q1" in str(excinfo.value)
    assert "q2" in str(excinfo.value)


# --- load_reference_pages ------------------------------------------------


def test_reference_pages_use_the_text_layer_and_the_ocr_cache_for_scans(
    make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = make_pdf([["Check the mud pump liner."], []], name="raw/MudPumpManual.pdf")
    cache = PageCache(tmp_path / "parsed")
    cache.put(file_sha256(pdf), 2, OCR_PARSER, OcrResult("Scanned page text", 100, 10))

    pages = load_reference_pages(raw, cache)

    assert list(pages) == ["mud-pump-manual"]
    assert "Check the mud pump liner." in pages["mud-pump-manual"][0]
    assert pages["mud-pump-manual"][1] == "Scanned page text"


def test_reference_page_without_text_or_ocr_is_empty_but_still_counted(
    make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_pdf([["Page one."], []], name="raw/Manual.pdf")
    pages = load_reference_pages(raw, PageCache(tmp_path / "parsed"))
    # Page 2 exists (so page numbers stay right) but no quote can match it.
    assert len(pages["manual"]) == 2
    assert pages["manual"][1] == ""
