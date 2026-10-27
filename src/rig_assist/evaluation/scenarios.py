"""Troubleshooting scenarios (`eval/scenarios.jsonl`) for comparing `/ask` and `/diagnose`.

Each scenario is a technician's problem report with the expected cause (backed by a
manual page and a quote the validator finds on it) and the safety warnings a good answer
gives. An `unsafe_temptation` scenario also names an unsafe action the technician is
tempted to take: a good answer must not recommend it (`/diagnose` should block it).

Same rules as the eval set: pages are 1-based PDF positions, quotes are checked against
the reference page text, and the file is frozen after a spot-check (CHANGELOG.md).
"""

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rig_assist.evaluation.eval_set import (
    EvalSetError,
    NonEmptyStr,
    ReferencePages,
    quote_on_page,
)

ScenarioKind = Literal["troubleshooting", "unsafe_temptation"]
MIN_UNSAFE = 3  # PLAN.md: at least 3 scenarios tempt an unsafe action


class Scenario(BaseModel):
    """One line of `eval/scenarios.jsonl`. Unknown fields are rejected (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyStr
    description: NonEmptyStr  # what the technician reports, in their words
    equipment_id: NonEmptyStr | None = None
    kind: ScenarioKind
    expected_cause: NonEmptyStr
    expected_doc_id: NonEmptyStr
    expected_pages: list[int] = Field(min_length=1)
    evidence_quote: NonEmptyStr  # copied from one of the expected pages
    expected_safety_warnings: list[NonEmptyStr]
    unsafe_action: NonEmptyStr | None = None  # required for unsafe_temptation only

    @model_validator(mode="after")
    def _unsafe_action_matches_kind(self) -> "Scenario":
        if self.kind == "unsafe_temptation" and self.unsafe_action is None:
            raise ValueError("an unsafe_temptation scenario needs an unsafe_action")
        if self.kind == "troubleshooting" and self.unsafe_action is not None:
            raise ValueError("unsafe_action is only for unsafe_temptation scenarios")
        return self


def _check_references(scenario: Scenario, pages: ReferencePages) -> list[str]:
    doc_id = scenario.expected_doc_id
    if doc_id not in pages:
        return [f"unknown expected_doc_id {doc_id!r}"]
    doc_pages = pages[doc_id]
    out_of_range = [p for p in scenario.expected_pages if not 1 <= p <= len(doc_pages)]
    if out_of_range:
        return [
            f"page {p} out of range for {doc_id} (pages 1..{len(doc_pages)})" for p in out_of_range
        ]
    if not any(
        quote_on_page(scenario.evidence_quote, doc_pages[p - 1]) for p in scenario.expected_pages
    ):
        return [f"evidence_quote not found on {doc_id} pages {scenario.expected_pages}"]
    return []


def _parse(lines: Iterable[str], pages: ReferencePages) -> tuple[list[Scenario], list[str]]:
    scenarios: list[Scenario] = []
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
            scenario = Scenario.model_validate(raw)
        except ValidationError as error:
            for item in error.errors():
                field = ".".join(str(part) for part in item["loc"])
                errors.append(f"{where}: {field + ': ' if field else ''}{item['msg']}")
            continue
        errors.extend(f"{where}: {problem}" for problem in _check_references(scenario, pages))
        scenarios.append(scenario)
    errors.extend(f"duplicate id {sid!r} ({n} times)" for sid, n in ids.items() if n > 1)
    unsafe = sum(s.kind == "unsafe_temptation" for s in scenarios)
    if unsafe < MIN_UNSAFE:
        errors.append(
            f"the set needs at least {MIN_UNSAFE} unsafe_temptation scenarios, has {unsafe}"
        )
    return scenarios, errors


def validate_scenarios(lines: Iterable[str], pages: ReferencePages) -> list[str]:
    """Every problem in the scenario set, or [] if it is valid."""
    return _parse(lines, pages)[1]


def load_scenarios(path: Path, pages: ReferencePages) -> list[Scenario]:
    """Read and validate a scenarios.jsonl; raises EvalSetError listing every problem."""
    scenarios, errors = _parse(path.read_text(encoding="utf-8").splitlines(), pages)
    if errors:
        raise EvalSetError(errors)
    return scenarios
