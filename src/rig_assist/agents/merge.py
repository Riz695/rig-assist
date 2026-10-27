"""Build the WorkOrder from the nodes' outputs. Code, not an LLM, so these always hold:

- **Safety precedence:** an action the safety review judges unsafe, or blocks in free
  text, is in `blocked_actions` and never in `recommended_actions`, whatever the diagnosis
  proposed. A free-text block wins over a "safe" verdict on the same action.
- **Escalation:** `escalate_to_engineer` is true when safety flags a hazard or no cause
  has supporting evidence (including no cause at all).
- **Citation integrity:** every citation is a retrieved chunk's page, with that chunk's
  text as the snippet. Evidence on a page that wasn't retrieved is dropped.

Without a diagnosis (triage flagged critical information missing) the work order carries
the open questions only, and isn't escalated: it asks before anything is diagnosed.
"""

from collections.abc import Sequence

from rig_assist.agents.schemas import (
    Cause,
    Citation,
    Diagnosis,
    EvidenceRef,
    Likelihood,
    SafetyReview,
    TriageResult,
    WorkOrder,
    normalise_action,
)
from rig_assist.rag.pipeline import make_snippet
from rig_assist.retrieval.models import ScoredChunk

_RANK: dict[Likelihood, int] = {"high": 0, "medium": 1, "low": 2}


def _equipment(triage: TriageResult) -> str:
    if triage.equipment_id:
        return f"{triage.equipment} ({triage.equipment_id})"
    return triage.equipment


def _unique(actions: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for action in actions:
        key = normalise_action(action)
        if key not in seen:
            seen.add(key)
            out.append(action)
    return out


def proposed_actions(diagnosis: Diagnosis) -> list[str]:
    """The diagnosis's actions without duplicates: the list the safety review numbers."""
    return _unique(diagnosis.recommended_actions)


class _Pages:
    """Retrieved chunks by page: the best-ranked chunk on each page gives the snippet."""

    def __init__(self, hits: Sequence[ScoredChunk]) -> None:
        self._by_page: dict[tuple[str, int], ScoredChunk] = {}
        for hit in hits:  # hits are in rank order: keep the first per page
            self._by_page.setdefault((hit.chunk.doc_id, hit.chunk.page), hit)

    def cite(self, refs: Sequence[EvidenceRef]) -> list[Citation]:
        """Citations for the refs that were retrieved, once each, in order."""
        out: list[Citation] = []
        seen: set[tuple[str, int]] = set()
        for ref in refs:
            key = (ref.doc_id, ref.page)
            hit = self._by_page.get(key)
            if hit is None or key in seen:
                continue
            seen.add(key)
            out.append(
                Citation(doc_id=ref.doc_id, page=ref.page, snippet=make_snippet(hit.chunk.text))
            )
        return out


def build_work_order(
    triage: TriageResult,
    hits: Sequence[ScoredChunk],
    diagnosis: Diagnosis | None,
    safety: SafetyReview | None,
) -> WorkOrder:
    if diagnosis is None:
        return WorkOrder(
            equipment=_equipment(triage),
            symptoms=triage.symptoms,
            probable_causes=[],
            recommended_actions=[],
            safety_warnings=[],
            blocked_actions=[],
            escalate_to_engineer=False,
            open_questions=triage.open_questions,
            citations=[],
        )
    if safety is None:
        raise ValueError("a diagnosis needs a safety review before it becomes a work order")

    proposed = proposed_actions(diagnosis)
    numbers = sorted({v.number for v in safety.verdicts if v.verdict == "unsafe"})
    for number in numbers:
        if number > len(proposed):
            raise ValueError(f"safety blocked action {number}; only {len(proposed)} proposed")
    blocked = _unique(
        [proposed[n - 1] for n in numbers] + [other.action for other in safety.other_blocked]
    )
    blocked_keys = {normalise_action(a) for a in blocked}
    recommended = [a for a in proposed if normalise_action(a) not in blocked_keys]

    pages = _Pages(hits)
    causes = [
        Cause(cause=c.cause, likelihood=c.likelihood, evidence=pages.cite(c.evidence))
        for c in sorted(diagnosis.causes, key=lambda c: _RANK[c.likelihood])
    ]
    all_refs = [ref for c in diagnosis.causes for ref in c.evidence] + safety.evidence
    supported = any(cause.evidence for cause in causes)
    return WorkOrder(
        equipment=_equipment(triage),
        symptoms=triage.symptoms,
        probable_causes=causes,
        recommended_actions=recommended,
        safety_warnings=safety.warnings,
        blocked_actions=blocked,
        escalate_to_engineer=safety.hazard or not supported,
        open_questions=triage.open_questions,
        citations=pages.cite(all_refs),
    )
