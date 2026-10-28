"""The LLM nodes: triage, retrieval (a bounded search loop), diagnosis, safety.

Plain functions (LLM and retriever in, validated output and token usage out), so they
are tested without the graph; `graph.py` wires them together, and the report node is
`merge.build_work_order`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rig_assist.agents.merge import proposed_actions
from rig_assist.agents.prompts import (
    DIAGNOSIS_PROMPT,
    SAFETY_PROMPT,
    SEARCH_PROMPT,
    TRIAGE_PROMPT,
)
from rig_assist.agents.schemas import (
    Diagnosis,
    SafetyReview,
    SearchPlan,
    TriageResult,
    normalise_action,
)
from rig_assist.agents.structured import Structured, complete_structured
from rig_assist.llm.models import LlmClient, Message, Usage
from rig_assist.rag.prompt import citation_label, context_block
from rig_assist.retrieval.models import Retriever, ScoredChunk

MAX_SEARCH_ROUNDS = 2  # planner calls after the first, fixed search
MAX_QUERIES_PER_ROUND = 3
PER_QUERY_K = 5
MAX_EVIDENCE = 12  # chunks handed to diagnosis
SAFETY_K = 5
PREVIEW_CHARS = 160  # how much of each excerpt the search planner sees
# Room for hidden reasoning tokens (thinking models count them as output) plus the JSON.
NODE_MAX_TOKENS = 4096


def _user(*parts: str) -> Message:
    return Message("user", "\n\n".join(part for part in parts if part))


def _problem(description: str, triage: TriageResult) -> str:
    lines = [f"Technician's description:\n{description}", f"Equipment: {triage.equipment}"]
    if triage.equipment_id:
        lines.append(f"Equipment ID: {triage.equipment_id}")
    lines.append("Symptoms: " + "; ".join(triage.symptoms))
    if triage.operating_context:
        lines.append(f"Operating context: {triage.operating_context}")
    return "\n".join(lines)


def _context(hits: Sequence[ScoredChunk]) -> str:
    if not hits:
        return "Manual excerpts: none found."
    return "Manual excerpts:\n\n" + "\n\n".join(context_block(hit) for hit in hits)


def _add_new(hits: list[ScoredChunk], found: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Append the chunks not already in `hits` (in place); return the ones added."""
    known = {hit.chunk.chunk_id for hit in hits}
    added = []
    for hit in found:
        if hit.chunk.chunk_id not in known:
            known.add(hit.chunk.chunk_id)
            hits.append(hit)
            added.append(hit)
    return added


# --- triage ------------------------------------------------------------------


def run_triage(
    llm: LlmClient, description: str, equipment_id: str | None = None
) -> Structured[TriageResult]:
    user = f"Technician's description:\n{description}"
    if equipment_id:
        user += f"\n\nEquipment ID given by the technician: {equipment_id}"
    messages = [Message("system", TRIAGE_PROMPT), Message("user", user)]
    result = complete_structured(llm, messages, TriageResult, NODE_MAX_TOKENS)
    if equipment_id:  # the technician's ID wins over the model's reading
        value = result.value.model_copy(update={"equipment_id": equipment_id})
        return Structured(value, result.usage, result.attempts)
    return result


# --- retrieval ---------------------------------------------------------------


@dataclass(frozen=True)
class Retrieval:
    hits: list[ScoredChunk]  # first-found order, no duplicates, at most MAX_EVIDENCE
    queries: list[str]  # every query searched, in order
    rounds: int  # planner rounds
    usage: Usage
    llm_calls: int  # planner calls, retries included


def _first_query(triage: TriageResult, description: str) -> str:
    if not triage.symptoms:
        return description
    return f"{triage.equipment}: " + "; ".join(triage.symptoms)


def _previews(hits: Sequence[ScoredChunk]) -> str:
    if not hits:
        return "Found so far: nothing."
    lines = []
    for hit in hits:
        text = " ".join(hit.chunk.text.split())[:PREVIEW_CHARS]
        lines.append(f"- {citation_label(hit.chunk.doc_id, hit.chunk.page)} {text}")
    return "Found so far:\n" + "\n".join(lines)


def run_retrieval(
    llm: LlmClient,
    retriever: Retriever,
    description: str,
    triage: TriageResult,
    per_query_k: int = PER_QUERY_K,
    max_rounds: int = MAX_SEARCH_ROUNDS,
) -> Retrieval:
    """One fixed search (equipment + symptoms), then up to `max_rounds` planner calls,
    each adding up to 3 new queries. Stops when the planner is done, proposes nothing
    new, or the evidence is full."""
    hits: list[ScoredChunk] = []
    queries: list[str] = []
    searched: set[str] = set()
    usage = Usage()
    rounds = 0
    llm_calls = 0

    def search(query: str) -> None:
        searched.add(" ".join(query.split()).casefold())
        queries.append(query)
        _add_new(hits, retriever.search(query, per_query_k))

    search(_first_query(triage, description))
    while rounds < max_rounds and len(hits) < MAX_EVIDENCE:
        messages = [
            Message("system", SEARCH_PROMPT),
            _user(
                _problem(description, triage),
                "Queries already run:\n" + "\n".join(f"- {q}" for q in queries),
                _previews(hits),
            ),
        ]
        plan = complete_structured(llm, messages, SearchPlan, NODE_MAX_TOKENS)
        usage += plan.usage
        llm_calls += plan.attempts
        rounds += 1
        new = [
            q
            for q in plan.value.queries[:MAX_QUERIES_PER_ROUND]
            if " ".join(q.split()).casefold() not in searched
        ]
        if plan.value.done or not new:
            break
        for query in new:
            search(query)
    return Retrieval(hits[:MAX_EVIDENCE], queries, rounds, usage, llm_calls)


# --- diagnosis ---------------------------------------------------------------


def run_diagnosis(
    llm: LlmClient, description: str, triage: TriageResult, hits: Sequence[ScoredChunk]
) -> Structured[Diagnosis]:
    messages = [
        Message("system", DIAGNOSIS_PROMPT),
        _user(_problem(description, triage), _context(hits)),
    ]
    return complete_structured(llm, messages, Diagnosis, NODE_MAX_TOKENS)


# --- safety ------------------------------------------------------------------


@dataclass(frozen=True)
class Safety:
    review: Structured[SafetyReview]
    new_hits: list[ScoredChunk]  # safety excerpts the earlier search hadn't found


def run_safety(
    llm: LlmClient,
    retriever: Retriever,
    description: str,
    triage: TriageResult,
    diagnosis: Diagnosis,
    hits: Sequence[ScoredChunk],
) -> Safety:
    """Searches the manuals for the equipment's hazards and the proposed actions, then
    reviews the actions against all the evidence: a verdict per numbered action."""
    actions = proposed_actions(diagnosis)
    query = f"{triage.equipment} safety hazards precautions warnings"
    if actions:
        query += ": " + "; ".join(actions)
    evidence = list(hits)
    new_hits = _add_new(evidence, retriever.search(query, SAFETY_K))

    numbered = "\n".join(f"{i}. {action}" for i, action in enumerate(actions, start=1))
    messages = [
        Message("system", SAFETY_PROMPT),
        _user(
            _problem(description, triage),
            "Proposed actions:\n" + (numbered or "(none)"),
            _context(evidence),
        ),
    ]

    review = complete_structured(
        llm,
        messages,
        SafetyReview,
        NODE_MAX_TOKENS,
        check=lambda review: check_verdicts(review, actions),
    )
    return Safety(review, new_hits)


def check_verdicts(review: SafetyReview, actions: Sequence[str]) -> None:
    """Exactly one verdict per numbered action, each copying the action it judges.
    Raises ValueError with what to fix: `complete_structured` sends it back for the retry."""
    numbers = [v.number for v in review.verdicts]
    wrong = sorted({n for n in numbers if n > len(actions)})
    if wrong:
        raise ValueError(
            f"verdict numbers {wrong} don't exist: there are {len(actions)} proposed actions"
        )
    repeated = sorted({n for n in numbers if numbers.count(n) > 1})
    if repeated:
        raise ValueError(f"verdict numbers {repeated} appear more than once: one per action")
    missing = [n for n in range(1, len(actions) + 1) if n not in numbers]
    if missing:
        raise ValueError(f"verdicts are missing for actions {missing}: give one per action")
    for v in review.verdicts:
        expected = actions[v.number - 1]
        if normalise_action(v.action) != normalise_action(expected):
            raise ValueError(
                f"verdict {v.number} copies {v.action!r}, but action {v.number} is "
                f"{expected!r}: copy each action exactly as numbered"
            )
