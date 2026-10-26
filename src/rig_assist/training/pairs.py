"""Training pairs for the embedder: (synthetic query, the chunk it was written for).

The pairs feed a contrastive loss with in-batch negatives: in a batch, each query's own
chunk is the positive and every other chunk in the batch is treated as a wrong answer.
Three rules follow from that and from evaluating on a frozen eval set:

- **Leakage guard:** no chunk on a page the eval set expects becomes a positive, so the
  model is never trained on the answers it will be tested on.
- **No ambiguous queries:** a query written for two different chunks is dropped, since one
  of them would be trained as a wrong answer to it. Repeats for the same chunk count once.
- **Stable split by page:** a page's split comes from a hash of (doc_id, page), not from the
  seed or the input order, so all chunks of a page land together (no near-duplicate text on
  both sides) and adding chunks or changing the seed never moves a page.

The seed only sets the shuffled order of the pairs, which decides which pairs share a batch.
"""

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from rig_assist.evaluation.eval_set import EvalQuestion
from rig_assist.ingestion.models import Chunk

PageKey = tuple[str, int]  # (doc_id, page)
Split = Literal["train", "val"]

MIN_WORDS = 30  # shorter chunks are title pages and dividers: nothing to ask about
VAL_FRACTION = 0.1


@dataclass(frozen=True)
class TrainingPair:
    query: str
    chunk_id: str
    doc_id: str
    page: int
    positive: str  # the chunk's text
    split: Split


def eval_pages(questions: Iterable[EvalQuestion]) -> set[PageKey]:
    """Every (doc_id, page) an eval question expects as its answer."""
    return {
        (q.expected_doc_id, page)
        for q in questions
        if q.expected_doc_id is not None
        for page in q.expected_pages
    }


def trainable_chunks(
    chunks: Iterable[Chunk], excluded: set[PageKey], min_words: int = MIN_WORDS
) -> list[Chunk]:
    """Chunks worth writing queries for, by chunk_id: not on an eval page, not too short."""
    kept = [
        c for c in chunks if (c.doc_id, c.page) not in excluded and len(c.text.split()) >= min_words
    ]
    return sorted(kept, key=lambda c: c.chunk_id)


def split_of(doc_id: str, page: int, val_fraction: float = VAL_FRACTION) -> Split:
    """'val' for about `val_fraction` of pages, decided by a hash of the page alone."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    digest = hashlib.sha256(f"{doc_id}:{page}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    return "val" if position < val_fraction else "train"


def _normalise(query: str) -> str:
    return " ".join(query.split()).casefold()


def build_pairs(
    chunks: Sequence[Chunk],
    queries: Mapping[str, Sequence[str]],
    excluded: set[PageKey],
    val_fraction: float = VAL_FRACTION,
    seed: int = 0,
) -> list[TrainingPair]:
    """One pair per distinct (query, chunk), shuffled by `seed`.

    `queries` maps chunk_id to the queries written for it. A chunk_id that isn't in
    `chunks` is an error (queries made for an older index); chunks on `excluded` pages
    are skipped even when they have queries.
    """
    by_id = {c.chunk_id: c for c in chunks}
    unknown = sorted(set(queries) - set(by_id))
    if unknown:
        raise ValueError(f"queries for chunks not in the index: {', '.join(unknown[:5])}")

    first_text: dict[str, str] = {}  # normalised query -> first spelling seen
    owners: dict[str, set[str]] = defaultdict(set)  # normalised query -> chunk_ids
    for chunk_id in sorted(queries):
        chunk = by_id[chunk_id]
        if (chunk.doc_id, chunk.page) in excluded:
            continue
        for query in queries[chunk_id]:
            key = _normalise(query)
            if key:
                first_text.setdefault(key, query.strip())
                owners[key].add(chunk_id)

    pairs = []
    for key in sorted(owners):
        if len(owners[key]) > 1:
            continue  # ambiguous: written for more than one chunk
        chunk = by_id[next(iter(owners[key]))]
        pairs.append(
            TrainingPair(
                query=first_text[key],
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                positive=chunk.text,
                split=split_of(chunk.doc_id, chunk.page, val_fraction),
            )
        )
    random.Random(seed).shuffle(pairs)
    return pairs
