"""Hard negatives: for each training pair, chunks that look relevant but aren't its answer.

In-batch negatives (`pairs.py`) are mostly easy: a random other chunk is usually about
something else. A hard negative is one the current retrievers rank high for the query, so
training on it teaches the fine distinctions (the right valve, the right interval).

Candidates are the `top_k` allowed chunks from BM25 (shared words) and from the base
embedder (similar meaning); the hardest by embedder similarity are kept. Rules:

- **Never the answer's page:** the positive chunk and every chunk on its page are skipped,
  and so is any chunk whose text equals the positive's (the same text on another page).
- **Likely false negatives dropped:** a candidate scoring at least `max_relative_score` of
  the positive's similarity probably answers the query too (repeated content across the
  manuals), and training it as wrong would hurt. NV-Retriever's "TopK-PercPos" rule, 95%.
- **Validation pages stay unseen:** a train pair never gets a negative from a validation
  page, so no validation text is trained on. A validation pair may take any other page.

The pool should be the trainable chunks, which already leaves out the eval pages.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.bm25 import BM25Index
from rig_assist.retrieval.embedder import Embedder
from rig_assist.training.pairs import VAL_FRACTION, Split, TrainingPair, split_of

N_NEGATIVES = 1
TOP_K = 30
MAX_RELATIVE_SCORE = 0.95

Source = Literal["bm25", "dense", "both"]


@dataclass(frozen=True)
class MinedPair:
    query: str
    chunk_id: str
    doc_id: str
    page: int
    positive: str
    split: Split
    negative_ids: list[str]  # hardest first; may be fewer than asked, or none
    negatives: list[str]  # their texts
    sources: list[Source]  # which retriever proposed each negative


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _ids(keys: list[object]) -> np.ndarray:
    first: dict[object, int] = {}
    return np.array([first.setdefault(key, len(first)) for key in keys])


def _unit_rows(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def mine_negatives(
    pairs: Sequence[TrainingPair],
    pool: Sequence[Chunk],
    embedder: Embedder,
    n_negatives: int = N_NEGATIVES,
    top_k: int = TOP_K,
    max_relative_score: float = MAX_RELATIVE_SCORE,
    val_fraction: float = VAL_FRACTION,
) -> list[MinedPair]:
    """Up to `n_negatives` hard negatives per pair, in the pairs' order."""
    if n_negatives < 1 or top_k < 1:
        raise ValueError("n_negatives and top_k must be at least 1")
    if not 0.0 < max_relative_score <= 1.0:
        raise ValueError(f"max_relative_score must be in (0, 1], got {max_relative_score}")

    chunks = sorted(pool, key=lambda c: c.chunk_id)  # ties break by chunk_id
    row = {c.chunk_id: i for i, c in enumerate(chunks)}
    missing = sorted({p.chunk_id for p in pairs} - set(row))
    if missing:
        raise ValueError(f"positive chunks not in the pool: {', '.join(missing[:5])}")

    # Same page / same text as integer ids, so "allowed" is one vectorised comparison per pair.
    page_ids = _ids([(c.doc_id, c.page) for c in chunks])
    text_ids = _ids([_normalise(c.text) for c in chunks])
    is_val = np.array([split_of(c.doc_id, c.page, val_fraction) == "val" for c in chunks])
    bm25 = BM25Index.build(chunks)
    passages = _unit_rows(embedder.embed_passages([c.text for c in chunks]))
    queries = _unit_rows([embedder.embed_query(p.query) for p in pairs])
    similarity = queries @ passages.T  # cosine, pairs x chunks

    mined = []
    for i, pair in enumerate(pairs):
        positive = row[pair.chunk_id]
        allowed = (page_ids != page_ids[positive]) & (text_ids != text_ids[positive])
        if pair.split == "train":
            allowed &= ~is_val

        scores = similarity[i]
        dense_order = sorted(np.flatnonzero(allowed), key=lambda j: (-scores[j], j))
        dense = set(dense_order[:top_k])
        hits = [row[hit.chunk.chunk_id] for hit in bm25.search(pair.query, len(chunks))]
        lexical = set([j for j in hits if allowed[j]][:top_k])

        ceiling = max_relative_score * scores[positive]
        candidates = sorted(dense | lexical, key=lambda j: (-scores[j], j))
        kept = [j for j in candidates if scores[j] < ceiling][:n_negatives]
        sources: list[Source] = [
            "both" if j in dense and j in lexical else "dense" if j in dense else "bm25"
            for j in kept
        ]
        mined.append(
            MinedPair(
                query=pair.query,
                chunk_id=pair.chunk_id,
                doc_id=pair.doc_id,
                page=pair.page,
                positive=pair.positive,
                split=pair.split,
                negative_ids=[chunks[j].chunk_id for j in kept],
                negatives=[chunks[j].text for j in kept],
                sources=sources,
            )
        )
    return mined


def mining_stats(mined: Sequence[MinedPair], n_negatives: int) -> dict[str, object]:
    sources: Counter[str] = Counter(s for m in mined for s in m.sources)
    return {
        "pairs": len(mined),
        "short": sum(len(m.negative_ids) < n_negatives for m in mined),
        "no_negative": sum(not m.negative_ids for m in mined),
        "negatives": sum(len(m.negative_ids) for m in mined),
        "sources": {name: sources[name] for name in ("bm25", "dense", "both")},
    }


def write_mined(mined: Sequence[MinedPair], out_dir: Path) -> dict[str, int]:
    """`train.jsonl` and `val.jsonl` in the given order; returns rows per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for split in ("train", "val"):
        rows = [asdict(m) for m in mined if m.split == split]
        counts[split] = len(rows)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (out_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
    return dict(counts)
