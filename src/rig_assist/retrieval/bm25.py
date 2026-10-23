"""Keyword search with BM25 (Okapi variant from `rank-bm25`).

Ported from v1 (`src/rag/bm25_index.py`), same tokenizer and scoring, so v1's
retrieval can be reproduced. One change: v1 pickled the index. Here nothing is
pickled: the chunks are saved as JSON by `IndexStore`, and the BM25 index is rebuilt
from them in memory (milliseconds for a few thousand chunks).
"""

import re

from rank_bm25 import BM25Okapi

from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.models import ScoredChunk

# Lowercase words and numbers, keeping codes like "h2s" and "rp-54" whole.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        # rank-bm25 divides by the corpus size, so an empty corpus has no model.
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        return cls(list(chunks))

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Return up to `k` chunks sharing at least one term with the query, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        # sorted() is stable, so ties keep index order and results are deterministic.
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in ranked[:k]
            if scores[i] > 0
        ]
