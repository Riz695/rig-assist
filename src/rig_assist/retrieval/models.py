"""Results shared by every retriever (BM25 now; dense, hybrid and rerank in Phase 2)."""

from dataclasses import dataclass
from typing import Protocol

from rig_assist.ingestion.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the score one retriever gave it. Higher is more relevant.

    Scores are only comparable within one retriever: a BM25 score and a cosine
    similarity are on different scales (which is why hybrid search fuses ranks).
    """

    chunk: Chunk
    score: float
    # Cosine similarity from dense retrieval, kept through fusion: Phase 3's "not found"
    # threshold needs a score on a fixed scale, which an RRF score isn't. None = dense
    # retrieval didn't return this chunk.
    dense_score: float | None = None


class Retriever(Protocol):
    """Anything that ranks chunks for a query: BM25, dense, hybrid, rerank."""

    def search(self, query: str, k: int) -> list[ScoredChunk]: ...
