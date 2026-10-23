"""Results shared by every retriever (BM25 now; dense, hybrid and rerank in Phase 2)."""

from dataclasses import dataclass

from rig_assist.ingestion.models import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with the score one retriever gave it. Higher is more relevant.

    Scores are only comparable within one retriever: a BM25 score and a cosine
    similarity are on different scales (which is why hybrid search fuses ranks).
    """

    chunk: Chunk
    score: float
