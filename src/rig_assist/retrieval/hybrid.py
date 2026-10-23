"""Hybrid search: fuse BM25 and dense rankings with reciprocal rank fusion (RRF).

Ported from v1 (`src/rag/hybrid.py`). RRF uses only ranks, never raw scores, which is
why it can combine BM25 scores (unbounded) and cosine similarities (-1..1) without any
normalisation: a chunk at rank r in a list earns 1 / (k + r), summed over the lists.
"""

from dataclasses import replace

from rig_assist.retrieval.models import Retriever, ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]], k: int = 60, top_n: int = 5
) -> list[ScoredChunk]:
    """Fuse ranked lists; the returned `score` is the RRF score, best first.

    `k` (60 in the original paper and in v1) damps the advantage of top ranks: with a
    small k, rank 1 dominates; with a large k, appearing in several lists matters more.
    Ties are broken by chunk_id so results are deterministic.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
            first_seen.setdefault(chunk_id, hit)
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [replace(first_seen[cid], score=scores[cid]) for cid in order[:top_n]]


class HybridRetriever:
    """Ask each retriever for `k_each` results (v1: 20) and fuse them with RRF."""

    def __init__(
        self, dense: Retriever, bm25: Retriever, k_each: int = 20, rrf_k: int = 60
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.k_each = k_each
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        dense_hits = self.dense.search(query, self.k_each)
        bm25_hits = self.bm25.search(query, self.k_each)
        dense_scores = {hit.chunk.chunk_id: hit.score for hit in dense_hits}
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], k=self.rrf_k, top_n=k)
        return [replace(hit, dense_score=dense_scores.get(hit.chunk.chunk_id)) for hit in fused]
