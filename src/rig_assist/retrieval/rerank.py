"""Second-stage reranking with a cross-encoder.

A first-stage retriever (BM25, dense or hybrid) is fast but scores the query and each
chunk separately. A cross-encoder reads the query and one chunk *together* through one
transformer and outputs a relevance score, which is more accurate but costs a full model
pass per (query, chunk) pair. So it only reranks the first stage's top `candidates`.

Measured on this laptop's CPU (12 threads), 20 real chunks per query:
`BAAI/bge-reranker-base` ~10.7 s, `Xenova/ms-marco-MiniLM-L-6-v2` ~2.4 s,
`jinaai/jina-reranker-v1-tiny-en` ~2.0 s. Phase 3 measures what the extra time buys.
"""

from dataclasses import replace
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rig_assist.retrieval.models import Retriever, ScoredChunk


class CrossEncoder(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, in input order. Higher is more relevant."""
        ...


class FastEmbedCrossEncoder:
    """fastembed cross-encoder (ONNX, CPU), loaded on first use.

    Scores are raw logits, not probabilities (bge-reranker-base gave -5.7..+2.4 on two
    real queries; a sigmoid would map them to 0..1). Only their order matters here.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None

    def _load(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._load().rerank(query, texts)]

    def unload(self) -> None:
        """Free the model's memory (bge-reranker-base: ~1.4 GB); the next call reloads it."""
        self._model = None


class RerankRetriever:
    """Take the first stage's top `candidates`, rescore them with a cross-encoder, keep k."""

    def __init__(
        self, first_stage: Retriever, cross_encoder: CrossEncoder, candidates: int = 20
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates must be at least 1")
        self.first_stage = first_stage
        self.cross_encoder = cross_encoder
        self.candidates = candidates

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if k < 1:
            raise ValueError("k must be at least 1")
        hits = self.first_stage.search(query, max(self.candidates, k))
        if not hits:
            return []
        scores = self.cross_encoder.score(query, [hit.chunk.text for hit in hits])
        # sorted() is stable: equal scores keep the first stage's order.
        ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: -pair[1])
        return [replace(hit, score=score) for hit, score in ranked[:k]]
