"""The searchable index as the API sees it: the store on disk, an in-memory BM25, the
dense index in Qdrant, and the cross-encoder, plus which retriever each search mode uses.

BM25 is built from all stored chunks on first use and rebuilt after every upload
(~0.1 s for the full corpus). One instance lives on the app for its whole life.
"""

import threading
from typing import Literal, get_args

from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.bm25 import BM25Index
from rig_assist.retrieval.dense import DenseIndex
from rig_assist.retrieval.hybrid import HybridRetriever
from rig_assist.retrieval.models import Retriever
from rig_assist.retrieval.rerank import CrossEncoder, RerankRetriever

SearchMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
SEARCH_MODES: tuple[SearchMode, ...] = get_args(SearchMode)
DENSE_MODES: frozenset[SearchMode] = frozenset({"dense", "hybrid", "hybrid_rerank"})


class SearchIndex:
    def __init__(self, store: IndexStore, dense: DenseIndex, cross_encoder: CrossEncoder) -> None:
        self.store = store
        self.dense = dense
        self.cross_encoder = cross_encoder
        self._bm25: BM25Index | None = None
        # Uploads run in FastAPI's thread pool, so two could refresh at once.
        self._lock = threading.Lock()

    def refresh(self) -> None:
        """Rebuild from the store. Call after a document is added or replaced."""
        bm25 = BM25Index.build(self.store.chunks())
        with self._lock:
            self._bm25 = bm25

    def bm25(self) -> BM25Index:
        with self._lock:
            bm25 = self._bm25
        if bm25 is None:
            self.refresh()
            return self.bm25()
        return bm25

    def is_empty(self) -> bool:
        return len(self.bm25()) == 0

    def retriever(self, mode: SearchMode) -> Retriever:
        """The retriever for `mode`. Built per call (cheap) so it sees the latest BM25."""
        if mode == "bm25":
            return self.bm25()
        if mode == "dense":
            return self.dense
        # v1's settings: 20 results from each retriever, RRF k=60.
        hybrid = HybridRetriever(self.dense, self.bm25(), k_each=20, rrf_k=60)
        if mode == "hybrid":
            return hybrid
        if mode == "hybrid_rerank":
            return RerankRetriever(hybrid, self.cross_encoder, candidates=20)
        raise ValueError(f"unknown search mode: {mode!r}")
