"""The searchable index as the API sees it: the store on disk plus an in-memory BM25.

BM25 is built from all stored chunks on first use and rebuilt after every upload
(~0.1 s for the full corpus). One instance lives on the app for its whole life.
"""

import threading

from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.bm25 import BM25Index


class SearchIndex:
    def __init__(self, store: IndexStore) -> None:
        self.store = store
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
