"""Dense (semantic) search: chunk embeddings stored in Qdrant, cosine similarity.

Each chunk is one Qdrant point: id = UUID5 of its chunk_id (deterministic, so the same
chunk always maps to the same point), vector = its embedding, payload = the Chunk's
fields. Re-ingesting a document deletes its points first, then writes the new ones, so
counts never grow and chunks that disappeared are removed.
"""

import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient, models

from rig_assist.config import Settings
from rig_assist.ingestion.models import Chunk
from rig_assist.retrieval.embedder import Embedder, FastEmbedEmbedder
from rig_assist.retrieval.models import ScoredChunk

# Any fixed namespace works; changing it would change every point ID.
_NAMESPACE = uuid.UUID("6f2d2c8e-7d1a-4b8e-9f3c-2a1b5e4d7c90")
_BATCH = 64  # chunks embedded and upserted per request


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be integers or UUIDs; a UUID5 of the chunk_id is stable."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


class DenseIndex:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE
            ),
        )
        # Index doc_id so "delete this document's points" doesn't scan everything.
        self.client.create_payload_index(
            self.collection, field_name="doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )

    def upsert_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Replace all of `doc_id`'s points with `chunks`."""
        self._ensure_collection()
        self.client.delete(
            self.collection, points_selector=models.FilterSelector(filter=_doc_filter(doc_id))
        )
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            vectors = self.embedder.embed_passages([c.text for c in batch])
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=asdict(c))
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        """Up to `k` chunks by cosine similarity, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not query.strip() or not self.client.collection_exists(self.collection):
            return []
        response = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(query),
            limit=k,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk(**point.payload), score=point.score, dense_score=point.score)
            for point in response.points
            if point.payload is not None
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def count_document(self, doc_id: str) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(
            self.collection, count_filter=_doc_filter(doc_id), exact=True
        ).count


def make_dense_index(settings: Settings) -> DenseIndex:
    """The real dense index: Qdrant at QDRANT_URL + the EMBEDDING_MODEL from MODELS_DIR,
    with EMBEDDING_QUERY_PREFIX on queries."""
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    # check_compatibility=False: otherwise the client calls the server when it's *created*
    # (and warns if it's down), so building the app would depend on Qdrant being up.
    # Versions are pinned instead: qdrant-client 1.19.x (pyproject) and server 1.19.2 (compose).
    client = QdrantClient(url=settings.qdrant_url, api_key=api_key, check_compatibility=False)
    embedder = FastEmbedEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        cache_dir=str(settings.models_dir),
    )
    return DenseIndex(client, settings.qdrant_collection, embedder)
