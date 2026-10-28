from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client.http.exceptions import ResponseHandlingException

from rig_assist import __version__
from rig_assist.api.dependencies import get_search_index, qdrant_unavailable
from rig_assist.api.schemas import HealthResponse, ReadyResponse
from rig_assist.retrieval.service import SearchIndex

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health() -> HealthResponse:
    """Return 200 while the process is running. Used by Docker and Azure health probes."""
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/ready",
    summary="Readiness check",
    responses={503: {"description": "Not ready: nothing indexed, or Qdrant down or empty."}},
)
def ready(index: Annotated[SearchIndex, Depends(get_search_index)]) -> ReadyResponse:
    """Return 200 only when the API can answer the default search mode: the BM25 index is
    loaded and Qdrant answers with vectors in the collection. The platform's readiness probe
    keeps traffic away until then (e.g. while Qdrant restores its snapshot)."""
    if index.is_empty():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No documents are indexed yet: upload one with POST /documents or run `make ingest`.",
        )
    try:
        vectors = index.dense.count()
    except ResponseHandlingException as error:  # can't connect to Qdrant
        raise qdrant_unavailable(error) from error
    if vectors == 0:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Qdrant has no vectors in the collection '{index.dense.collection}': run "
            "`make ingest`, or restore the collection's snapshot.",
        )
    return ReadyResponse(status="ready", chunks=len(index.bm25()), vectors=vectors)
