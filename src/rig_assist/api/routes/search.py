from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client.http.exceptions import ResponseHandlingException

from rig_assist.api.dependencies import get_search_index, qdrant_unavailable
from rig_assist.api.schemas import SearchHit, SearchRequest, SearchResponse
from rig_assist.retrieval.service import DENSE_MODES, SearchIndex

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    summary="Search the manuals",
    responses={
        409: {"description": "Nothing is indexed yet (or, for dense modes, not in Qdrant)"},
        503: {"description": "Qdrant is unreachable (dense modes only)"},
    },
)
def search(
    request: SearchRequest, index: Annotated[SearchIndex, Depends(get_search_index)]
) -> SearchResponse:
    """Return the chunks that best match the query, with document, page and score."""
    if index.is_empty():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No documents are indexed yet: upload one with POST /documents or run `make ingest`.",
        )
    try:
        if request.mode in DENSE_MODES and index.dense.count() == 0:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Documents are indexed for BM25 but not in the vector database: run "
                "`make ingest` (without --skip-dense), or search with mode 'bm25'.",
            )
        hits = index.retriever(request.mode).search(request.query, k=request.top_k)
    except ResponseHandlingException as error:  # can't connect to Qdrant
        raise qdrant_unavailable(error) from error
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=[
            SearchHit(
                chunk_id=hit.chunk.chunk_id,
                doc_id=hit.chunk.doc_id,
                doc_title=hit.chunk.doc_title,
                page=hit.chunk.page,
                section=hit.chunk.section,
                parser=hit.chunk.parser,
                score=hit.score,
                dense_score=hit.dense_score,
                text=hit.chunk.text,
            )
            for hit in hits
        ],
    )
