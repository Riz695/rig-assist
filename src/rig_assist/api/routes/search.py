from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rig_assist.api.dependencies import get_search_index
from rig_assist.api.schemas import SearchHit, SearchRequest, SearchResponse
from rig_assist.retrieval.service import SearchIndex

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    summary="Search the manuals",
    responses={409: {"description": "Nothing is indexed yet"}},
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
    hits = index.bm25().search(request.query, k=request.top_k)
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
                text=hit.chunk.text,
            )
            for hit in hits
        ],
    )
