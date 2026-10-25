from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client.http.exceptions import ResponseHandlingException

from rig_assist.api.dependencies import (
    get_app_settings,
    get_llm,
    get_search_index,
    qdrant_unavailable,
    searchable_retriever,
)
from rig_assist.api.schemas import AskCitation, AskLatency, AskRequest, AskResponse, AskUsage
from rig_assist.config import Settings
from rig_assist.llm import LlmClient, LlmError
from rig_assist.rag import RagPipeline
from rig_assist.retrieval.service import SearchIndex

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    summary="Answer a question from the manuals, with citations",
    responses={
        409: {"description": "Nothing is indexed yet (or, for dense modes, not in Qdrant)"},
        502: {"description": "The LLM call failed"},
        503: {"description": "Qdrant is unreachable (dense modes only)"},
    },
)
def ask(
    request: AskRequest,
    index: Annotated[SearchIndex, Depends(get_search_index)],
    llm: Annotated[LlmClient, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> AskResponse:
    """Retrieve the best chunks, answer from them with one LLM call, and check that every
    citation points to a retrieved page. Otherwise the answer is 'Not found in the manuals.'
    """
    try:
        retriever = searchable_retriever(index, request.mode)
        pipeline = RagPipeline(retriever, llm, top_k=request.top_k, prices=settings.llm_prices)
        result = pipeline.ask(request.question)
    except ResponseHandlingException as error:  # can't connect to Qdrant
        raise qdrant_unavailable(error) from error
    except LlmError as error:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"The LLM call failed: {error}") from error
    return AskResponse(
        question=request.question,
        mode=request.mode,
        top_k=request.top_k,
        answer=result.answer,
        found=result.found,
        refusal_reason=result.refusal_reason,
        citations=[
            AskCitation(doc_id=c.doc_id, doc_title=c.doc_title, page=c.page, snippet=c.snippet)
            for c in result.citations
        ],
        usage=AskUsage(
            llm_calls=result.llm_calls,
            model=result.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cost_usd=result.cost_usd,
        ),
        latency=AskLatency(
            retrieval_s=result.retrieval_s, llm_s=result.llm_s, total_s=result.total_s
        ),
    )
