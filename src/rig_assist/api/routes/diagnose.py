from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from qdrant_client.http.exceptions import ResponseHandlingException

from rig_assist.agents.graph import TroubleshootingGraph
from rig_assist.api.dependencies import (
    get_app_settings,
    get_job_store,
    get_llm,
    get_search_index,
    get_tracer,
    qdrant_unavailable,
    searchable_retriever,
)
from rig_assist.api.schemas import DiagnoseRequest, JobCreated, JobResponse
from rig_assist.config import Settings
from rig_assist.jobs import JobStore, run_job
from rig_assist.llm import LlmClient
from rig_assist.retrieval.service import SearchIndex, SearchMode
from rig_assist.tracing import Tracer

router = APIRouter(tags=["diagnose"])

DIAGNOSE_MODE: SearchMode = "hybrid_rerank"  # the served retrieval, as /ask's default


@router.post(
    "/diagnose",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a troubleshooting job; poll GET /jobs/{job_id} for the work order",
    responses={
        409: {"description": "Nothing is indexed yet (or not in Qdrant)"},
        503: {"description": "Qdrant is unreachable"},
    },
)
def diagnose(
    request: DiagnoseRequest,
    background: BackgroundTasks,
    index: Annotated[SearchIndex, Depends(get_search_index)],
    llm: Annotated[LlmClient, Depends(get_llm)],
    jobs: Annotated[JobStore, Depends(get_job_store)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    tracer: Annotated[Tracer, Depends(get_tracer)],
) -> JobCreated:
    """Triage → retrieval → diagnosis → safety → work order, run in the background.
    The safety review overrides the diagnosis: blocked actions are never recommended.
    """
    try:
        retriever = searchable_retriever(index, DIAGNOSE_MODE)
    except ResponseHandlingException as error:  # can't connect to Qdrant
        raise qdrant_unavailable(error) from error
    job = jobs.create(request.model_dump())
    graph = TroubleshootingGraph(llm, retriever, tracer=tracer)
    background.add_task(run_job, jobs, graph, job.id, settings.llm_prices, tracer)
    return JobCreated(job_id=job.id, status=job.status, status_url=f"/jobs/{job.id}")


@router.get(
    "/jobs/{job_id}",
    summary="Status and result of a /diagnose job",
    responses={404: {"description": "No job with this ID"}},
)
def get_job(job_id: str, jobs: Annotated[JobStore, Depends(get_job_store)]) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job with ID {job_id!r}.")
    return JobResponse.model_validate(
        {
            "job_id": job.id,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "trace_url": job.trace_url,
            "usage": job.usage,
            "steps": job.steps,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }
    )
