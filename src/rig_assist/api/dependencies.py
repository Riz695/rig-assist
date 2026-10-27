"""Shared objects the routes need, stored on the app by `create_app`."""

from fastapi import HTTPException, Request, status

from rig_assist.config import Settings
from rig_assist.jobs import JobStore
from rig_assist.llm import LlmClient
from rig_assist.retrieval.models import Retriever
from rig_assist.retrieval.service import DENSE_MODES, SearchIndex, SearchMode
from rig_assist.tracing import Tracer


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_search_index(request: Request) -> SearchIndex:
    index: SearchIndex = request.app.state.search_index
    return index


def get_job_store(request: Request) -> JobStore:
    jobs: JobStore = request.app.state.jobs
    return jobs


def get_tracer(request: Request) -> Tracer:
    tracer: Tracer = request.app.state.tracer
    return tracer


def get_llm(request: Request) -> LlmClient:
    llm: LlmClient = request.app.state.llm
    return llm


def qdrant_unavailable(error: Exception) -> HTTPException:
    """503 for when the vector database can't be reached (BM25 search still works)."""
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The vector database (Qdrant) is unreachable: start it with "
        f"`docker compose up -d qdrant`, or search with mode 'bm25'. ({error})",
    )


def searchable_retriever(index: SearchIndex, mode: SearchMode) -> Retriever:
    """The retriever for `mode`, or a 409 saying what to ingest first.

    Checking the dense index talks to Qdrant: callers catch ResponseHandlingException
    (Qdrant down) around this and the search itself, and return `qdrant_unavailable`.
    """
    if index.is_empty():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No documents are indexed yet: upload one with POST /documents or run `make ingest`.",
        )
    if mode in DENSE_MODES and index.dense.count() == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Documents are indexed for BM25 but not in the vector database: run "
            "`make ingest` (without --skip-dense), or search with mode 'bm25'.",
        )
    return index.retriever(mode)
