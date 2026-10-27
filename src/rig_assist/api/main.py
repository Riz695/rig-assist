"""FastAPI application. Run locally with: uv run uvicorn rig_assist.api.main:app --reload"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rig_assist import __version__
from rig_assist.api.routes import ask, diagnose, documents, health, search
from rig_assist.config import Settings, get_settings
from rig_assist.ingestion.store import IndexStore
from rig_assist.jobs import JobStore
from rig_assist.llm import LlmClient, make_llm
from rig_assist.retrieval.dense import DenseIndex, make_dense_index
from rig_assist.retrieval.rerank import CrossEncoder, FastEmbedCrossEncoder
from rig_assist.retrieval.service import SearchIndex
from rig_assist.tracing import Tracer, make_tracer


def create_app(
    settings: Settings | None = None,
    dense: DenseIndex | None = None,
    cross_encoder: CrossEncoder | None = None,
    llm: LlmClient | None = None,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Build a fresh app. Tests pass their own settings (temp folders) and fakes for the
    dense index, cross-encoder and LLM (default: the one LLM_PROVIDER names). The real
    ones connect and load models on first use, so the app starts even when Qdrant is
    down (dense searches then return 503).
    """
    settings = settings or get_settings()
    jobs = JobStore(settings.job_db_path)
    tracer = tracer or make_tracer(settings)  # off unless both Langfuse keys are set

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Jobs run inside this process: any left queued or running died with the last one.
        jobs.fail_unfinished()
        yield
        tracer.flush()  # send the spans still buffered

    if dense is None:
        dense = make_dense_index(settings)
    if cross_encoder is None:
        cross_encoder = FastEmbedCrossEncoder(
            settings.reranker_model, cache_dir=str(settings.models_dir)
        )
    app = FastAPI(
        lifespan=lifespan,
        title="Rig Assist API",
        version=__version__,
        description=(
            "Troubleshooting assistant for oil & gas equipment. "
            "Not a safety-certified system: always follow site procedures."
        ),
    )
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.tracer = tracer
    app.state.llm = llm if llm is not None else make_llm(settings)
    app.state.search_index = SearchIndex(
        IndexStore(settings.data_index_dir), dense=dense, cross_encoder=cross_encoder
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(diagnose.router)
    return app


app = create_app()
