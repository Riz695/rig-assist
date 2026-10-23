"""FastAPI application. Run locally with: uv run uvicorn rig_assist.api.main:app --reload"""

from fastapi import FastAPI

from rig_assist import __version__
from rig_assist.api.routes import documents, health, search
from rig_assist.config import Settings, get_settings
from rig_assist.ingestion.store import IndexStore
from rig_assist.retrieval.service import SearchIndex


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fresh app. Tests pass their own settings (temp folders) for isolation."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Rig Assist API",
        version=__version__,
        description=(
            "Troubleshooting assistant for oil & gas equipment. "
            "Not a safety-certified system: always follow site procedures."
        ),
    )
    app.state.settings = settings
    app.state.search_index = SearchIndex(IndexStore(settings.data_index_dir))
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    return app


app = create_app()
