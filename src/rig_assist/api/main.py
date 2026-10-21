"""FastAPI application. Run locally with: uv run uvicorn rig_assist.api.main:app --reload"""

from fastapi import FastAPI

from rig_assist import __version__
from rig_assist.api.routes import health


def create_app() -> FastAPI:
    """Build a fresh app. Tests call this to get an isolated instance."""
    app = FastAPI(
        title="Rig Assist API",
        version=__version__,
        description=(
            "Troubleshooting assistant for oil & gas equipment. "
            "Not a safety-certified system: always follow site procedures."
        ),
    )
    app.include_router(health.router)
    return app


app = create_app()
