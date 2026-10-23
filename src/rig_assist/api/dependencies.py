"""Shared objects the routes need, stored on the app by `create_app`."""

from fastapi import HTTPException, Request, status

from rig_assist.config import Settings
from rig_assist.retrieval.service import SearchIndex


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_search_index(request: Request) -> SearchIndex:
    index: SearchIndex = request.app.state.search_index
    return index


def qdrant_unavailable(error: Exception) -> HTTPException:
    """503 for when the vector database can't be reached (BM25 search still works)."""
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"The vector database (Qdrant) is unreachable: start it with "
        f"`docker compose up -d qdrant`, or search with mode 'bm25'. ({error})",
    )
