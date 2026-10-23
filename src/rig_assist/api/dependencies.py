"""Shared objects the routes need, stored on the app by `create_app`."""

from fastapi import Request

from rig_assist.config import Settings
from rig_assist.retrieval.service import SearchIndex


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_search_index(request: Request) -> SearchIndex:
    index: SearchIndex = request.app.state.search_index
    return index
