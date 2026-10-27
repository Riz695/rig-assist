"""The UI's copy of the OpenAPI schema (ui/openapi.json) must match the API's code."""

from pathlib import Path

from rig_assist.api.openapi_export import openapi_schema, render

COMMITTED = Path(__file__).parent.parent / "ui" / "openapi.json"


def test_the_ui_schema_matches_the_api() -> None:
    assert COMMITTED.read_text(encoding="utf-8") == render(openapi_schema()), (
        "ui/openapi.json is out of date with the API: run `make ui-types` and commit it"
    )


def test_the_schema_has_the_routes_the_ui_calls() -> None:
    paths = openapi_schema()["paths"]
    for route in [
        "/health",
        "/documents",
        "/documents/{doc_id}/pages/{page}",
        "/search",
        "/ask",
        "/diagnose",
        "/jobs/{job_id}",
    ]:
        assert route in paths, route
