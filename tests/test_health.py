import pytest
from fastapi.testclient import TestClient

import rig_assist
from rig_assist.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_ok_and_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": rig_assist.__version__}


def test_health_is_documented_in_openapi(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert schema["info"]["title"] == "Rig Assist API"
    assert schema["info"]["version"] == rig_assist.__version__
    assert "get" in schema["paths"]["/health"]
    assert "HealthResponse" in schema["components"]["schemas"]


def test_swagger_ui_is_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/does-not-exist").status_code == 404
