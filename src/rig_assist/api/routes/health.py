from fastapi import APIRouter

from rig_assist import __version__
from rig_assist.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health() -> HealthResponse:
    """Return 200 while the process is running. Used by Docker and Azure health probes."""
    return HealthResponse(status="ok", version=__version__)
