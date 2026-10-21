"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(description="Always 'ok' when the API process is up.")
    version: str = Field(description="Installed rig-assist version.", examples=["0.1.0"])
