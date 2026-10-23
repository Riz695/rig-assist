"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(description="Always 'ok' when the API process is up.")
    version: str = Field(description="Installed rig-assist version.", examples=["0.1.0"])


class DocumentResponse(BaseModel):
    doc_id: str = Field(description="Stable ID used in citations.", examples=["drilling-manual"])
    title: str = Field(examples=["Drilling Manual"])
    filename: str = Field(examples=["Drilling_Manual.pdf"])
    pages: int = Field(description="Pages in the PDF.", examples=[68])
    chunks: int = Field(description="Searchable chunks stored.", examples=[100])
    missing_pages: list[int] = Field(
        description="Pages left without text (scans when OCR is off). Not searchable.",
        examples=[[]],
    )


class DocumentList(BaseModel):
    documents: list[DocumentResponse]


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"query": "what causes a kick while drilling", "top_k": 5, "mode": "bm25"}]
        }
    )

    query: str = Field(min_length=1, max_length=1000, description="A question or keywords.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many chunks to return.")
    mode: Literal["bm25"] = Field(
        default="bm25",
        description="Retrieval method. Only 'bm25' for now; dense and hybrid come in Phase 2.",
    )


class SearchHit(BaseModel):
    chunk_id: str = Field(examples=["drilling-manual::p36::c0"])
    doc_id: str = Field(examples=["drilling-manual"])
    doc_title: str = Field(examples=["Drilling Manual"])
    page: int = Field(
        description="1-based page in the PDF (not the printed number).", examples=[36]
    )
    section: str | None = Field(
        description="Nearest heading, if detected.", examples=["7.1 Causes of a Kick"]
    )
    parser: str = Field(
        description="'pdfplumber' (text layer) or 'olmocr' (OCR).", examples=["pdfplumber"]
    )
    score: float = Field(description="BM25 score: higher is more relevant.", examples=[6.67])
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchHit] = Field(description="Best first. Empty when nothing matches.")
