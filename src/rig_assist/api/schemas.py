"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rig_assist.agents.schemas import WorkOrder
from rig_assist.jobs import JobStatus
from rig_assist.rag import RefusalReason
from rig_assist.retrieval.service import SearchMode


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
            "examples": [
                {"query": "what causes a kick while drilling", "top_k": 5, "mode": "hybrid"}
            ]
        }
    )

    query: str = Field(min_length=1, max_length=1000, description="A question or keywords.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many chunks to return.")
    mode: SearchMode = Field(
        default="hybrid_rerank",
        description=(
            "Retrieval method. 'hybrid_rerank' (default): hybrid's top 20 reordered by a "
            "cross-encoder; most accurate on the eval set, about 2 s per query on CPU. "
            "'hybrid': BM25 and dense fused by reciprocal rank fusion (~0.1 s). 'dense': "
            "embedding similarity. 'bm25': keyword match, the only mode that works "
            "without Qdrant."
        ),
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
    score: float = Field(
        description=(
            "Higher is more relevant; the scale depends on the mode: BM25 score, cosine "
            "similarity (dense), RRF score (hybrid) or cross-encoder logit (hybrid_rerank)."
        ),
        examples=[6.67],
    )
    dense_score: float | None = Field(
        description=(
            "Cosine similarity from dense retrieval (-1..1), on the same scale in every "
            "dense mode. Null in 'bm25' mode, or when dense retrieval didn't return the chunk."
        ),
        examples=[0.82],
    )
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchHit] = Field(description="Best first. Empty when nothing matches.")


DISCLAIMER = (
    "Rig Assist is not a safety-certified system. Check every answer against the cited "
    "manual pages and follow site procedures and your permit-to-work system."
)


class AskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"question": "What should I do when a kick is detected while drilling?"}]
        }
    )

    question: str = Field(min_length=1, max_length=1000, description="A question in plain words.")
    mode: SearchMode = Field(
        default="hybrid_rerank",
        description="Retrieval method for the context, as in POST /search.",
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="How many chunks the LLM gets as context."
    )


class AskCitation(BaseModel):
    doc_id: str = Field(examples=["drilling-manual"])
    doc_title: str = Field(examples=["Drilling Manual"])
    page: int = Field(description="1-based page in the PDF.", examples=[36])
    snippet: str = Field(
        description="Start of the retrieved text on that page (up to 300 characters).",
        examples=["7.1 Causes of a Kick. A kick is an influx of formation fluid…"],
    )


class AskUsage(BaseModel):
    llm_calls: int = Field(description="0 when the answer was refused before the LLM.")
    model: str | None = Field(description="The model that answered.", examples=["gpt-x"])
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(
        description="Estimate from LLM_*_PRICE_PER_1M; null when prices aren't configured."
    )


class AskLatency(BaseModel):
    retrieval_s: float
    llm_s: float
    total_s: float


class AskResponse(BaseModel):
    question: str
    mode: SearchMode
    top_k: int
    answer: str = Field(
        description="The answer with [doc-id p.N] citations, or 'Not found in the manuals.'"
    )
    found: bool = Field(description="False when the answer is 'Not found in the manuals.'")
    refusal_reason: RefusalReason | None = Field(
        description=(
            "Why the answer is 'not found': no_results (retrieval found nothing), "
            "not_relevant (the chunks didn't pass the relevance check), llm_not_found "
            "(the LLM said the context doesn't answer it), no_citations or "
            "invalid_citation (the answer cited nothing, or a page that wasn't retrieved)."
        ),
        examples=[None],
    )
    citations: list[AskCitation] = Field(
        description="One per cited page, in the order the answer cites them."
    )
    usage: AskUsage
    latency: AskLatency
    disclaimer: str = DISCLAIMER


# --- /diagnose and /jobs -------------------------------------------------------


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "description": "Mud pump 2 is losing discharge pressure and knocking "
                    "since the last liner change.",
                    "equipment_id": "MP-2",
                }
            ]
        },
    )

    description: str = Field(
        min_length=1, max_length=4000, description="The problem, in the technician's words."
    )
    equipment_id: str | None = Field(
        default=None, max_length=64, description="Tag or number of the equipment, if known."
    )


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus = Field(description="Always 'queued' when created.")
    status_url: str = Field(description="Poll this for the result.", examples=["/jobs/3f2a…"])


class JobUsage(BaseModel):
    llm_calls: int = Field(description="Every LLM call, retries included.")
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(
        description="Estimate from LLM_*_PRICE_PER_1M; null when prices aren't configured."
    )


class JobStep(BaseModel):
    node: str = Field(examples=["diagnosis"])
    seconds: float
    input_tokens: int
    output_tokens: int


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: WorkOrder | None = Field(description="The work order, when status is 'done'.")
    error: str | None = Field(description="Why the job failed, when status is 'failed'.")
    trace_url: str | None = Field(description="Link to the run's trace, when tracing is on.")
    usage: JobUsage | None = Field(description="LLM usage, when status is 'done'.")
    steps: list[JobStep] = Field(description="Per node, in the order they ran.")
    created_at: str
    started_at: str | None
    finished_at: str | None
    disclaimer: str = DISCLAIMER
