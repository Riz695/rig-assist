"""Application settings, loaded from environment variables and an optional `.env` file.

Every variable is documented in `.env.example`. Field names map to env vars
case-insensitively: `qdrant_url` is read from `QDRANT_URL`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rig_assist.llm.usage import Prices


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Data ---
    data_raw_dir: Path = Path("data/raw")
    data_index_dir: Path = Path("data/index")
    data_parsed_dir: Path = Path("data/parsed")  # OCR page cache
    models_dir: Path = Path("data/models")  # downloaded embedding/reranker models
    job_db_path: Path = Path("data/jobs.sqlite")

    # --- OCR (olmOCR 2 via any OpenAI-compatible endpoint) ---
    # "off" by default: ingestion runs offline and free; scanned pages are reported.
    ocr_mode: Literal["off", "missing", "all"] = "off"
    ocr_base_url: str | None = None
    ocr_api_key: SecretStr | None = None
    ocr_model: str = "allenai/olmOCR-2-7B-1025"
    # Pages OCR'd at once. 1 is gentle on hosted APIs; ~8 suits a dedicated vLLM GPU,
    # which batches concurrent requests at almost the same per-request speed.
    ocr_concurrency: int = Field(default=1, ge=1)

    # --- Retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Prepended to search queries only (not chunks). "" = none, as v1 and fastembed.
    embedding_query_prefix: str = ""
    # Best on the eval set (eval/results/retrieval.md) and ~4x faster on CPU than
    # BAAI/bge-reranker-base (~2 s vs ~10 s per query for 20 candidates).
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rig_assist_chunks"

    # --- Experiment tracking (eval runs only; MLflow is in the `eval` dependency group) ---
    mlflow_tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    mlflow_artifact_dir: Path = Path("data/mlflow/artifacts")

    # --- LLM ---
    # "fake" is the default so the app and tests run offline and free.
    llm_provider: Literal["fake", "openai", "azure", "gemini", "openai_compatible"] = "fake"
    llm_model: str = ""
    openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None  # Google AI Studio key
    # Any other OpenAI-compatible server, e.g. vLLM on a rented GPU (docs/llm-gpu.md).
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None  # optional: a self-hosted server may not need one
    # The openai SDK retries 429 (rate limit), 5xx and connection errors with backoff.
    llm_max_retries: int = Field(default=2, ge=0)
    # "off" asks a vLLM server running a thinking model (Qwen3) not to think: faster, and
    # hidden reasoning can't use up the output limit. openai_compatible only.
    llm_thinking: Literal["default", "off"] = "default"
    # USD per 1M tokens, for the cost estimate in responses. Unset = cost unknown (null).
    llm_input_price_per_1m: float | None = Field(default=None, ge=0)
    llm_output_price_per_1m: float | None = Field(default=None, ge=0)

    # --- Tracing (optional: Langfuse, on when both keys are set) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # EU cloud by default; https://us.cloud.langfuse.com for the US region.
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def llm_prices(self) -> "Prices | None":
        if self.llm_input_price_per_1m is None or self.llm_output_price_per_1m is None:
            return None
        return Prices(self.llm_input_price_per_1m, self.llm_output_price_per_1m)

    @model_validator(mode="after")
    def _check_credentials(self) -> "Settings":
        """Fail at startup, not on the first request, if a real provider is misconfigured."""
        required: dict[str, list[tuple[str, object]]] = {
            "fake": [],
            "openai": [("OPENAI_API_KEY", self.openai_api_key), ("LLM_MODEL", self.llm_model)],
            "azure": [
                ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
                ("LLM_MODEL", self.llm_model),
            ],
            "gemini": [("GEMINI_API_KEY", self.gemini_api_key), ("LLM_MODEL", self.llm_model)],
            "openai_compatible": [
                ("LLM_BASE_URL", self.llm_base_url),
                ("LLM_MODEL", self.llm_model),
            ],
        }
        missing = [name for name, value in required[self.llm_provider] if not value]
        if missing:
            raise ValueError(f"LLM_PROVIDER={self.llm_provider} requires: {', '.join(missing)}")
        prices = {
            "LLM_INPUT_PRICE_PER_1M": self.llm_input_price_per_1m,
            "LLM_OUTPUT_PRICE_PER_1M": self.llm_output_price_per_1m,
        }
        unset = [name for name, value in prices.items() if value is None]
        if len(unset) == 1:
            raise ValueError(f"Set both LLM prices or neither; missing: {unset[0]}")
        if self.llm_thinking == "off" and self.llm_provider != "openai_compatible":
            raise ValueError("LLM_THINKING=off is only for LLM_PROVIDER=openai_compatible (vLLM)")
        keys = {
            "LANGFUSE_PUBLIC_KEY": self.langfuse_public_key,
            "LANGFUSE_SECRET_KEY": self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value(),
        }
        missing_keys = [name for name, value in keys.items() if not value]
        if len(missing_keys) == 1:
            raise ValueError(f"Set both Langfuse keys or neither; missing: {missing_keys[0]}")
        # The API key stays optional: a self-hosted vLLM server may not need one.
        if self.ocr_mode != "off" and not self.ocr_base_url:
            raise ValueError(f"OCR_MODE={self.ocr_mode} requires: OCR_BASE_URL")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process. Use as a FastAPI dependency; tests can override it."""
    return Settings()
