"""Application settings, loaded from environment variables and an optional `.env` file.

Every variable is documented in `.env.example`. Field names map to env vars
case-insensitively: `qdrant_url` is read from `QDRANT_URL`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Data ---
    data_raw_dir: Path = Path("data/raw")
    data_index_dir: Path = Path("data/index")
    job_db_path: Path = Path("data/jobs.sqlite")

    # --- Retrieval ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rig_assist_chunks"

    # --- LLM ---
    # "fake" is the default so the app and tests run offline and free.
    llm_provider: Literal["fake", "openai", "azure"] = "fake"
    llm_model: str = ""
    openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None

    # --- Tracing (optional) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @model_validator(mode="after")
    def _check_llm_credentials(self) -> "Settings":
        """Fail at startup, not on the first request, if a real provider is misconfigured."""
        required: dict[str, list[tuple[str, object]]] = {
            "fake": [],
            "openai": [("OPENAI_API_KEY", self.openai_api_key), ("LLM_MODEL", self.llm_model)],
            "azure": [
                ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
                ("AZURE_OPENAI_API_KEY", self.azure_openai_api_key),
                ("LLM_MODEL", self.llm_model),
            ],
        }
        missing = [name for name, value in required[self.llm_provider] if not value]
        if missing:
            raise ValueError(f"LLM_PROVIDER={self.llm_provider} requires: {', '.join(missing)}")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process. Use as a FastAPI dependency; tests can override it."""
    return Settings()
