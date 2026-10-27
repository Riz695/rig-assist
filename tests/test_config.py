from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rig_assist.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any real settings from the shell so tests only see what they set."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def make_settings(**overrides: Any) -> Settings:
    # _env_file=None: never read a developer's real .env during tests.
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_defaults_run_offline() -> None:
    settings = make_settings()

    assert settings.llm_provider == "fake"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.data_raw_dir == Path("data/raw")
    assert settings.langfuse_enabled is False
    assert settings.embedding_query_prefix == ""  # v1 and fastembed use none
    # Best reranker on the eval set and ~4x faster than bge-reranker-base on CPU.
    assert settings.reranker_model == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("DATA_INDEX_DIR", "/tmp/index")

    settings = make_settings()

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.data_index_dir == Path("/tmp/index")


def test_reads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("QDRANT_COLLECTION=from_file\n")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.qdrant_collection == "from_file"


def test_unknown_llm_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(llm_provider="cohere")


def test_openai_provider_requires_api_key_and_model() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        make_settings(llm_provider="openai", llm_model="some-model")
    with pytest.raises(ValidationError, match="LLM_MODEL"):
        make_settings(llm_provider="openai", openai_api_key="sk-test")


def test_gemini_provider_requires_api_key_and_model() -> None:
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        make_settings(llm_provider="gemini", llm_model="gemini-test")
    with pytest.raises(ValidationError, match="LLM_MODEL"):
        make_settings(llm_provider="gemini", gemini_api_key="g-test")


def test_openai_compatible_provider_requires_a_base_url_and_model() -> None:
    with pytest.raises(ValidationError, match="LLM_BASE_URL"):
        make_settings(llm_provider="openai_compatible", llm_model="m")
    with pytest.raises(ValidationError, match="LLM_MODEL"):
        make_settings(llm_provider="openai_compatible", llm_base_url="http://pod:8000/v1")
    # The key is optional: a self-hosted vLLM server may run without one.
    settings = make_settings(
        llm_provider="openai_compatible", llm_model="m", llm_base_url="http://pod:8000/v1"
    )
    assert settings.llm_api_key is None


def test_llm_retries_default_to_two_and_cannot_be_negative() -> None:
    assert make_settings().llm_max_retries == 2
    assert make_settings(llm_max_retries=6).llm_max_retries == 6
    with pytest.raises(ValidationError):
        make_settings(llm_max_retries=-1)


def test_azure_provider_requires_endpoint_and_key() -> None:
    with pytest.raises(ValidationError, match="AZURE_OPENAI_ENDPOINT"):
        make_settings(llm_provider="azure", llm_model="dep", azure_openai_api_key="k")
    with pytest.raises(ValidationError, match="AZURE_OPENAI_API_KEY"):
        make_settings(
            llm_provider="azure", llm_model="dep", azure_openai_endpoint="https://x.azure.com"
        )


def test_secrets_are_hidden_when_printed() -> None:
    settings = make_settings(llm_provider="openai", llm_model="m", openai_api_key="sk-secret")

    assert "sk-secret" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-secret"


def test_langfuse_enabled_only_with_both_keys() -> None:
    assert make_settings().langfuse_enabled is False
    assert make_settings(langfuse_public_key="", langfuse_secret_key="").langfuse_enabled is False
    both = make_settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    assert both.langfuse_enabled is True


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_ocr_is_off_by_default() -> None:
    settings = make_settings()
    assert settings.ocr_mode == "off"
    assert settings.data_parsed_dir == Path("data/parsed")


def test_ocr_mode_requires_a_base_url() -> None:
    with pytest.raises(ValidationError, match="OCR_BASE_URL"):
        make_settings(ocr_mode="missing")


def test_ocr_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_MODE", "all")
    monkeypatch.setenv("OCR_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("OCR_API_KEY", "ocr-secret")

    settings = make_settings()

    assert settings.ocr_mode == "all"
    assert settings.ocr_model == "allenai/olmOCR-2-7B-1025"
    assert settings.ocr_api_key is not None
    assert "ocr-secret" not in repr(settings)


def test_unknown_ocr_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(ocr_mode="sometimes")


def test_ocr_concurrency_defaults_to_one_and_must_be_positive() -> None:
    assert make_settings().ocr_concurrency == 1
    assert make_settings(ocr_concurrency=8).ocr_concurrency == 8
    with pytest.raises(ValidationError):
        make_settings(ocr_concurrency=0)


def test_models_are_cached_under_data_by_default() -> None:
    assert make_settings().models_dir == Path("data/models")
