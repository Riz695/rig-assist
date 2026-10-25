"""Provider-agnostic LLM access: one `LlmClient` interface, chosen by LLM_PROVIDER."""

from typing import TYPE_CHECKING

from rig_assist.llm.fake import FAKE_REPLY, FakeLlm
from rig_assist.llm.models import LlmClient, LlmError, LlmResponse, Message, Usage
from rig_assist.llm.openai_llm import OpenAiLlm, azure_base_url
from rig_assist.llm.usage import Prices, cost_usd

if TYPE_CHECKING:  # config imports llm.usage, so a runtime import here would be circular
    from rig_assist.config import Settings


def make_llm(settings: "Settings") -> LlmClient:
    """The client LLM_PROVIDER asks for. Settings already checked the required variables."""
    if settings.llm_provider == "openai":
        assert settings.openai_api_key is not None
        return OpenAiLlm(
            model=settings.llm_model, api_key=settings.openai_api_key.get_secret_value()
        )
    if settings.llm_provider == "azure":
        assert settings.azure_openai_endpoint and settings.azure_openai_api_key
        return OpenAiLlm(
            model=settings.llm_model,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            base_url=azure_base_url(settings.azure_openai_endpoint),
        )
    return FakeLlm()


__all__ = [
    "FAKE_REPLY",
    "FakeLlm",
    "LlmClient",
    "LlmError",
    "LlmResponse",
    "Message",
    "OpenAiLlm",
    "Prices",
    "Usage",
    "azure_base_url",
    "cost_usd",
    "make_llm",
]
