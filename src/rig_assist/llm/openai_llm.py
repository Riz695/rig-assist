"""OpenAI and Azure OpenAI through one client.

Azure's v1 API (GA since August 2025) accepts the plain `openai.OpenAI` client: point
`base_url` at `https://<resource>.openai.azure.com/openai/v1/`, pass the Azure key, and
use the deployment name as the model. No `AzureOpenAI` class and no `api-version`.
Any other OpenAI-compatible server (vLLM, a gateway) works the same way.
"""

import httpx2
import openai

from rig_assist.llm.models import LlmError, LlmResponse, Message, Usage


def azure_base_url(endpoint: str) -> str:
    """https://<resource>.openai.azure.com → its v1 API base URL."""
    return endpoint.rstrip("/") + "/openai/v1/"


class OpenAiLlm:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,  # None = api.openai.com
        timeout: float = 60.0,
        max_retries: int = 2,  # the SDK retries 429/5xx/connection errors with backoff
        http_client: httpx2.Client | None = None,  # tests pass a mock transport here
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
                # Not max_tokens: reasoning models only accept max_completion_tokens.
                max_completion_tokens=max_tokens,
            )
        except openai.APIStatusError as error:
            raise LlmError(f"HTTP {error.status_code}: {error.message}") from error
        except openai.APIError as error:  # connection errors and timeouts
            raise LlmError(f"{type(error).__name__}: {error}") from error

        choice = response.choices[0]
        if choice.finish_reason != "stop":
            # "length": cut off at max_tokens; "content_filter": blocked. Neither is an answer.
            raise LlmError(f"incomplete output (finish_reason={choice.finish_reason})")
        if not choice.message.content:
            raise LlmError("empty output")
        usage = response.usage
        return LlmResponse(
            text=choice.message.content,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            model=response.model,
        )
