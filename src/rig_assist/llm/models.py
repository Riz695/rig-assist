"""Types shared by every LLM backend: messages in, text and token usage out."""

from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    """Tokens one or more LLM calls used. Add them up with `sum(usages, Usage())`."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class LlmResponse:
    text: str
    usage: Usage
    model: str  # the model that answered (providers add a version, e.g. "gpt-x-2026-01-01")


class LlmError(Exception):
    """An LLM call failed (HTTP or connection error, cut-off or empty output)."""


class LlmClient(Protocol):
    """Anything that completes a chat. Real: OpenAiLlm (OpenAI or Azure). Tests: FakeLlm."""

    model: str

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse: ...
