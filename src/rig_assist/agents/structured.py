"""Ask the LLM for a JSON object and validate it against a Pydantic model.

An invalid reply (not JSON, or failing validation) gets one retry: the reply and what was
wrong with it go back to the model. A second invalid reply raises StructuredOutputError,
carrying the reply and the tokens spent, so the caller can fail the job with a reason.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from rig_assist.llm.json_reply import parse_json_object
from rig_assist.llm.models import LlmClient, Message, Usage

RETRY_PROMPT = (
    "That reply was not valid: {error}. Reply again with only the corrected JSON object, "
    "no other text."
)
MAX_ATTEMPTS = 2  # the first call and one retry

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Structured(Generic[T]):
    value: T
    usage: Usage  # over every attempt
    attempts: int


class StructuredOutputError(ValueError):
    def __init__(self, message: str, reply: str, usage: Usage) -> None:
        super().__init__(message)
        self.reply = reply  # the last invalid reply
        self.usage = usage


def _describe(error: ValidationError) -> str:
    """Pydantic's errors as `field.path: message; ...`, short enough to send back."""
    parts = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "object"
        parts.append(f"{location}: {item['msg']}")
    return "; ".join(parts)


def _parse(text: str, model: type[T]) -> T:
    try:
        return model.model_validate(parse_json_object(text))
    except ValidationError as error:
        raise ValueError(_describe(error)) from error


def complete_structured(
    llm: LlmClient, messages: list[Message], model: type[T], max_tokens: int = 1024
) -> Structured[T]:
    """The reply as a `model`; one retry on an invalid reply. LLM errors propagate."""
    usage = Usage()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = llm.complete(messages, max_tokens=max_tokens)
        usage += response.usage
        try:
            return Structured(_parse(response.text, model), usage, attempt)
        except ValueError as error:
            problem = str(error)
            messages = [
                *messages,
                Message("assistant", response.text),
                Message("user", RETRY_PROMPT.format(error=problem)),
            ]
    raise StructuredOutputError(
        f"invalid {model.__name__} after {MAX_ATTEMPTS} attempts: {problem}",
        reply=response.text,
        usage=usage,
    )
