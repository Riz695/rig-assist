"""A scripted LLM for tests and offline runs: no network, no cost, same output every time."""

import re

from rig_assist.llm.models import LlmError, LlmResponse, Message, Usage

# What LLM_PROVIDER=fake answers when no script is given (the app's default).
FAKE_REPLY = (
    "This is a placeholder answer from the fake LLM (LLM_PROVIDER=fake). "
    "Set LLM_PROVIDER to openai or azure for real answers."
)
# A context label such as "[bop-manual p.12]", the format the RAG prompt uses.
_LABEL = re.compile(r"\[[a-z0-9][a-z0-9-]* p\.\d+\]")


def _default_reply(messages: list[Message]) -> str:
    """FAKE_REPLY, citing the first context label in the last message if there is one."""
    match = _LABEL.search(messages[-1].content) if messages else None
    return f"{FAKE_REPLY} {match.group()}" if match else FAKE_REPLY


def _words(text: str) -> int:
    return len(text.split())


class FakeLlm:
    """Returns `replies` in order and records every prompt.

    Without `replies` it always answers FAKE_REPLY, citing the prompt's first context
    label, so the app's default (LLM_PROVIDER=fake) returns a checkable cited answer.

    Token usage is a word count, so tests can compute it by hand.
    """

    def __init__(self, replies: list[str] | None = None, model: str = "fake") -> None:
        self.model = model
        self._replies = list(replies) if replies is not None else None
        self.calls: list[list[Message]] = []
        self.max_tokens: list[int] = []  # the token limit each call asked for

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse:
        self.calls.append(list(messages))
        self.max_tokens.append(max_tokens)
        if self._replies is None:
            text = _default_reply(messages)
        elif self._replies:
            text = self._replies.pop(0)
        else:
            raise LlmError(f"FakeLlm: no scripted reply left for call {len(self.calls)}")
        usage = Usage(
            input_tokens=sum(_words(message.content) for message in messages),
            output_tokens=_words(text),
        )
        return LlmResponse(text=text, usage=usage, model=self.model)
