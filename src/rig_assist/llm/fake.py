"""A scripted LLM for tests and offline runs: no network, no cost, same output every time."""

from rig_assist.llm.models import LlmError, LlmResponse, Message, Usage

# What LLM_PROVIDER=fake answers when no script is given (the app's default).
FAKE_REPLY = (
    "This is a placeholder answer from the fake LLM (LLM_PROVIDER=fake). "
    "Set LLM_PROVIDER to openai or azure for real answers. [1]"
)


def _words(text: str) -> int:
    return len(text.split())


class FakeLlm:
    """Returns `replies` in order (or FAKE_REPLY forever if none) and records every prompt.

    Token usage is a word count, so tests can compute it by hand.
    """

    def __init__(self, replies: list[str] | None = None, model: str = "fake") -> None:
        self.model = model
        self._replies = list(replies) if replies is not None else None
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], max_tokens: int = 1024) -> LlmResponse:
        self.calls.append(list(messages))
        if self._replies is None:
            text = FAKE_REPLY
        elif self._replies:
            text = self._replies.pop(0)
        else:
            raise LlmError(f"FakeLlm: no scripted reply left for call {len(self.calls)}")
        usage = Usage(
            input_tokens=sum(_words(message.content) for message in messages),
            output_tokens=_words(text),
        )
        return LlmResponse(text=text, usage=usage, model=self.model)
