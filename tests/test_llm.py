import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import DefaultHttpx2Client

from rig_assist.config import Settings
from rig_assist.llm import (
    FAKE_REPLY,
    GEMINI_BASE_URL,
    FakeLlm,
    LlmError,
    LlmResponse,
    Message,
    OpenAiLlm,
    Prices,
    Usage,
    azure_base_url,
    cost_usd,
    make_llm,
)


def make_settings(**overrides: Any) -> Settings:
    """Settings from arguments only, ignoring the developer's .env."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


PROMPT = [
    Message("system", "Answer from the manuals only."),
    Message("user", "How often is the BOP tested?"),
]


# --- FakeLlm ------------------------------------------------------------


def test_fake_returns_scripted_replies_in_order() -> None:
    llm = FakeLlm(replies=["first", "second"])
    assert llm.complete(PROMPT).text == "first"
    assert llm.complete(PROMPT).text == "second"


def test_fake_records_every_prompt() -> None:
    llm = FakeLlm(replies=["a", "b"])
    llm.complete(PROMPT)
    llm.complete([Message("user", "Second question")])
    assert llm.calls == [PROMPT, [Message("user", "Second question")]]


def test_fake_raises_when_the_script_runs_out() -> None:
    llm = FakeLlm(replies=["only one"])
    llm.complete(PROMPT)
    with pytest.raises(LlmError, match="no scripted reply left"):
        llm.complete(PROMPT)


def test_fake_without_a_script_always_gives_the_default_reply() -> None:
    llm = FakeLlm()
    assert [llm.complete(PROMPT).text for _ in range(3)] == [FAKE_REPLY] * 3


def test_fake_default_reply_cites_the_first_label_in_the_prompt() -> None:
    # So LLM_PROVIDER=fake gives /ask an answer with a citation it can check.
    prompt = [
        Message("system", "Cite like [example p.7]."),
        Message("user", "[bop-manual p.12] Test weekly.\n\n[safety p.3] Wear PPE.\n\nQ?"),
    ]
    assert FakeLlm().complete(prompt).text == f"{FAKE_REPLY} [bop-manual p.12]"
    assert "[" not in FAKE_REPLY


def test_fake_is_deterministic_across_instances() -> None:
    first = FakeLlm(replies=["same"]).complete(PROMPT)
    second = FakeLlm(replies=["same"]).complete(PROMPT)
    assert first == second


def test_fake_records_the_token_limit_of_each_call() -> None:
    llm = FakeLlm(replies=["a", "b"])
    llm.complete(PROMPT)
    llm.complete(PROMPT, max_tokens=4096)
    assert llm.max_tokens == [1024, 4096]


def test_fake_usage_counts_words_of_the_prompt_and_the_reply() -> None:
    response = FakeLlm(replies=["Every 14 days."]).complete(PROMPT)
    # "Answer from the manuals only." (5) + "How often is the BOP tested?" (6)
    assert response.usage == Usage(input_tokens=11, output_tokens=3)
    assert response.model == "fake"


# --- Usage and cost -----------------------------------------------------


def test_usages_add_up() -> None:
    total = sum([Usage(100, 20), Usage(50, 5)], Usage())
    assert total == Usage(input_tokens=150, output_tokens=25)


def test_cost_uses_prices_per_million_tokens() -> None:
    prices = Prices(input_per_1m=0.40, output_per_1m=1.60)
    # 2,000 × 0.40 / 1e6 + 500 × 1.60 / 1e6 = 0.0008 + 0.0008
    assert cost_usd(Usage(2_000, 500), prices) == pytest.approx(0.0016)


def test_cost_is_zero_for_no_tokens() -> None:
    assert cost_usd(Usage(), Prices(0.40, 1.60)) == 0.0


def test_cost_is_unknown_without_prices() -> None:
    assert cost_usd(Usage(2_000, 500), None) is None


# --- OpenAiLlm (real client, fake HTTP) ---------------------------------


def completion(
    content: str | None, finish_reason: str = "stop", total_tokens: int = 876
) -> dict[str, Any]:
    """A chat-completions response body, as OpenAI and Azure's v1 endpoint return it."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test-2026-01-01",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 812, "completion_tokens": 64, "total_tokens": total_tokens},
    }


def make_openai(
    respond: Callable[[httpx2.Request], httpx2.Response],
    base_url: str = "https://llm.example/v1",
) -> tuple[OpenAiLlm, list[httpx2.Request]]:
    """A real OpenAiLlm whose HTTP calls go to `respond` instead of the network."""
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return respond(request)

    llm = OpenAiLlm(
        model="gpt-test",
        api_key="test-key",
        base_url=base_url,
        max_retries=0,
        http_client=DefaultHttpx2Client(transport=httpx2.MockTransport(handler)),
    )
    return llm, sent


def test_openai_sends_the_model_messages_and_token_limit() -> None:
    llm, sent = make_openai(lambda _: httpx2.Response(200, json=completion("Every 14 days.")))
    llm.complete(PROMPT, max_tokens=300)

    body = json.loads(sent[0].content)
    assert sent[0].url == "https://llm.example/v1/chat/completions"
    assert body["model"] == "gpt-test"
    assert body["messages"] == [
        {"role": "system", "content": "Answer from the manuals only."},
        {"role": "user", "content": "How often is the BOP tested?"},
    ]
    # max_completion_tokens, not max_tokens: reasoning models reject the old name.
    assert body["max_completion_tokens"] == 300
    assert "max_tokens" not in body


def test_openai_returns_text_usage_and_the_served_model() -> None:
    llm, _ = make_openai(lambda _: httpx2.Response(200, json=completion("Every 14 days.")))
    assert llm.complete(PROMPT) == LlmResponse(
        text="Every 14 days.",
        usage=Usage(input_tokens=812, output_tokens=64),
        model="gpt-test-2026-01-01",
    )


def test_hidden_thinking_tokens_count_as_output() -> None:
    # Gemini leaves its thinking out of completion_tokens and counts it only in the total
    # (seen 2026-10-25: prompt 2092, completion 193, total 2893). It's billed as output.
    llm, _ = make_openai(lambda _: httpx2.Response(200, json=completion("ok", total_tokens=1500)))
    assert llm.complete(PROMPT).usage == Usage(input_tokens=812, output_tokens=1500 - 812)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_openai_incomplete_output_is_an_error(finish_reason: str) -> None:
    llm, _ = make_openai(lambda _: httpx2.Response(200, json=completion("Every", finish_reason)))
    with pytest.raises(LlmError, match=finish_reason):
        llm.complete(PROMPT)


def test_openai_empty_content_is_an_error() -> None:
    llm, _ = make_openai(lambda _: httpx2.Response(200, json=completion(None)))
    with pytest.raises(LlmError, match="empty"):
        llm.complete(PROMPT)


def test_openai_http_errors_surface_as_llm_errors() -> None:
    llm, _ = make_openai(lambda _: httpx2.Response(500, json={"error": {"message": "boom"}}))
    with pytest.raises(LlmError, match="HTTP 500"):
        llm.complete(PROMPT)


def test_openai_connection_errors_surface_as_llm_errors() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    llm, _ = make_openai(refuse)
    with pytest.raises(LlmError):
        llm.complete(PROMPT)


# --- Azure and the factory ----------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    ["https://my-rig.openai.azure.com", "https://my-rig.openai.azure.com/"],
)
def test_azure_base_url_is_the_v1_endpoint(endpoint: str) -> None:
    assert azure_base_url(endpoint) == "https://my-rig.openai.azure.com/openai/v1/"


def test_azure_requests_go_to_the_v1_path() -> None:
    llm, sent = make_openai(
        lambda _: httpx2.Response(200, json=completion("ok")),
        base_url=azure_base_url("https://my-rig.openai.azure.com"),
    )
    llm.complete(PROMPT)
    assert sent[0].url == "https://my-rig.openai.azure.com/openai/v1/chat/completions"


def test_make_llm_defaults_to_the_fake() -> None:
    assert isinstance(make_llm(make_settings()), FakeLlm)


def test_make_llm_builds_an_openai_client() -> None:
    settings = make_settings(llm_provider="openai", llm_model="gpt-test", openai_api_key="sk-test")
    llm = make_llm(settings)
    assert isinstance(llm, OpenAiLlm)
    assert llm.model == "gpt-test"
    assert llm.base_url is None


def test_make_llm_builds_an_azure_client_on_the_v1_endpoint() -> None:
    settings = make_settings(
        llm_provider="azure",
        llm_model="my-deployment",
        azure_openai_endpoint="https://my-rig.openai.azure.com",
        azure_openai_api_key="azure-test",
    )
    llm = make_llm(settings)
    assert isinstance(llm, OpenAiLlm)
    assert llm.model == "my-deployment"
    assert llm.base_url == "https://my-rig.openai.azure.com/openai/v1/"


def test_make_llm_builds_a_gemini_client_on_its_openai_compatible_endpoint() -> None:
    settings = make_settings(
        llm_provider="gemini", llm_model="gemini-test", gemini_api_key="g-test", llm_max_retries=5
    )
    llm = make_llm(settings)
    assert isinstance(llm, OpenAiLlm)
    assert llm.model == "gemini-test"
    assert llm.base_url == GEMINI_BASE_URL
    assert llm.max_retries == 5


def test_gemini_base_url_is_googles_openai_compatibility_endpoint() -> None:
    assert GEMINI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_gemini_requests_go_to_the_chat_completions_path() -> None:
    llm, sent = make_openai(
        lambda _: httpx2.Response(200, json=completion("ok")), base_url=GEMINI_BASE_URL
    )
    llm.complete(PROMPT)
    assert sent[0].url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert sent[0].headers["authorization"] == "Bearer test-key"


def test_make_llm_builds_a_client_for_any_openai_compatible_server() -> None:
    # e.g. vLLM on a rented GPU: `vllm serve <model> --api-key ...` serves /v1.
    settings = make_settings(
        llm_provider="openai_compatible",
        llm_model="org/open-model",
        llm_base_url="https://pod-8000.proxy.example/v1",
        llm_api_key="pod-key",
        llm_max_retries=3,
    )
    llm = make_llm(settings)
    assert isinstance(llm, OpenAiLlm)
    assert (llm.model, llm.base_url, llm.max_retries) == (
        "org/open-model",
        "https://pod-8000.proxy.example/v1",
        3,
    )


def test_openai_compatible_works_without_a_key() -> None:
    settings = make_settings(
        llm_provider="openai_compatible", llm_model="m", llm_base_url="http://localhost:8000/v1"
    )
    assert isinstance(make_llm(settings), OpenAiLlm)


def test_every_real_provider_gets_the_configured_retries() -> None:
    settings = make_settings(
        llm_provider="openai", llm_model="gpt-test", openai_api_key="sk-test", llm_max_retries=4
    )
    llm = make_llm(settings)
    assert isinstance(llm, OpenAiLlm)
    assert llm.max_retries == 4


def test_settings_prices_are_unset_by_default() -> None:
    settings = make_settings()
    assert settings.llm_prices is None


def test_settings_prices_need_both_values() -> None:
    settings = make_settings(llm_input_price_per_1m=0.40, llm_output_price_per_1m=1.60)
    assert settings.llm_prices == Prices(input_per_1m=0.40, output_per_1m=1.60)
    with pytest.raises(ValueError, match="LLM_OUTPUT_PRICE_PER_1M"):
        make_settings(llm_input_price_per_1m=0.40)


# --- thinking switch (vLLM / Qwen: LLM_THINKING=off) ------------------------

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_fields_are_sent_with_every_request() -> None:
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=completion("ok"))

    llm = OpenAiLlm(
        model="qwen",
        api_key="k",
        base_url="https://llm.example/v1",
        max_retries=0,
        http_client=DefaultHttpx2Client(transport=httpx2.MockTransport(handler)),
        extra_body=NO_THINKING,
    )
    llm.complete(PROMPT)
    assert json.loads(sent[0].content)["chat_template_kwargs"] == {"enable_thinking": False}


def test_no_extra_body_by_default() -> None:
    llm, sent = make_openai(lambda _: httpx2.Response(200, json=completion("ok")))
    llm.complete(PROMPT)
    assert "chat_template_kwargs" not in json.loads(sent[0].content)


def test_llm_thinking_off_turns_thinking_off_on_an_openai_compatible_server() -> None:
    base = {"llm_provider": "openai_compatible", "llm_model": "m", "llm_base_url": "http://x/v1"}
    off = make_llm(make_settings(**base, llm_thinking="off"))
    assert isinstance(off, OpenAiLlm) and off.extra_body == NO_THINKING
    default = make_llm(make_settings(**base))
    assert isinstance(default, OpenAiLlm) and default.extra_body is None


def test_llm_thinking_off_is_refused_for_other_providers() -> None:
    with pytest.raises(ValueError, match="LLM_THINKING"):
        make_settings(llm_provider="openai", llm_model="m", openai_api_key="k", llm_thinking="off")
