"""Completion-boundary failures must be explicit and never expose partial output."""

from types import SimpleNamespace

import pytest

from fusion_agent.research.engine import ResearchEngine
from fusion_agent.research.providers import NebiusModel, ProviderError, Settings


def model_returning(response):
    model = object.__new__(NebiusModel)
    model.model = "test-model"
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_kwargs: response,
    )))
    return model


def completion(*, finish_reason="stop", content="A complete answer", tools=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, tool_calls=tools),
    )])


@pytest.mark.parametrize("content,tools", [
    ("PARTIAL_SECRET_MARKER", None),
    (None, [SimpleNamespace(id="call1", function=SimpleNamespace(
        name="finish_report", arguments='{"answer":"PARTIAL_SECRET_MARKER',
    ))]),
])
def test_output_limit_rejects_partial_text_and_tool_calls(content, tools):
    response = completion(finish_reason="length", content=content, tools=tools)
    with pytest.raises(ProviderError, match="truncated.*token limit") as raised:
        model_returning(response).complete([], [])
    assert "SECRET" not in str(raised.value)


def test_content_filter_rejects_even_present_output_without_exposing_it():
    response = completion(finish_reason="content_filter", content="SECRET_MARKER")
    with pytest.raises(ProviderError, match="content filtering") as raised:
        model_returning(response).complete([], [])
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize("response", [SimpleNamespace(), SimpleNamespace(choices=[])])
def test_missing_choices_are_a_specific_safe_error(response):
    with pytest.raises(ProviderError, match="no completion choices"):
        model_returning(response).complete([], [])


def test_missing_message_is_a_specific_safe_error():
    with pytest.raises(ProviderError, match="no completion message"):
        model_returning(SimpleNamespace(choices=[SimpleNamespace()])).complete([], [])


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_empty_or_reasoning_only_response_is_not_accepted(content):
    response = completion(content=content)
    response.choices[0].message.reasoning_content = "SECRET_PRIVATE_REASONING"
    with pytest.raises(ProviderError, match="empty completion") as raised:
        model_returning(response).complete([], [])
    assert "SECRET" not in str(raised.value)


def test_normal_text_completion_preserves_contract_with_optional_usage_absent():
    message, usage = model_returning(completion()).complete([], [])
    assert message == {"role": "assistant", "content": "A complete answer"}
    assert usage == {}


def test_complete_tool_call_with_no_text_is_accepted():
    tools = [SimpleNamespace(id="call1", function=SimpleNamespace(
        name="plan_research", arguments='{"reason":"Plan"}',
    ))]
    response = completion(finish_reason="tool_calls", content=None, tools=tools)
    message, usage = model_returning(response).complete([], [])
    assert message == {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call1", "type": "function",
        "function": {"name": "plan_research", "arguments": '{"reason":"Plan"}'},
    }]}
    assert usage == {}


def test_malformed_tool_response_is_sanitized():
    response = completion(content=None, tools=[SimpleNamespace(id="SECRET_MARKER")])
    with pytest.raises(ProviderError, match="invalid completion") as raised:
        model_returning(response).complete([], [])
    assert "SECRET" not in str(raised.value)
    assert raised.value.__suppress_context__


def test_upstream_provider_error_is_not_trusted_as_an_already_safe_message():
    def fail(**_kwargs):
        raise ProviderError("SECRET_MARKER upstream body")

    model = model_returning(None)
    model.client.chat.completions.create = fail
    with pytest.raises(ProviderError, match="Nebius request failed") as raised:
        model.complete([], [])
    assert "SECRET" not in str(raised.value)
    assert raised.value.__suppress_context__


@pytest.mark.parametrize("limit", [256, 5000, 12000, 16000])
def test_configured_output_limit_reaches_model_request(monkeypatch, limit):
    captured = {}

    def respond(**kwargs):
        captured.update(kwargs)
        return completion()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: client)
    settings = Settings(nebius_key="fake-n", tavily_key="fake-t", max_output_tokens=limit)
    settings.validate()
    NebiusModel(settings).complete([], [])
    assert captured["max_tokens"] == limit
    assert settings.public()["max_output_tokens"] == limit


@pytest.mark.parametrize("value", [255, 16001, True, 5000.5, "secret", "1e4"])
def test_invalid_output_limit_is_rejected_without_echoing_input(value):
    settings = Settings(nebius_key="fake-n", tavily_key="fake-t", max_output_tokens=value)
    with pytest.raises(ValueError, match="integer from 256 to 16000") as raised:
        settings.validate()
    assert "secret" not in str(raised.value)
    assert settings.public()["max_output_tokens"] == "invalid token limit"


def test_output_limit_environment_override(monkeypatch):
    # Blank unrelated credentials instead of reading locally configured values.
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.setenv("NEBIUS_MAX_OUTPUT_TOKENS", "12000")
    assert Settings.from_env().max_output_tokens == 12000
    monkeypatch.delenv("NEBIUS_MAX_OUTPUT_TOKENS")
    assert Settings.from_env().max_output_tokens == 5000
    monkeypatch.setenv("NEBIUS_MAX_OUTPUT_TOKENS", "malformed-secret")
    with pytest.raises(ValueError) as raised:
        Settings.from_env()
    assert "malformed-secret" not in str(raised.value)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "stop"])
def test_rejected_completion_keeps_only_safe_usage_and_finish_reason(finish_reason):
    response = completion(finish_reason=finish_reason, content=None)
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5000, total_tokens=5010,
        provider_body="SECRET_MARKER", private_reasoning="SECRET_MARKER",
    )
    with pytest.raises(ProviderError) as raised:
        model_returning(response).complete([], [])
    assert raised.value.finish_reason == finish_reason
    assert raised.value.usage == {
        "prompt_tokens": 10, "completion_tokens": 5000, "total_tokens": 5010,
    }
    assert "SECRET_MARKER" not in repr(vars(raised.value))


def test_failure_event_counts_paid_tokens_with_role_without_response_content():
    response = completion(finish_reason="length", content="SECRET_PARTIAL_RESPONSE")
    response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5000, total_tokens=5010)
    events = []
    engine = ResearchEngine([], model_returning(response), None, events.append)
    engine.total_usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    with engine.agent("federation_strategist"), pytest.raises(ProviderError):
        engine.request([{"role": "user", "content": "SECRET_PROMPT"}], [])
    assert engine.model_calls == 1
    assert engine.total_usage == {
        "prompt_tokens": 13, "completion_tokens": 5004, "total_tokens": 5017,
    }
    assert len(events) == 1
    failure = events[0]
    assert failure["kind"] == "provider_failure"
    assert failure["role"] == failure["agent"] == "federation_strategist"
    assert failure["finish_reason"] == "length"
    assert failure["response_usage"]["total_tokens"] == 5010
    assert failure["usage"] == engine.total_usage
    assert "SECRET" not in repr(events)


def test_provider_error_metadata_rejects_nonusage_and_unknown_finish_reason():
    error = ProviderError("safe", finish_reason="SECRET_REASON", usage={
        "prompt_tokens": True, "completion_tokens": -1, "total_tokens": 9,
        "provider_secret": "SECRET_VALUE",
    })
    assert error.finish_reason is None
    assert error.usage == {"total_tokens": 9}


@pytest.mark.parametrize("thinking", [None, False, True])
def test_thinking_extension_is_only_sent_when_explicitly_configured(monkeypatch, thinking):
    captured = {}

    def respond(**kwargs):
        captured.update(kwargs)
        return completion()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: client)
    settings = Settings(
        nebius_key="fake-n", tavily_key="fake-t", enable_thinking=thinking,
    )
    settings.validate()
    NebiusModel(settings).complete([], [])
    if thinking is None:
        assert "extra_body" not in captured
    else:
        assert captured["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
    assert settings.public()["enable_thinking"] is thinking
    assert captured["max_tokens"] == 5000


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("true", True), ("false", False), (" FALSE ", False),
])
def test_thinking_environment_is_optional_and_parsed_as_boolean(monkeypatch, value, expected):
    monkeypatch.setenv("NEBIUS_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    monkeypatch.delenv("NEBIUS_MAX_OUTPUT_TOKENS", raising=False)
    if value is None:
        monkeypatch.delenv("NEBIUS_ENABLE_THINKING", raising=False)
    else:
        monkeypatch.setenv("NEBIUS_ENABLE_THINKING", value)
    assert Settings.from_env().enable_thinking is expected


@pytest.mark.parametrize("value", ["secret-value", 1, 0, [], {}, "enabled"])
def test_invalid_thinking_setting_is_rejected_without_echoing_it(value):
    settings = Settings(nebius_key="fake-n", tavily_key="fake-t", enable_thinking=value)
    with pytest.raises(ValueError, match="true, false or unset") as raised:
        settings.validate()
    assert "secret-value" not in str(raised.value)
    assert settings.public()["enable_thinking"] == "invalid thinking setting"


def test_thinking_override_keeps_truncation_and_usage_failure_handling():
    response = completion(finish_reason="length", content="SECRET_PARTIAL")
    response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=100, total_tokens=110)
    model = model_returning(response)
    model.enable_thinking = False
    with pytest.raises(ProviderError, match="truncated") as raised:
        model.complete([], [])
    assert raised.value.finish_reason == "length"
    assert raised.value.usage["total_tokens"] == 110
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize("kind,http_status", [
    ("APITimeoutError", None), ("APIConnectionError", None), ("APIStatusError", 429),
])
def test_network_failures_record_only_exception_class_and_http_status(kind, http_status):
    import httpx
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    request = httpx.Request(
        "POST", "https://api.example.org/chat?token=SECRET_URL",
        headers={"Authorization": "Bearer SECRET_HEADER"},
    )
    errors = {
        "APITimeoutError": APITimeoutError(request=request),
        "APIConnectionError": APIConnectionError(message="SECRET_BODY", request=request),
        "APIStatusError": APIStatusError(
            "SECRET_BODY", response=httpx.Response(429, request=request),
            body={"secret": "SECRET_BODY"},
        ),
    }

    def fail(**_kwargs):
        raise errors[kind]

    model = model_returning(None)
    model.client.chat.completions.create = fail
    events = []
    engine = ResearchEngine([], model, None, events.append)
    with engine.agent("literature_researcher"), pytest.raises(ProviderError) as raised:
        engine.request([], [])
    error = raised.value
    assert error.error_type == kind
    assert error.http_status == http_status
    assert error.__suppress_context__
    assert "SECRET" not in str(error) + repr(vars(error)) + repr(events)
    assert events[0]["kind"] == "provider_failure"
    assert events[0]["role"] == "literature_researcher"
    assert events[0]["error_type"] == kind
    assert events[0]["http_status"] == http_status
    assert events[0]["response_usage"] == {}


def test_non_sdk_exception_status_is_not_recorded_as_sdk_http_status():
    class LocalFailure(Exception):
        status_code = 400

    def fail(**_kwargs):
        raise LocalFailure("SECRET_BODY")

    model = model_returning(None)
    model.client.chat.completions.create = fail
    with pytest.raises(ProviderError) as raised:
        model.complete([], [])
    assert raised.value.error_type == "LocalFailure"
    assert raised.value.http_status is None


@pytest.mark.parametrize("error_type,http_status", [
    ("https://example.org/SECRET_URL", "429"),
    ("Failure\nSECRET_BODY", True),
    ("a" * 81, 999),
])
def test_diagnostic_metadata_rejects_unbounded_types_and_invalid_status(error_type, http_status):
    error = ProviderError("safe", error_type=error_type, http_status=http_status)
    assert error.error_type is None
    assert error.http_status is None
