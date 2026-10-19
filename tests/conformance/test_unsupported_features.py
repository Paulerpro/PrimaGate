import json
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller

from primagate import PrimaGate
from primagate.adapters.anthropic import AnthropicAdapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema import errors
from primagate.schema.wire import Capabilities

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
IMAGE = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
FEATURES: dict[str, dict[str, Any]] = {
    "tools": {"tools": [TOOL]},
    "response_format": {"response_format": {"type": "json_object"}},
    "image": {"messages": [{"role": "user", "content": IMAGE}]},
}
TEXT = [{"role": "user", "content": "hi"}]


def _gate() -> tuple[PrimaGate, list[httpx2.Request]]:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(500, json={"error": {"message": "should not be reached"}})

    gate = PrimaGate(
        api_keys={"openai": "sk-o", "anthropic": "sk-a"}, transport=httpx2.MockTransport(handle)
    )
    return gate, sent


@pytest.fixture
def bare_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    for adapter in (OpenAIAdapter, AnthropicAdapter):
        monkeypatch.setattr(adapter, "capabilities", lambda self: Capabilities())


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("provider", MODELS)
def test_no_feature_is_silently_dropped(call: Caller, provider: str, feature: str) -> None:
    gate, sent = _gate()
    kwargs: dict[str, Any] = {"model": MODELS[provider], "messages": TEXT, **FEATURES[feature]}
    with pytest.raises(errors.UnsupportedParamsError) as info:
        call(gate, **kwargs)
    assert feature in info.value.message and provider == info.value.llm_provider
    assert sent == []


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("provider", MODELS)
def test_streaming_calls_are_checked_too(
    stream_call: StreamCaller, provider: str, feature: str
) -> None:
    gate, sent = _gate()
    kwargs: dict[str, Any] = {"model": MODELS[provider], "messages": TEXT, **FEATURES[feature]}
    run = stream_call(gate, **kwargs)
    assert isinstance(run.call_error, errors.UnsupportedParamsError)
    assert sent == []


def test_openai_passes_image_parts_through(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.PrimaGateError):
        call(gate, model="gpt-4o", messages=FEATURES["image"]["messages"])
    assert json.loads(sent[0].content)["messages"][0]["content"] == IMAGE


def test_anthropic_rejects_images_before_any_request(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError, match="image"):
        call(gate, model=MODELS["anthropic"], messages=FEATURES["image"]["messages"])
    assert sent == []


def test_declared_features_still_reach_the_provider(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.PrimaGateError):
        call(gate, model="gpt-4o", messages=TEXT, tools=[TOOL])
    assert len(sent) == 1 and "tools" in json.loads(sent[0].content)
