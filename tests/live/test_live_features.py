"""Live: an unsupported feature is refused before sending, or dropped and recorded on request."""

import pytest

from primagate import UnsupportedParamsError
from primagate.adapters.anthropic import AnthropicAdapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema.wire import Capabilities

from .support import ASK, TOOLS, Live, LiveCaller

pytestmark = pytest.mark.live

IMAGE = {"type": "image_url", "image_url": {"url": "https://example.invalid/never-fetched.png"}}
WITH_IMAGE = [
    {"role": "user", "content": [{"type": "text", "text": "Reply with the word: ready"}, IMAGE]}
]


@pytest.fixture
def bare_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both providers take tools, so a provider without them is made by hiding the capability.
    # What is then sent, and the real provider's answer to it, are still live.
    for adapter in (OpenAIAdapter, AnthropicAdapter):
        monkeypatch.setattr(adapter, "capabilities", lambda self: Capabilities())


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR4", "FR5", "FR6")
def test_a_feature_the_model_lacks_is_refused_then_dropped_when_named(
    call: LiveCaller, live: Live
) -> None:
    gate = live.gate()
    with pytest.raises(UnsupportedParamsError):
        call(gate, model=live.model, messages=ASK, tools=TOOLS)
    with pytest.raises(UnsupportedParamsError):
        call(gate, model=live.model, messages=ASK, tools=TOOLS, drop_params=True)
    assert "dispatch_started" not in live.events.types

    result = call(gate, model=live.model, messages=ASK, tools=TOOLS, drop_params=["tools"])
    assert result.metadata is not None and result.metadata.dropped_params == ["tools"]
    assert result.choices[0].message.content and not result.choices[0].message.tool_calls
    assert live.events.types.count("dispatch_started") == 1


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR6")
def test_a_stream_records_the_drop_on_its_final_chunk(call: LiveCaller, live: Live) -> None:
    chunks = call.stream(
        live.gate(), model=live.model, messages=ASK, tools=TOOLS, drop_params=["tools"]
    )
    assert all(c.metadata is None for c in chunks[:-1])
    final = chunks[-1].metadata
    assert final is not None and final.dropped_params == ["tools"]


@pytest.mark.parametrize("provider", ["anthropic"], indirect=True)
@pytest.mark.fr("FR4", "FR5", "FR6")
def test_anthropic_refuses_an_image_and_drops_it_when_named(call: LiveCaller, live: Live) -> None:
    # Images are a feature the Anthropic adapter does not implement, so nothing is faked here.
    gate = live.gate()
    with pytest.raises(UnsupportedParamsError):
        call(gate, model=live.model, messages=WITH_IMAGE)
    assert "dispatch_started" not in live.events.types
    result = call(gate, model=live.model, messages=WITH_IMAGE, drop_params=["image"])
    assert result.metadata is not None and result.metadata.dropped_params == ["image"]
    assert result.choices[0].message.content


@pytest.mark.parametrize("provider", ["anthropic"], indirect=True)
@pytest.mark.fr("FR5", "FR6")
def test_anthropic_drops_hint_params_under_true_and_records_them(
    call: LiveCaller, live: Live
) -> None:
    gate = live.gate()
    with pytest.raises(UnsupportedParamsError):
        call(gate, model=live.model, messages=ASK, seed=1)
    live.events.refuse()
    result = call(gate, model=live.model, messages=ASK, seed=1, n=2, drop_params=True)
    assert result.metadata is not None and result.metadata.dropped_params == ["n", "seed"]
    assert len(result.choices) == 1 and result.choices[0].message.content
