import json
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller

import primagate
import primagate.client as client_module
from primagate import PrimaGate
from primagate.adapters.anthropic import AnthropicAdapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema import errors
from primagate.schema.wire import Capabilities

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
TEXT = [{"role": "user", "content": "hi"}]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
IMAGE_PART = {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
MIXED = [{"role": "user", "content": [{"type": "text", "text": "what?"}, IMAGE_PART]}]
SSE = {"content-type": "text/event-stream", "date": "Thu, 01 Jan 1970 00:00:01 GMT"}


def _ok(provider: str) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "id": "msg_1",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    return {
        "id": "c",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _line(event: str | None, data: Any) -> str:
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"


def _stream_body(provider: str) -> bytes:
    if provider == "anthropic":
        start = {"id": "m", "model": "c", "role": "assistant", "usage": {"input_tokens": 1}}
        return "".join(
            [
                _line("message_start", {"type": "message_start", "message": start}),
                _line(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                ),
                _line(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 2},
                    },
                ),
                _line("message_stop", {"type": "message_stop"}),
            ]
        ).encode()
    base = {"id": "c", "created": 1, "model": "m"}
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    return (
        _line(None, {**base, "choices": [{"index": 0, "delta": {"content": "ok"}}]})
        + _line(None, {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + _line(None, {**base, "choices": [], "usage": usage})
        + "data: [DONE]\n\n"
    ).encode()


def _gate(**kwargs: Any) -> tuple[PrimaGate, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        provider = "anthropic" if request.url.host == "api.anthropic.com" else "openai"
        if body.get("stream"):
            return httpx2.Response(200, content=_stream_body(provider), headers=SSE)
        return httpx2.Response(200, json=_ok(provider), headers={"date": SSE["date"]})

    gate = PrimaGate(
        api_keys={"openai": "sk-o", "anthropic": "sk-a"},
        transport=httpx2.MockTransport(handle),
        **kwargs,
    )
    return gate, sent


@pytest.fixture
def bare_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    for adapter in (OpenAIAdapter, AnthropicAdapter):
        monkeypatch.setattr(adapter, "capabilities", lambda self: Capabilities())


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR4", "FR5")
def test_true_never_drops_a_behavior_feature(call: Caller, provider: str) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model=MODELS[provider], messages=TEXT, tools=[TOOL], drop_params=True)
    assert sent == []


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR5", "FR6")
def test_named_features_are_dropped_and_recorded(call: Caller, provider: str) -> None:
    gate, sent = _gate()
    result = call(
        gate,
        model=MODELS[provider],
        messages=MIXED,
        tools=[TOOL],
        response_format={"type": "json_object"},
        drop_params=["tools", "response_format", "image"],
    )
    body = sent[0]
    assert "tools" not in body and "response_format" not in body and "output_config" not in body
    assert body["messages"][0]["content"] in ("what?", [{"type": "text", "text": "what?"}])
    assert result.metadata is not None
    assert result.metadata.dropped_params == ["tools", "response_format", "image"]


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR5")
def test_an_image_only_message_becomes_empty_text(call: Caller) -> None:
    gate, sent = _gate()
    messages = [{"role": "user", "content": [IMAGE_PART]}]
    call(gate, model="gpt-4o", messages=messages, drop_params=["image"])
    assert sent[0]["messages"][0]["content"] == ""


@pytest.mark.fr("FR5")
def test_supported_named_features_are_kept(call: Caller) -> None:
    gate, sent = _gate()
    result = call(gate, model="gpt-4o", messages=TEXT, tools=[TOOL], drop_params=["tools"])
    assert "tools" in sent[0]
    assert result.metadata is not None and result.metadata.dropped_params == []


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR4", "FR5")
def test_tool_history_still_raises_when_tools_are_dropped(call: Caller) -> None:
    gate, sent = _gate()
    history = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model="gpt-4o", messages=history, tools=[TOOL], drop_params=["tools"])
    assert sent == []


@pytest.mark.fr("FR5", "FR6")
def test_hint_params_are_dropped_and_recorded_under_true(call: Caller) -> None:
    gate, sent = _gate()
    result = call(gate, model=MODELS["anthropic"], messages=TEXT, seed=1, n=2, drop_params=True)
    assert "seed" not in sent[0] and "n" not in sent[0]
    assert result.metadata is not None and result.metadata.dropped_params == ["n", "seed"]


@pytest.mark.fr("FR5")
def test_hint_params_still_raise_without_the_opt_in(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model=MODELS["anthropic"], messages=TEXT, seed=1)
    assert sent == []


@pytest.mark.fr("FR5")
def test_n_of_one_is_not_a_drop(call: Caller) -> None:
    gate, _ = _gate()
    result = call(gate, model=MODELS["anthropic"], messages=TEXT, n=1, drop_params=True)
    assert result.metadata is not None and result.metadata.dropped_params == []


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR5")
def test_per_call_value_beats_init_value(call: Caller) -> None:
    gate, sent = _gate(drop_params=["tools"])
    result = call(gate, model="gpt-4o", messages=TEXT, tools=[TOOL])
    assert result.metadata is not None and result.metadata.dropped_params == ["tools"]
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model="gpt-4o", messages=TEXT, tools=[TOOL], drop_params=False)
    assert len(sent) == 1


@pytest.mark.parametrize("bad", ["tools", ["nope"], 3, [1]])
@pytest.mark.fr("FR5")
def test_bad_values_are_rejected(call: Caller, bad: object) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.BadRequestError, match="drop_params"):
        call(gate, model="gpt-4o", messages=TEXT, drop_params=bad)
    assert sent == []
    with pytest.raises(errors.BadRequestError, match="drop_params"):
        PrimaGate(drop_params=bad)  # type: ignore[arg-type]


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.fr("FR5")
def test_module_level_value_is_read_at_call_time(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate, _ = _gate()
    monkeypatch.setattr(client_module, "_default", gate)
    monkeypatch.setattr(primagate, "drop_params", None, raising=False)
    with pytest.raises(errors.UnsupportedParamsError):
        call.module(model="gpt-4o", messages=TEXT, tools=[TOOL])
    monkeypatch.setattr(primagate, "drop_params", ["tools"], raising=False)
    result = call.module(model="gpt-4o", messages=TEXT, tools=[TOOL])
    assert result.metadata is not None and result.metadata.dropped_params == ["tools"]
    other, _ = _gate()
    with pytest.raises(errors.UnsupportedParamsError):
        call(other, model="gpt-4o", messages=TEXT, tools=[TOOL])


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR6")
def test_streams_carry_dropped_params_on_the_final_chunk(
    stream_call: StreamCaller, provider: str
) -> None:
    gate, _ = _gate()
    run = stream_call(
        gate, model=MODELS[provider], messages=TEXT, tools=[TOOL], drop_params=["tools"]
    )
    assert run.call_error is None and run.read_error is None
    assert all(c.metadata is None for c in run.chunks[:-1])
    final = run.chunks[-1]
    assert final.choices[0].finish_reason is not None
    assert final.metadata is not None and final.metadata.dropped_params == ["tools"]


@pytest.mark.parametrize("value", [["tools"], False, []])
@pytest.mark.fr("FR5")
def test_hint_params_are_not_dropped_without_true(call: Caller, value: object) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model=MODELS["anthropic"], messages=TEXT, seed=1, drop_params=value)
    assert sent == []


@pytest.mark.fr("FR5")
def test_n_below_two_is_not_a_hint_drop(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError):
        call(gate, model=MODELS["anthropic"], messages=TEXT, n=0, drop_params=True)
    assert sent == []


@pytest.mark.usefixtures("bare_adapters")
@pytest.mark.parametrize("explicit", [False, []])
@pytest.mark.fr("FR5")
def test_per_call_off_beats_the_module_level_value(
    call: Caller, monkeypatch: pytest.MonkeyPatch, explicit: object
) -> None:
    gate, sent = _gate()
    monkeypatch.setattr(client_module, "_default", gate)
    monkeypatch.setattr(primagate, "drop_params", ["tools"], raising=False)
    with pytest.raises(errors.UnsupportedParamsError):
        call.module(model="gpt-4o", messages=TEXT, tools=[TOOL], drop_params=explicit)
    assert sent == []
    monkeypatch.setattr(primagate, "drop_params", None, raising=False)
    result = call.module(model="gpt-4o", messages=TEXT, tools=[TOOL], drop_params=["tools"])
    assert result.metadata is not None and result.metadata.dropped_params == ["tools"]


@pytest.mark.fr("FR5", "FR6")
def test_temperature_on_a_model_that_rejects_it_is_dropped_and_recorded(call: Caller) -> None:
    gate, sent = _gate()
    result = call(gate, model=MODELS["anthropic"], messages=TEXT, temperature=0, drop_params=True)
    assert "temperature" not in sent[0]
    assert result.metadata is not None and result.metadata.dropped_params == ["temperature"]


@pytest.mark.fr("FR5")
def test_temperature_on_a_model_that_rejects_it_raises_without_the_opt_in(call: Caller) -> None:
    gate, sent = _gate()
    with pytest.raises(errors.UnsupportedParamsError) as info:
        call(gate, model=MODELS["anthropic"], messages=TEXT, temperature=0)
    assert "temperature" in info.value.message
    assert sent == []


@pytest.mark.fr("FR5")
def test_temperature_is_still_sent_to_openai(call: Caller) -> None:
    gate, sent = _gate()
    result = call(gate, model=MODELS["openai"], messages=TEXT, temperature=0, drop_params=True)
    assert sent[0]["temperature"] == 0
    assert result.metadata is not None and result.metadata.dropped_params == []


@pytest.mark.fr("FR5", "FR6")
def test_top_p_on_a_model_that_rejects_it_is_dropped_and_recorded(call: Caller) -> None:
    gate, sent = _gate()
    result = call(
        gate, model=MODELS["anthropic"], messages=TEXT, temperature=0, top_p=0.9, drop_params=True
    )
    assert "top_p" not in sent[0] and "temperature" not in sent[0]
    assert result.metadata is not None
    assert result.metadata.dropped_params == ["temperature", "top_p"]
