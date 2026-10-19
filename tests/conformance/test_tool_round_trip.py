import json
from typing import Any

import httpx2
import pytest
from conftest import Caller

from primagate import PrimaGate

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
CITIES = ["Oslo", "Lima"]
FINAL_TEXT = "Oslo is cold, Lima is mild."


def _openai_call_body() -> dict[str, Any]:
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "weather", "arguments": json.dumps({"city": city})},
        }
        for i, city in enumerate(CITIES)
    ]
    return {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": calls},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _openai_final_body() -> dict[str, Any]:
    body = _openai_call_body()
    body["choices"][0] = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": FINAL_TEXT},
    }
    return body


def _anthropic_call_body() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "tool_use", "id": f"toolu_{i}", "name": "weather", "input": {"city": city}}
            for i, city in enumerate(CITIES)
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _anthropic_final_body() -> dict[str, Any]:
    body = _anthropic_call_body()
    body["content"] = [{"type": "text", "text": FINAL_TEXT}]
    body["stop_reason"] = "end_turn"
    return body


def _has_tool_result(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message["content"]
        if message["role"] == "tool":
            return True
        if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
            return True
    return False


def _gate(sent: list[dict[str, Any]]) -> PrimaGate:
    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        done = _has_tool_result(body["messages"])
        if request.url.host == "api.anthropic.com":
            reply = _anthropic_final_body() if done else _anthropic_call_body()
        else:
            reply = _openai_final_body() if done else _openai_call_body()
        return httpx2.Response(200, json=reply, headers={"date": "Thu, 01 Jan 1970 00:00:01 GMT"})

    return PrimaGate(
        api_keys={"openai": "sk-o", "anthropic": "sk-a"}, transport=httpx2.MockTransport(handle)
    )


def _shape(value: Any) -> Any:
    """Keys and value types only, so providers compare on shape rather than content."""
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value]
    return type(value).__name__


@pytest.mark.parametrize("provider", MODELS)
def test_tool_round_trip_completes(call: Caller, provider: str) -> None:
    sent: list[dict[str, Any]] = []
    gate = _gate(sent)
    model = MODELS[provider]
    messages: list[Any] = [{"role": "user", "content": "weather in Oslo and Lima?"}]

    first = call(gate, model=model, messages=messages, tools=TOOLS)
    message = first.choices[0].message
    assert first.choices[0].finish_reason == "tool_calls"
    assert message.tool_calls is not None
    assert [json.loads(c.function.arguments)["city"] for c in message.tool_calls] == CITIES
    assert len({c.id for c in message.tool_calls}) == 2

    messages.append(message)
    for tool_call, result in zip(message.tool_calls, ["cold", "mild"], strict=True):
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    second = call(gate, model=model, messages=messages, tools=TOOLS)
    assert second.choices[0].finish_reason == "stop"
    assert second.choices[0].message.content == FINAL_TEXT
    assert len(sent) == 2


def test_tool_call_response_shape_is_identical_across_providers(call: Caller) -> None:
    gate = _gate([])
    shapes = {
        provider: _shape(
            call(
                gate, model=model, messages=[{"role": "user", "content": "w?"}], tools=TOOLS
            ).to_dict()
        )
        for provider, model in MODELS.items()
    }
    assert shapes["openai"] == shapes["anthropic"]


def test_anthropic_receives_one_merged_result_turn(call: Caller) -> None:
    sent: list[dict[str, Any]] = []
    gate = _gate(sent)
    model = MODELS["anthropic"]
    first = call(gate, model=model, messages=[{"role": "user", "content": "w?"}], tools=TOOLS)
    assistant = first.choices[0].message
    assert assistant.tool_calls is not None
    results = [{"role": "tool", "tool_call_id": c.id, "content": "x"} for c in assistant.tool_calls]
    call(
        gate,
        model=model,
        messages=[{"role": "user", "content": "w?"}, assistant, *results],
        tools=TOOLS,
    )
    last = sent[-1]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user"]
    assert len(last[-1]["content"]) == 2
