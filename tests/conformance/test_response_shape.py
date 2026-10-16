from typing import Any

import httpx2
from conftest import Caller

from primagate import PrimaGate

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]

BODIES = {
    "openai": {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    },
    "anthropic": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    },
}
MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}


def _gate() -> PrimaGate:
    def handle(request: httpx2.Request) -> httpx2.Response:
        provider = "anthropic" if request.url.host == "api.anthropic.com" else "openai"
        return httpx2.Response(
            200,
            json=BODIES[provider],
            headers={"date": "Thu, 01 Jan 1970 00:00:01 GMT"},
        )

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


def test_both_providers_return_the_identical_response_shape(call: Caller) -> None:
    gate = _gate()
    shapes = {
        provider: _shape(call(gate, model=model, messages=MSGS).to_dict())
        for provider, model in MODELS.items()
    }
    assert shapes["openai"] == shapes["anthropic"]
    assert shapes["anthropic"]["usage"] == {
        "prompt_tokens": "int",
        "completion_tokens": "int",
        "total_tokens": "int",
    }


def test_only_the_model_string_changes_between_providers(call: Caller) -> None:
    gate = _gate()
    first = call(gate, model="gpt-4o", messages=MSGS)
    second = call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert type(first) is type(second)
    assert first.choices[0].message.content == second.choices[0].message.content == "hi"
    assert first.choices[0].finish_reason == second.choices[0].finish_reason == "stop"
