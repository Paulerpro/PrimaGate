import json
from typing import Any

import httpx2
import pytest
from conftest import Caller

from primagate import PrimaGate

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
CONTENT = json.dumps({"city": "Oslo"})
FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "place",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _gate(sent: list[dict[str, Any]]) -> PrimaGate:
    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        if request.url.host == "api.anthropic.com":
            reply: dict[str, Any] = {
                "id": "msg_1",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": CONTENT}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            }
        else:
            reply = {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": CONTENT},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        return httpx2.Response(200, json=reply, headers={"date": "Thu, 01 Jan 1970 00:00:01 GMT"})

    return PrimaGate(
        api_keys={"openai": "sk-o", "anthropic": "sk-a"}, transport=httpx2.MockTransport(handle)
    )


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value]
    return type(value).__name__


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR2")
def test_structured_content_is_valid_json_for_the_schema(call: Caller, provider: str) -> None:
    sent: list[dict[str, Any]] = []
    result = call(
        _gate(sent),
        model=MODELS[provider],
        messages=[{"role": "user", "content": "where?"}],
        response_format=FORMAT,
    )
    assert json.loads(result.choices[0].message.content or "") == {"city": "Oslo"}
    assert ("response_format" in sent[0]) == (provider == "openai")
    assert ("output_config" in sent[0]) == (provider == "anthropic")


@pytest.mark.fr("FR1", "FR2")
def test_response_shape_is_identical_across_providers(call: Caller) -> None:
    gate = _gate([])
    shapes = {
        provider: _shape(
            call(
                gate,
                model=model,
                messages=[{"role": "user", "content": "where?"}],
                response_format=FORMAT,
            ).to_dict()
        )
        for provider, model in MODELS.items()
    }
    assert shapes["openai"] == shapes["anthropic"]
