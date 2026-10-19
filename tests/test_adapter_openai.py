import ast
import json
from pathlib import Path

import pytest

import primagate.adapters.openai as openai_adapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema import errors
from primagate.schema.types import ChatCompletionRequest, FunctionCall, Message, ToolCall
from primagate.schema.wire import RawEvent, RawOutcome


def _outcome(status: int, body: object, *, headers: dict[str, str] | None = None) -> RawOutcome:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return RawOutcome(status=status, headers=headers or {}, body=raw)


def test_build_request() -> None:
    adapter = OpenAIAdapter()
    req = ChatCompletionRequest(
        model="gpt-4o", messages=[Message(role="user", content="hi")], temperature=0.1
    )
    out = adapter.build_request(req, api_key="sk-secret", base_url="https://x.test/v1/")
    assert out.method == "POST"
    assert out.url == "https://x.test/v1/chat/completions"
    assert out.headers["Authorization"] == "Bearer sk-secret"
    assert out.json_body == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.1,
    }
    assert "sk-secret" not in repr(out)


def test_parse_response_ignores_unknown_fields() -> None:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 5,
        "model": "gpt-4o",
        "system_fingerprint": "fp",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": "hello", "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "x": {}},
    }
    resp = OpenAIAdapter().parse_response(_outcome(200, body))
    assert resp.choices[0].message.content == "hello"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 3


def test_parse_response_without_usage() -> None:
    body = {"id": "c", "created": 1, "model": "m", "choices": []}
    assert OpenAIAdapter().parse_response(_outcome(200, body)).usage is None


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"choices": 5}'])
def test_parse_response_invalid_body_is_api_error(body: bytes) -> None:
    with pytest.raises(errors.APIError):
        OpenAIAdapter().parse_response(_outcome(200, body))


def _err(status: int, code: str | None = None, msg: str = "boom") -> RawOutcome:
    return _outcome(status, {"error": {"message": msg, "code": code, "type": "t"}})


@pytest.mark.parametrize(
    ("outcome", "cls"),
    [
        (_err(400), errors.BadRequestError),
        (_err(400, "context_length_exceeded"), errors.ContextWindowExceededError),
        (_err(400, "content_policy_violation"), errors.ContentPolicyViolationError),
        (_err(401), errors.AuthenticationError),
        (_err(403), errors.PermissionDeniedError),
        (_err(404), errors.NotFoundError),
        (_err(408), errors.Timeout),
        (_err(429), errors.RateLimitError),
        (_err(500), errors.ServiceUnavailableError),
        (_err(503), errors.ServiceUnavailableError),
        (_outcome(502, b"<html>bad gateway</html>"), errors.ServiceUnavailableError),
        (_err(418), errors.APIError),
        (RawOutcome(status=None, headers={}, body=b"", failure="timeout"), errors.Timeout),
        (
            RawOutcome(status=None, headers={}, body=b"", failure="connection"),
            errors.APIConnectionError,
        ),
    ],
)
def test_map_error(outcome: RawOutcome, cls: type[errors.PrimaGateError]) -> None:
    err = OpenAIAdapter().map_error(outcome)
    assert type(err) is cls
    assert err.llm_provider == "openai"
    assert err.status_code == outcome.status


def test_map_error_uses_provider_message() -> None:
    assert OpenAIAdapter().map_error(_err(429, msg="slow down")).message == "slow down"


def test_streaming_capability_and_usage_request() -> None:
    adapter = OpenAIAdapter()
    assert adapter.capabilities().streaming is True
    request = ChatCompletionRequest(
        model="gpt-4o", messages=[Message(role="user", content="hi")], stream=True
    )
    body = adapter.build_request(request, api_key="k", base_url="https://x.test/v1").json_body
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def _data(payload: object) -> RawEvent:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return RawEvent("message", text)


def _delta(delta: dict[str, object], finish: str | None = None, **extra: object) -> RawEvent:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return _data({"id": "c", "created": 1, "model": "m", "choices": [choice], **extra})


USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_stream_parser_puts_usage_on_the_finish_chunk() -> None:
    parser = OpenAIAdapter().new_stream_parser({})
    out = []
    for event in (
        _delta({"role": "assistant", "content": ""}),
        _delta({"content": "hi"}),
        _delta({}, "stop"),
        _data({"id": "c", "created": 1, "model": "m", "choices": [], "usage": USAGE}),
        _data("[DONE]"),
    ):
        out += parser.feed(event)
    assert parser.finish() == []
    assert [c.choices[0].delta.content for c in out] == ["hi", None]
    assert out[0].choices[0].delta.role == "assistant"
    assert out[1].choices[0].finish_reason == "stop"
    assert out[1].usage is not None and out[1].usage.total_tokens == 3


def test_stream_parser_keeps_finish_chunk_that_already_has_usage() -> None:
    parser = OpenAIAdapter().new_stream_parser({})
    out = parser.feed(_delta({"content": "x"}, "stop", usage=USAGE)) + parser.feed(_data("[DONE]"))
    assert len(out) == 1 and out[0].usage is not None


def test_stream_parser_releases_finish_chunk_without_usage_at_done() -> None:
    parser = OpenAIAdapter().new_stream_parser({})
    assert parser.feed(_delta({}, "length")) == []
    out = parser.feed(_data("[DONE]"))
    assert out[0].choices[0].finish_reason == "length" and out[0].usage is None


def test_stream_parser_maps_in_band_error_and_early_end() -> None:
    parser = OpenAIAdapter().new_stream_parser({})
    with pytest.raises(errors.APIError, match="boom"):
        parser.feed(_data({"error": {"message": "boom"}}))
    with pytest.raises(errors.APIError, match="DONE"):
        OpenAIAdapter().new_stream_parser({}).finish()
    with pytest.raises(errors.APIError, match="Unparseable"):
        parser.feed(_data("not json"))


def test_adapter_module_performs_no_io() -> None:
    tree = ast.parse(Path(openai_adapter.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"httpx2", "socket", "urllib", "requests", "os", "subprocess"}


@pytest.mark.parametrize("status", [500, 502, 504])
def test_all_5xx_map_to_service_unavailable(status: int) -> None:
    assert type(OpenAIAdapter().map_error(_err(status))) is errors.ServiceUnavailableError


@pytest.mark.parametrize("code", [["a"], {"k": 1}, 5])
def test_non_string_error_code_does_not_crash(code: object) -> None:
    out = _outcome(400, {"error": {"message": "bad", "code": code}})
    assert type(OpenAIAdapter().map_error(out)) is errors.BadRequestError


@pytest.mark.parametrize("message", [None, ""])
def test_empty_provider_message_falls_back_to_body(message: object) -> None:
    out = _outcome(400, {"error": {"message": message, "code": None}})
    assert OpenAIAdapter().map_error(out).message not in ("None", "")


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def test_tools_and_tool_messages_are_sent_in_the_unified_shape() -> None:
    call = ToolCall(id="call_1", function=FunctionCall(name="f", arguments='{"a":1}'))
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="call_1", content="42"),
        ],
        tools=TOOLS,
    )
    body = OpenAIAdapter().build_request(req, api_key="k", base_url="https://x.test/v1").json_body
    assert body["tools"] == TOOLS
    assert body["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
        ],
    }
    assert body["messages"][2] == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


def test_parse_returns_parallel_tool_calls_in_order() -> None:
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "f", "arguments": f'{{"i":{i}}}'},
        }
        for i in range(3)
    ]
    body = {
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
    }
    choice = OpenAIAdapter().parse_response(_outcome(200, body)).choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert choice.message.tool_calls is not None
    assert [c.id for c in choice.message.tool_calls] == ["call_0", "call_1", "call_2"]
    assert choice.message.tool_calls[1].function.arguments == '{"i":1}'


def test_adapter_declares_tool_support() -> None:
    assert OpenAIAdapter().capabilities().tools is True


def test_structured_output_capability() -> None:
    assert OpenAIAdapter().capabilities().structured_output is True


def test_images_capability() -> None:
    assert OpenAIAdapter().capabilities().images is True
