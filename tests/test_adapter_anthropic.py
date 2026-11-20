import ast
import json
from pathlib import Path

import pytest

import primagate.adapters.anthropic as anthropic_adapter
from primagate.adapters.anthropic import AnthropicAdapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema import errors
from primagate.schema.types import ChatCompletionRequest, FunctionCall, Message, ToolCall
from primagate.schema.wire import RawEvent, RawOutcome

BASE = "https://x.test/v1"


def _request(*messages: Message, **kwargs: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="claude-sonnet-5-5", messages=list(messages), **kwargs)  # type: ignore[arg-type]


def _build(*messages: Message, **kwargs: object):  # type: ignore[no-untyped-def]
    return AnthropicAdapter().build_request(
        _request(*messages, **kwargs), api_key="sk-ant-secret", base_url=BASE + "/"
    )


def _outcome(status: int, body: object, *, headers: dict[str, str] | None = None) -> RawOutcome:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return RawOutcome(status=status, headers=headers or {}, body=raw)


OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 4},
}


def test_build_request_shape_and_headers() -> None:
    # A model that takes `temperature`; version 5 models reject it.
    out = _for_model("claude-sonnet-4-5", temperature=0.2, top_p=0.9)
    assert out.method == "POST"
    assert out.url == f"{BASE}/messages"
    assert out.headers["x-api-key"] == "sk-ant-secret"
    assert out.headers["anthropic-version"] == "2023-06-01"
    assert out.json_body == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "top_p": 0.9,
    }
    assert "sk-ant-secret" not in repr(out)


def test_caller_max_tokens_wins_over_the_default() -> None:
    assert _build(Message(role="user", content="hi"), max_tokens=7).json_body["max_tokens"] == 7


def test_system_messages_are_joined_and_removed_from_messages() -> None:
    out = _build(
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
        Message(role="system", content="be kind"),
    )
    assert out.json_body["system"] == "be brief\n\nbe kind"
    assert out.json_body["messages"] == [{"role": "user", "content": "hi"}]


def test_no_system_key_without_system_messages() -> None:
    assert "system" not in _build(Message(role="user", content="hi")).json_body


def test_build_does_not_mutate_the_callers_request() -> None:
    request = _request(Message(role="system", content="s"), Message(role="user", content="hi"))
    before = request.to_dict()
    AnthropicAdapter().build_request(request, api_key="k", base_url=BASE)
    assert request.to_dict() == before


@pytest.mark.parametrize(("stop", "expected"), [("x", ["x"]), (["a", "b"], ["a", "b"])])
def test_stop_becomes_stop_sequences(stop: object, expected: list[str]) -> None:
    body = _build(Message(role="user", content="hi"), stop=stop).json_body
    assert body["stop_sequences"] == expected
    assert "stop" not in body


@pytest.mark.parametrize("kwargs", [{"n": 2}, {"seed": 1}])
def test_unsupported_params_raise_instead_of_being_dropped(kwargs: dict[str, int]) -> None:
    with pytest.raises(errors.UnsupportedParamsError) as info:
        _build(Message(role="user", content="hi"), **kwargs)
    assert next(iter(kwargs)) in info.value.message
    assert info.value.llm_provider == "anthropic"


def test_n_equal_to_one_is_accepted() -> None:
    assert "n" not in _build(Message(role="user", content="hi"), n=1).json_body


def test_unsupported_role_raises_bad_request() -> None:
    with pytest.raises(errors.BadRequestError):
        _build(Message(role="function", content="x"))


def test_parse_response_matches_the_unified_shape() -> None:
    resp = AnthropicAdapter().parse_response(
        _outcome(200, OK, headers={"date": "Thu, 01 Jan 1970 00:00:10 GMT"})
    )
    assert resp.id == "msg_1"
    assert resp.model == "claude-sonnet-5-5"
    assert resp.created == 10
    assert resp.object == "chat.completion"
    assert len(resp.choices) == 1
    choice = resp.choices[0]
    assert (choice.index, choice.message.role, choice.finish_reason) == (0, "assistant", "stop")
    assert choice.message.content == "hello there"
    assert resp.usage is not None
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens) == (
        3,
        4,
        7,
    )


def test_created_is_zero_without_a_usable_date_header() -> None:
    adapter = AnthropicAdapter()
    assert adapter.parse_response(_outcome(200, OK)).created == 0
    assert adapter.parse_response(_outcome(200, OK, headers={"date": "garbage"})).created == 0


def test_naive_date_zone_is_read_as_utc() -> None:
    headers = {"date": "Thu, 01 Jan 1970 00:00:10 -0000"}
    assert AnthropicAdapter().parse_response(_outcome(200, OK, headers=headers)).created == 10


def test_date_header_is_found_case_insensitively() -> None:
    headers = {"Date": "Thu, 01 Jan 1970 00:00:10 GMT"}
    assert AnthropicAdapter().parse_response(_outcome(200, OK, headers=headers)).created == 10


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"), ("odd", "odd")],
)
def test_stop_reason_mapping(reason: str, expected: str) -> None:
    body = {**OK, "stop_reason": reason}
    out = AnthropicAdapter().parse_response(_outcome(200, body))
    assert out.choices[0].finish_reason == expected


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"content": 5}', {**OK, "usage": {}}])
def test_parse_response_invalid_body_is_api_error(body: object) -> None:
    with pytest.raises(errors.APIError) as info:
        AnthropicAdapter().parse_response(_outcome(200, body))
    assert info.value.llm_provider == "anthropic"


def _err(status: int, *, kind: str = "invalid_request_error", msg: str = "bad") -> RawOutcome:
    return _outcome(status, {"type": "error", "error": {"type": kind, "message": msg}})


@pytest.mark.parametrize(
    ("outcome", "cls"),
    [
        (_err(400), errors.BadRequestError),
        (_err(401, kind="authentication_error"), errors.AuthenticationError),
        (_err(403, kind="permission_error"), errors.PermissionDeniedError),
        (_err(404, kind="not_found_error"), errors.NotFoundError),
        (_err(408), errors.Timeout),
        (_err(413, kind="request_too_large"), errors.BadRequestError),
        (_err(429, kind="rate_limit_error"), errors.RateLimitError),
        (_err(500, kind="api_error"), errors.ServiceUnavailableError),
        (_err(529, kind="overloaded_error"), errors.ServiceUnavailableError),
        (_err(418), errors.APIError),
        (RawOutcome(status=None, headers={}, body=b"", failure="timeout"), errors.Timeout),
        (
            RawOutcome(status=None, headers={}, body=b"", failure="connection"),
            errors.APIConnectionError,
        ),
    ],
)
def test_map_error(outcome: RawOutcome, cls: type[errors.PrimaGateError]) -> None:
    err = AnthropicAdapter().map_error(outcome)
    assert type(err) is cls
    assert err.llm_provider == "anthropic"
    assert err.status_code == outcome.status


@pytest.mark.parametrize(
    "msg", ["prompt is too long: 250000 tokens > 200000 maximum", "exceeds the context window"]
)
def test_context_window_failure_maps_to_context_window_exceeded(msg: str) -> None:
    err = AnthropicAdapter().map_error(_err(400, msg=msg))
    assert type(err) is errors.ContextWindowExceededError


def test_map_error_uses_provider_message_and_bounds_non_json_bodies() -> None:
    adapter = AnthropicAdapter()
    assert adapter.map_error(_err(429, msg="slow down")).message == "slow down"
    assert len(adapter.map_error(_outcome(502, b"x" * 2000)).message) == 500


def test_streaming_capability_and_request_flag() -> None:
    assert AnthropicAdapter().capabilities().streaming is True
    assert _build(Message(role="user", content="hi"), stream=True).json_body["stream"] is True
    assert "stream" not in _build(Message(role="user", content="hi")).json_body


def _events(*items: tuple[str, object]) -> list[RawEvent]:
    return [RawEvent(kind, json.dumps({"type": kind, **fields})) for kind, fields in items]


START = (
    "message_start",
    {"message": {"id": "m", "model": "c", "role": "assistant", "usage": {"input_tokens": 4}}},
)


def test_stream_parser_numbers_tool_calls_from_zero_and_skips_unused_blocks() -> None:
    parser = AnthropicAdapter().new_stream_parser({"date": "Thu, 01 Jan 1970 00:00:01 GMT"})
    chunks = []
    for event in _events(
        START,
        ("content_block_start", {"index": 0, "content_block": {"type": "thinking"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}}),
        (
            "content_block_start",
            {"index": 3, "content_block": {"type": "tool_use", "id": "t", "name": "f"}},
        ),
        (
            "content_block_delta",
            {"index": 3, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        ),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
        ("message_stop", {}),
    ):
        chunks += parser.feed(event)
    assert parser.finish() == []
    calls = [c.choices[0].delta.tool_calls for c in chunks[:2]]
    assert [c[0].index for c in calls if c] == [0, 0]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[1].choices[0].delta.role is None
    assert chunks[0].created == 1
    final = chunks[-1]
    assert final.choices[0].finish_reason == "tool_calls"
    assert final.usage is not None and final.usage.total_tokens == 9


def test_stream_parser_maps_in_band_errors_by_type() -> None:
    parser = AnthropicAdapter().new_stream_parser({})
    error = RawEvent(
        "error",
        json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}),
    )
    with pytest.raises(errors.RateLimitError, match="slow"):
        parser.feed(error)


def test_stream_parser_rejects_garbage_and_early_end() -> None:
    parser = AnthropicAdapter().new_stream_parser({})
    with pytest.raises(errors.APIError):
        parser.feed(RawEvent("message", "not json"))
    with pytest.raises(errors.APIError):
        parser.finish()


def test_adapter_module_performs_no_io() -> None:
    tree = ast.parse(Path(anthropic_adapter.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"httpx2", "socket", "urllib", "requests", "os", "subprocess"}


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


def _calls(*pairs: tuple[str, str]) -> list[ToolCall]:
    return [ToolCall(id=i, function=FunctionCall(name="weather", arguments=a)) for i, a in pairs]


def test_tool_definitions_become_input_schema() -> None:
    body = _build(Message(role="user", content="hi"), tools=TOOLS).json_body
    assert body["tools"] == [
        {
            "name": "weather",
            "description": "Look up weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


def test_tool_definition_without_parameters_gets_an_empty_object_schema() -> None:
    tools = [{"type": "function", "function": {"name": "ping"}}]
    body = _build(Message(role="user", content="hi"), tools=tools).json_body
    assert body["tools"] == [{"name": "ping", "input_schema": {"type": "object"}}]


def test_assistant_calls_and_merged_results_become_blocks() -> None:
    body = _build(
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="checking",
            tool_calls=_calls(("toolu_1", '{"city":"A"}'), ("toolu_2", '{"city":"B"}')),
        ),
        Message(role="tool", tool_call_id="toolu_1", content="sunny"),
        Message(role="tool", tool_call_id="toolu_2", content="rainy"),
    ).json_body
    assert body["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "A"}},
            {"type": "tool_use", "id": "toolu_2", "name": "weather", "input": {"city": "B"}},
        ],
    }
    assert body["messages"][2:] == [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "rainy"},
            ],
        }
    ]


def test_assistant_call_without_text_has_no_empty_text_block() -> None:
    body = _build(
        Message(role="assistant", content=None, tool_calls=_calls(("toolu_1", "{}")))
    ).json_body
    assert [b["type"] for b in body["messages"][0]["content"]] == ["tool_use"]


@pytest.mark.parametrize("arguments", ["not json", "[1]", '"s"'])
def test_arguments_that_are_not_a_json_object_are_rejected(arguments: str) -> None:
    with pytest.raises(errors.BadRequestError):
        _build(Message(role="assistant", tool_calls=_calls(("toolu_1", arguments))))


def test_parse_returns_parallel_tool_calls_in_order_with_text() -> None:
    body = {
        **OK,
        "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "A"}},
            {"type": "tool_use", "id": "toolu_2", "name": "weather", "input": {"city": "B"}},
        ],
        "stop_reason": "tool_use",
    }
    choice = AnthropicAdapter().parse_response(_outcome(200, body)).choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "looking"
    assert choice.message.tool_calls is not None
    assert [c.id for c in choice.message.tool_calls] == ["toolu_1", "toolu_2"]
    assert json.loads(choice.message.tool_calls[1].function.arguments) == {"city": "B"}
    assert choice.message.tool_calls[0].type == "function"


def test_tool_use_only_response_has_null_content() -> None:
    body = {
        **OK,
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "w", "input": {}}],
        "stop_reason": "tool_use",
    }
    message = AnthropicAdapter().parse_response(_outcome(200, body)).choices[0].message
    assert message.content is None
    assert message.to_dict()["content"] is None


def test_adapter_declares_tool_support() -> None:
    assert AnthropicAdapter().capabilities().tools is True


def test_non_string_arguments_are_rejected_not_raised_as_type_error() -> None:
    call = ToolCall(id="toolu_1", function=FunctionCall(name="weather", arguments={"a": 1}))  # type: ignore[arg-type]
    with pytest.raises(errors.BadRequestError):
        _build(Message(role="assistant", tool_calls=[call]))


SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "object", "properties": {}}},
        "open": {"type": "object", "additionalProperties": True},
    },
    "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    "anyOf": [{"type": "object", "properties": {}}, {"type": "null"}],
}


def _format(schema: dict[str, object]) -> dict[str, object]:
    return {"type": "json_schema", "json_schema": {"name": "p", "schema": schema}}


def test_json_schema_becomes_output_config_with_closed_objects() -> None:
    original = json.loads(json.dumps(SCHEMA))
    body = _build(Message(role="user", content="hi"), response_format=_format(SCHEMA)).json_body
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    schema = fmt["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["tags"]["items"]["additionalProperties"] is False
    assert schema["$defs"]["Inner"]["additionalProperties"] is False
    assert schema["anyOf"][0]["additionalProperties"] is False
    assert schema["anyOf"][1] == {"type": "null"}
    # An explicit value is the caller's choice.
    assert schema["properties"]["open"]["additionalProperties"] is True
    assert SCHEMA == original


def test_json_object_response_format_is_unsupported() -> None:
    with pytest.raises(errors.UnsupportedParamsError):
        _build(Message(role="user", content="hi"), response_format={"type": "json_object"})


def test_no_response_format_sends_no_output_config() -> None:
    assert "output_config" not in _build(Message(role="user", content="hi")).json_body


def test_structured_output_capability() -> None:
    assert AnthropicAdapter().capabilities().structured_output is True


def test_stream_parser_gives_a_tool_call_without_arguments_an_empty_object() -> None:
    parser = AnthropicAdapter().new_stream_parser({})
    chunks = []
    for event in _events(
        START,
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "f"}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "input_json_delta", "partial_json": ""}},
        ),
        ("content_block_stop", {"index": 0}),
    ):
        chunks += parser.feed(event)
    arguments = "".join(
        c.choices[0].delta.tool_calls[0].function.arguments or ""  # type: ignore[union-attr,index]
        for c in chunks
    )
    assert arguments == "{}"


def test_stream_parser_rejects_a_stop_without_a_stop_reason() -> None:
    parser = AnthropicAdapter().new_stream_parser({})
    parser.feed(_events(START)[0])
    with pytest.raises(errors.APIError, match="stop reason"):
        parser.feed(_events(("message_stop", {}))[0])


def test_images_capability_is_off() -> None:
    assert AnthropicAdapter().capabilities().images is False


def test_declares_the_hint_params_it_cannot_honor() -> None:
    assert AnthropicAdapter().capabilities().unsupported_hints == {"n", "seed"}
    assert OpenAIAdapter().capabilities().unsupported_hints == frozenset()


def _for_model(model: str, **kwargs: object):  # type: ignore[no-untyped-def]
    request = ChatCompletionRequest(
        model=model,
        messages=[Message(role="user", content="hi")],
        **kwargs,  # type: ignore[arg-type]
    )
    return AnthropicAdapter().build_request(request, api_key="sk-ant-secret", base_url=BASE)


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"])
def test_temperature_is_refused_for_models_that_reject_it(model: str) -> None:
    with pytest.raises(errors.UnsupportedParamsError) as info:
        _for_model(model, temperature=0)
    assert "temperature" in info.value.message and model in info.value.message
    assert info.value.llm_provider == "anthropic"


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
    ],
)
def test_temperature_is_still_sent_to_models_that_take_it(model: str) -> None:
    assert _for_model(model, temperature=0.5).json_body["temperature"] == 0.5


def test_a_model_that_rejects_temperature_declares_it_as_a_hint() -> None:
    caps = AnthropicAdapter().capabilities()
    assert caps.hints_for("claude-sonnet-5-5") == {"n", "seed", "temperature"}
    assert caps.hints_for("claude-sonnet-4-5") == {"n", "seed"}
    assert OpenAIAdapter().capabilities().hints_for("gpt-4o") == frozenset()


def test_thinking_is_turned_off_on_request_for_models_that_think_by_default() -> None:
    body = _for_model("claude-haiku-5-5", disable_thinking=True).json_body
    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-4-5"])
def test_thinking_is_left_alone_unless_asked(model: str) -> None:
    assert "thinking" not in _for_model(model).json_body
    assert "thinking" not in _for_model(model, disable_thinking=False).json_body


def test_thinking_is_never_sent_to_a_model_that_does_not_think_by_default() -> None:
    assert "thinking" not in _for_model("claude-sonnet-4-5", disable_thinking=True).json_body


def test_a_date_after_the_family_is_not_read_as_a_version() -> None:
    assert _for_model("claude-sonnet-20250101", temperature=0.5).json_body["temperature"] == 0.5
