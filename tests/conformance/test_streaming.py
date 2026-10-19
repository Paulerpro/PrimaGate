import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx2
import pytest
from conftest import StreamCaller

from primagate import PrimaGate
from primagate.schema import errors

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
HEADERS = {"content-type": "text/event-stream", "date": "Thu, 01 Jan 1970 00:00:01 GMT"}
USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def sse(*events: tuple[str | None, Any]) -> bytes:
    out = ""
    for name, data in events:
        out += f"event: {name}\n" if name else ""
        out += f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
    return out.encode()


def _oa(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> tuple[None, Any]:
    choices = [{"index": 0, "delta": delta, "finish_reason": finish}] if delta or finish else []
    return None, {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "gpt-4o",
        "choices": choices,
        **extra,
    }


def _an(kind: str, **fields: Any) -> tuple[str, Any]:
    return kind, {"type": kind, **fields}


def _tool_delta(**function: Any) -> dict[str, Any]:
    return {"tool_calls": [{"index": 0, **function}]}


OPENAI_TEXT = sse(
    _oa({"role": "assistant", "content": ""}),
    _oa({"content": "Hel"}),
    _oa({"content": "lo"}),
    _oa({}, "stop"),
    _oa({}, usage=USAGE),
    (None, "[DONE]"),
)
OPENAI_TOOLS = sse(
    _oa({"role": "assistant", "content": None}),
    _oa(_tool_delta(id="call_0", type="function", function={"name": "weather", "arguments": ""})),
    _oa(_tool_delta(function={"arguments": '{"city":'})),
    _oa(_tool_delta(function={"arguments": ' "Oslo"}'})),
    _oa({}, "tool_calls"),
    _oa({}, usage=USAGE),
    (None, "[DONE]"),
)
ANTHROPIC_START = _an(
    "message_start",
    message={
        "id": "msg_1",
        "model": "claude-sonnet-5-5",
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 1, "output_tokens": 0},
    },
)
ANTHROPIC_TEXT_OPEN = (
    ANTHROPIC_START,
    _an("content_block_start", index=0, content_block={"type": "text", "text": ""}),
    ("ping", {"type": "ping"}),
    _an("content_block_delta", index=0, delta={"type": "text_delta", "text": "Hel"}),
)
ANTHROPIC_TEXT = sse(
    *ANTHROPIC_TEXT_OPEN,
    _an("content_block_delta", index=0, delta={"type": "text_delta", "text": "lo"}),
    _an("content_block_stop", index=0),
    _an("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 2}),
    _an("message_stop"),
)


def _json_delta(partial: str) -> tuple[str, Any]:
    return _an(
        "content_block_delta",
        index=1,
        delta={"type": "input_json_delta", "partial_json": partial},
    )


ANTHROPIC_TOOLS = sse(
    ANTHROPIC_START,
    _an("content_block_start", index=0, content_block={"type": "text", "text": ""}),
    _an("content_block_stop", index=0),
    _an(
        "content_block_start",
        index=1,
        content_block={"type": "tool_use", "id": "toolu_0", "name": "weather", "input": {}},
    ),
    _json_delta(""),
    _json_delta('{"city":'),
    _json_delta(' "Oslo"}'),
    _an("content_block_stop", index=1),
    _an("message_delta", delta={"stop_reason": "tool_use"}, usage={"output_tokens": 2}),
    _an("message_stop"),
)
BODIES = {
    ("openai", "text"): OPENAI_TEXT,
    ("openai", "tools"): OPENAI_TOOLS,
    ("anthropic", "text"): ANTHROPIC_TEXT,
    ("anthropic", "tools"): ANTHROPIC_TOOLS,
}
CUT = {  # a stream that opens, delivers text, and never reaches its terminal event
    "openai": sse(_oa({"content": "Hel"})),
    "anthropic": sse(*ANTHROPIC_TEXT_OPEN),
}
ERROR_EVENT = {
    "openai": sse((None, {"error": {"message": "boom", "type": "server_error"}})),
    "anthropic": sse(
        ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "boom"}})
    ),
}


class Dropping(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    """Delivers the body, then fails the way a dropped connection does."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __iter__(self) -> Iterator[bytes]:
        yield self._body
        raise httpx2.ReadError("dropped")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body
        raise httpx2.ReadError("dropped")


def _gate(responder: Any) -> tuple[PrimaGate, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        provider = "anthropic" if request.url.host == "api.anthropic.com" else "openai"
        return responder(provider)

    gate = PrimaGate(
        api_keys={"openai": "sk-o", "anthropic": "sk-a"}, transport=httpx2.MockTransport(handle)
    )
    return gate, sent


def _ok(kind: str) -> Any:
    return lambda provider: httpx2.Response(200, content=BODIES[(provider, kind)], headers=HEADERS)


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(v) for v in value]
    return type(value).__name__


def _joined(chunks: list[Any], field_name: str) -> str:
    return "".join(getattr(c.choices[0].delta, field_name) or "" for c in chunks if c.choices)


@pytest.mark.parametrize("provider", MODELS)
def test_text_stream_ends_with_finish_reason_and_usage(
    stream_call: StreamCaller, provider: str
) -> None:
    gate, sent = _gate(_ok("text"))
    run = stream_call(gate, model=MODELS[provider], messages=MSGS)
    assert run.call_error is None and run.read_error is None
    assert sent[0]["stream"] is True
    assert _joined(run.chunks, "content") == "Hello"
    assert run.chunks[0].choices[0].delta.role == "assistant"
    last = run.chunks[-1]
    assert last.choices[0].finish_reason == "stop"
    assert last.usage is not None
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (1, 2)
    assert last.usage.total_tokens == 3
    assert all(c.usage is None for c in run.chunks[:-1])


@pytest.mark.parametrize("provider", MODELS)
def test_tool_call_deltas_join_to_the_call(stream_call: StreamCaller, provider: str) -> None:
    gate, _ = _gate(_ok("tools"))
    run = stream_call(gate, model=MODELS[provider], messages=MSGS, tools=TOOLS)
    deltas = [d for c in run.chunks if c.choices for d in c.choices[0].delta.tool_calls or []]
    assert {d.index for d in deltas} == {0}
    assert deltas[0].id is not None and deltas[0].function is not None
    assert deltas[0].function.name == "weather"
    arguments = "".join(d.function.arguments or "" for d in deltas if d.function)
    assert json.loads(arguments) == {"city": "Oslo"}
    assert run.chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.parametrize("kind", ["text", "tools"])
def test_chunk_shape_is_identical_across_providers(stream_call: StreamCaller, kind: str) -> None:
    gate, _ = _gate(_ok(kind))
    runs = {p: stream_call(gate, model=m, messages=MSGS, tools=TOOLS) for p, m in MODELS.items()}
    for position in (0, -1):
        shapes = {p: _shape(r.chunks[position].to_dict()) for p, r in runs.items()}
        assert shapes["openai"] == shapes["anthropic"]


@pytest.mark.parametrize("provider", MODELS)
def test_http_error_before_the_first_chunk_raises_from_the_call(
    stream_call: StreamCaller, provider: str
) -> None:
    body = {"error": {"message": "slow down", "type": "rate_limit_error", "code": "x"}}
    gate, _ = _gate(lambda p: httpx2.Response(429, json=body))
    run = stream_call(gate, model=MODELS[provider], messages=MSGS)
    assert isinstance(run.call_error, errors.RateLimitError)
    assert run.chunks == [] and run.read_error is None


@pytest.mark.parametrize("provider", MODELS)
def test_in_band_error_as_first_event_raises_from_the_call(
    stream_call: StreamCaller, provider: str
) -> None:
    gate, _ = _gate(lambda p: httpx2.Response(200, content=ERROR_EVENT[p], headers=HEADERS))
    run = stream_call(gate, model=MODELS[provider], messages=MSGS)
    assert isinstance(run.call_error, errors.PrimaGateError)
    assert run.call_error.message == "boom"
    assert run.chunks == []


def test_anthropic_error_right_after_message_start_is_still_before_the_first_chunk(
    stream_call: StreamCaller,
) -> None:
    content = sse(ANTHROPIC_START) + ERROR_EVENT["anthropic"]
    gate, _ = _gate(lambda p: httpx2.Response(200, content=content, headers=HEADERS))
    run = stream_call(gate, model=MODELS["anthropic"], messages=MSGS)
    assert isinstance(run.call_error, errors.ServiceUnavailableError)
    assert run.chunks == []


@pytest.mark.parametrize("provider", MODELS)
def test_error_after_the_first_chunk_ends_the_stream_with_one_error(
    stream_call: StreamCaller, provider: str
) -> None:
    calls: list[int] = []

    def respond(p: str) -> httpx2.Response:
        calls.append(1)
        return httpx2.Response(200, stream=Dropping(CUT[p]), headers=HEADERS)

    gate, _ = _gate(respond)
    run = stream_call(gate, model=MODELS[provider], messages=MSGS)
    assert run.call_error is None
    assert _joined(run.chunks, "content") == "Hel"
    assert isinstance(run.read_error, errors.APIConnectionError)
    assert len(calls) == 1, "a failed stream is never retried"


@pytest.mark.parametrize("provider", MODELS)
def test_stream_without_a_terminal_event_raises_api_error(
    stream_call: StreamCaller, provider: str
) -> None:
    gate, _ = _gate(lambda p: httpx2.Response(200, content=CUT[p], headers=HEADERS))
    run = stream_call(gate, model=MODELS[provider], messages=MSGS)
    assert type(run.read_error) is errors.APIError
    assert _joined(run.chunks, "content") == "Hel"


@pytest.mark.parametrize("provider", MODELS)
def test_closing_early_releases_the_stream_without_error(
    stream_call: StreamCaller, provider: str
) -> None:
    gate, _ = _gate(_ok("text"))
    run = stream_call(gate, stop_after=1, model=MODELS[provider], messages=MSGS)
    assert run.closed_early and run.read_error is None and len(run.chunks) == 1


def test_provider_key_echoed_in_a_stream_error_is_redacted(stream_call: StreamCaller) -> None:
    content = sse((None, {"error": {"message": "bad key sk-o"}}))
    gate, _ = _gate(lambda p: httpx2.Response(200, content=content, headers=HEADERS))
    run = stream_call(gate, model=MODELS["openai"], messages=MSGS)
    assert isinstance(run.call_error, errors.PrimaGateError)
    assert "sk-o" not in run.call_error.message


@pytest.mark.parametrize("provider", MODELS)
def test_reading_after_close_ends_quietly(provider: str) -> None:
    gate, _ = _gate(_ok("text"))
    stream = gate.completion(stream=True, model=MODELS[provider], messages=MSGS)
    next(stream)
    stream.close()
    assert list(stream) == []
