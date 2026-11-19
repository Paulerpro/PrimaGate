import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller

from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.types import Attempt, DispatchRecord, ErrorInfo

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}
MSGS = [{"role": "user", "content": "hi"}]
DATE = {"date": "Thu, 01 Jan 1970 00:00:01 GMT"}
SSE = {"content-type": "text/event-stream", **DATE}


def _event(name: str | None, data: Any) -> str:
    return (f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n"


def _json_ok(provider: str) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "id": "msg_1",
            "model": MODELS[provider],
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    return {
        "id": "c",
        "created": 1,
        "model": MODELS[provider],
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


def _openai_chunk(delta: dict[str, Any], finish: str | None = None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return _event(None, {"id": "c", "created": 1, "model": "gpt-4o", "choices": [choice]})


def _openai_usage() -> str:
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    return _event(None, {"id": "c", "created": 1, "model": "gpt-4o", "choices": [], "usage": usage})


def _anthropic_open() -> str:
    start = {
        "id": "m",
        "model": MODELS["anthropic"],
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 1},
    }
    block = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "ok"},
    }
    return (
        _event("message_start", {"type": "message_start", "message": start})
        + _event("content_block_start", block)
        + _event("content_block_delta", delta)
    )


def _anthropic_close() -> str:
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    return _event("message_delta", delta) + _event("message_stop", {"type": "message_stop"})


def _stream_body(provider: str, *, full: bool) -> bytes:
    """One text chunk, then (when `full`) the terminal events."""
    if provider == "anthropic":
        return (_anthropic_open() + (_anthropic_close() if full else "")).encode()
    tail = _openai_chunk({}, "stop") + _openai_usage() + "data: [DONE]\n\n"
    return (_openai_chunk({"content": "ok"}) + (tail if full else "")).encode()


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


def _gate(responder: Any) -> PrimaGate:
    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        provider = "anthropic" if request.url.host == "api.anthropic.com" else "openai"
        return responder(provider, bool(body.get("stream")))

    return PrimaGate(api_keys=KEYS, transport=httpx2.MockTransport(handle))


def _answer(provider: str, stream: bool) -> httpx2.Response:
    if stream:
        return httpx2.Response(200, content=_stream_body(provider, full=True), headers=SSE)
    return httpx2.Response(200, json=_json_ok(provider), headers=DATE)


def _fail(status: int, message: str = "slow down") -> Any:
    openai = {"error": {"message": message, "type": "rate_limit_error", "code": "x"}}
    anthropic = {"type": "error", "error": {"type": "rate_limit_error", "message": message}}
    return lambda provider, stream: httpx2.Response(
        status, json=anthropic if provider == "anthropic" else openai
    )


def _row(provider: str, error: ErrorInfo | None = None) -> Attempt:
    return Attempt(f"{provider}/{MODELS[provider]}", [DispatchRecord(provider, 0, error)])


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR11")
def test_a_success_reports_one_attempt_with_one_dispatch(call: Caller, provider: str) -> None:
    result = call(_gate(_answer), model=MODELS[provider], messages=MSGS)
    assert result.metadata is not None
    assert result.metadata.attempt_chain == [_row(provider)]
    dispatch = result.to_dict()["metadata"]["attempt_chain"][0]["dispatches"][0]
    assert dispatch == {"endpoint": provider, "retries": 0}


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR11")
def test_a_failure_raises_the_plain_error_with_the_chain(call: Caller, provider: str) -> None:
    with pytest.raises(errors.RateLimitError) as raised:
        call(_gate(_fail(429)), model=MODELS[provider], messages=MSGS)
    info = ErrorInfo("RateLimitError", "slow down", 429)
    assert raised.value.attempt_chain == [_row(provider, info)]


@pytest.mark.fr("FR11")
def test_a_failure_before_any_dispatch_has_an_empty_chain(call: Caller) -> None:
    gate = _gate(_answer)
    with pytest.raises(errors.BadRequestError) as unknown:
        call(gate, model="nope/x", messages=MSGS)
    assert unknown.value.attempt_chain == []
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]
    with pytest.raises(errors.UnsupportedParamsError) as unsupported:
        call(gate, model="claude-sonnet-5-5", messages=image)
    assert unsupported.value.attempt_chain == []


@pytest.mark.parametrize("provider", MODELS)
def test_the_key_never_reaches_the_chain(call: Caller, provider: str) -> None:
    gate = _gate(_fail(401, "bad key sk-o sk-a"))
    with pytest.raises(errors.AuthenticationError) as raised:
        call(gate, model=MODELS[provider], messages=MSGS)
    _assert_no_key(raised.value, KEYS[provider])


@pytest.mark.parametrize("provider", MODELS)
def test_the_key_never_reaches_the_chain_of_a_stream_that_fails_to_open(
    stream_call: StreamCaller, provider: str
) -> None:
    run = stream_call(_gate(_fail(401, "bad key sk-o sk-a")), model=MODELS[provider], messages=MSGS)
    assert isinstance(run.call_error, errors.AuthenticationError)
    _assert_no_key(run.call_error, KEYS[provider])


def test_the_key_never_reaches_the_chain_of_an_in_band_stream_error(
    stream_call: StreamCaller,
) -> None:
    def respond(provider: str, stream: bool) -> httpx2.Response:
        error = _event(None, {"error": {"message": "bad key sk-o", "type": "server_error"}})
        body = (_openai_chunk({"content": "ok"}) + error).encode()
        return httpx2.Response(200, content=body, headers=SSE)

    run = stream_call(_gate(respond), model="gpt-4o", messages=MSGS)
    assert isinstance(run.read_error, errors.PrimaGateError)
    _assert_no_key(run.read_error, "sk-o")


def _assert_no_key(error: errors.PrimaGateError, key: str) -> None:
    info = error.attempt_chain[0].dispatches[0].error
    assert info is not None and info.message
    assert key not in info.message


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR11")
def test_the_final_stream_chunk_carries_the_chain(stream_call: StreamCaller, provider: str) -> None:
    run = stream_call(_gate(_answer), model=MODELS[provider], messages=MSGS)
    assert run.call_error is None and run.read_error is None
    assert all(c.metadata is None for c in run.chunks[:-1])
    final = run.chunks[-1]
    assert final.metadata is not None and final.metadata.attempt_chain == [_row(provider)]


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR11")
def test_a_stream_that_fails_to_open_raises_with_the_chain(
    stream_call: StreamCaller, provider: str
) -> None:
    run = stream_call(_gate(_fail(429)), model=MODELS[provider], messages=MSGS)
    assert isinstance(run.call_error, errors.PrimaGateError)
    info = ErrorInfo("RateLimitError", "slow down", 429)
    assert run.call_error.attempt_chain == [_row(provider, info)]


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR11")
def test_a_mid_stream_failure_carries_the_chain(stream_call: StreamCaller, provider: str) -> None:
    def respond(p: str, stream: bool) -> httpx2.Response:
        body = _stream_body(p, full=False)
        return httpx2.Response(200, stream=Dropping(body), headers=SSE)

    run = stream_call(_gate(respond), model=MODELS[provider], messages=MSGS)
    assert isinstance(run.read_error, errors.APIConnectionError)
    chain = run.read_error.attempt_chain
    assert len(chain) == 1 and chain[0].dispatches[0].error is not None
    assert chain[0].dispatches[0].error.type == "APIConnectionError"
