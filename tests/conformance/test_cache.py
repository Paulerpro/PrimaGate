import asyncio
import logging
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, SSE, _json_ok, _stream_body
from test_events import Recorder
from test_fallback_run import CLAUDE, GPT, _fallbacks
from test_retry_routing import MODELS, PRIMARY, Run, _status

from primagate.schema import errors, events

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]
ASKED, ANSWERED = "CACHE-SECRET-QUESTION", "CACHE-SECRET-ANSWER"
MSGS = [{"role": "user", "content": ASKED}]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _answer(provider: str, stream: bool) -> httpx2.Response:
    if stream:
        return httpx2.Response(200, content=_stream_body(provider, full=True), headers=SSE)
    body = _json_ok(provider)
    if provider == "anthropic":
        body["content"][0]["text"] = ANSWERED
    else:
        body["choices"][0]["message"]["content"] = ANSWERED
    return httpx2.Response(200, json=body, headers=DATE)


@pytest.mark.parametrize("provider", MODELS)
def test_the_second_identical_call_is_served_from_memory(call: Caller, provider: str) -> None:
    run = Run({PRIMARY[provider]: [_answer]})
    first = call(run.gate, model=MODELS[provider], messages=MSGS)
    second = call(run.gate, model=MODELS[provider], messages=MSGS)
    assert len(run.sent) == 1
    assert first.metadata is not None and first.metadata.cache_hit is False
    assert second.metadata is not None and second.metadata.cache_hit is True
    assert second.choices == first.choices and second.id == first.id


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.5},
        {"max_tokens": 5},
        {"messages": [{"role": "user", "content": "another"}]},
        {"tools": [TOOL]},
        {"response_format": {"type": "json_object"}},
        {"model": "gpt-4o-mini"},
        {"seed": 3},
        {"drop_params": True},
        {"drop_params": ["tools"]},
    ],
)
def test_changing_any_key_field_is_a_miss(call: Caller, change: dict[str, Any]) -> None:
    run = Run({OPENAI: [_answer]})
    base: dict[str, Any] = {"model": GPT, "messages": MSGS}
    call(run.gate, **base)
    result = call(run.gate, **{**base, **change})
    assert len(run.sent) == 2
    assert result.metadata is not None and result.metadata.cache_hit is False


def test_calls_that_differ_only_in_drop_params_are_separate_entries(call: Caller) -> None:
    run = Run({OPENAI: [_answer]})
    call(run.gate, model=GPT, messages=MSGS, drop_params=True)
    call(run.gate, model=GPT, messages=MSGS)
    call(run.gate, model=GPT, messages=MSGS, drop_params=True)
    call(run.gate, model=GPT, messages=MSGS)
    assert len(run.sent) == 2


def test_callers_never_share_a_mutable_object_with_the_cache(call: Caller) -> None:
    run = Run({OPENAI: [_answer]})
    first = call(run.gate, model=GPT, messages=MSGS)
    first.choices[0].message.content = "scribbled on by the first caller"
    second = call(run.gate, model=GPT, messages=MSGS)
    assert second.choices[0].message.content == ANSWERED
    second.choices[0].message.content = "scribbled on by the second caller"
    assert second.metadata is not None
    second.metadata.cache_hit = False
    third = call(run.gate, model=GPT, messages=MSGS)
    assert third.choices[0].message.content == ANSWERED
    assert third.metadata is not None and third.metadata.cache_hit is True
    assert len(run.sent) == 1


def test_a_hit_says_nothing_was_dispatched_and_emits_only_a_cache_event(call: Caller) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    call(run.gate, model=GPT, messages=MSGS)
    handler.seen.clear()
    result = call(run.gate, model=GPT, messages=MSGS)
    meta = result.metadata
    assert meta is not None
    assert (meta.attempt_chain, meta.token_counts, meta.cost) == ([], None, None)
    assert meta.unknown_model is False
    assert handler.types == ["cache_hit"]
    (hit,) = handler.seen
    assert isinstance(hit, events.CacheHit) and hit.model == f"openai/{GPT}"
    assert len(hit.key) == 64


@pytest.mark.parametrize("retention", [False, True])
def test_cached_content_never_reaches_events_or_logs(
    call: Caller, caplog: pytest.LogCaptureFixture, retention: bool
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler, payload_retention=retention)
    call(run.gate, model=GPT, messages=MSGS)
    handler.seen.clear()
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        call(run.gate, model=GPT, messages=MSGS)
    text = repr(handler.seen) + "".join(r.getMessage() for r in caplog.records)
    assert "cache_hit" in text
    assert ASKED not in text and ANSWERED not in text


def test_a_failed_call_is_not_cached(call: Caller) -> None:
    run = Run({OPENAI: [_status(400), _answer]})
    with pytest.raises(errors.BadRequestError):
        call(run.gate, model=GPT, messages=MSGS)
    result = call(run.gate, model=GPT, messages=MSGS)
    assert len(run.sent) == 2
    assert result.metadata is not None and result.metadata.cache_hit is False


def test_an_invalid_request_raises_every_time_and_never_touches_the_cache(call: Caller) -> None:
    run = Run({OPENAI: [_answer]})
    for _ in range(2):
        with pytest.raises(errors.BadRequestError):
            call(run.gate, model="nope/x", messages=MSGS)
    assert run.sent == []


def test_streams_neither_read_nor_write_the_cache(call: Caller, stream_call: StreamCaller) -> None:
    run = Run({OPENAI: [_answer]})
    call(run.gate, model=GPT, messages=MSGS)
    streamed = stream_call(run.gate, model=GPT, messages=MSGS)
    assert streamed.read_error is None and streamed.chunks
    assert len(run.sent) == 2, "a stream is never served from the cache"
    final = streamed.chunks[-1]
    assert final.metadata is not None and final.metadata.cache_hit is False
    after = call(run.gate, model=GPT, messages=MSGS)
    assert len(run.sent) == 2, "the stream did not replace the cached answer"
    assert after.metadata is not None and after.metadata.cache_hit is True


def test_a_stream_alone_leaves_nothing_behind(call: Caller, stream_call: StreamCaller) -> None:
    run = Run({OPENAI: [_answer]})
    stream_call(run.gate, model=GPT, messages=MSGS)
    call(run.gate, model=GPT, messages=MSGS)
    assert len(run.sent) == 2


def test_the_sync_and_async_drivers_share_one_cache() -> None:
    run = Run({OPENAI: [_answer]})
    run.gate.completion(model=GPT, messages=MSGS)
    result = asyncio.run(run.gate.acompletion(model=GPT, messages=MSGS))
    assert len(run.sent) == 1
    assert result.metadata is not None and result.metadata.cache_hit is True


def test_each_instance_has_its_own_cache() -> None:
    first, second = Run({OPENAI: [_answer]}), Run({OPENAI: [_answer]})
    first.gate.completion(model=GPT, messages=MSGS)
    second.gate.completion(model=GPT, messages=MSGS)
    assert len(first.sent) == 1 and len(second.sent) == 1


def test_a_fallback_answer_is_cached_under_the_original_request(call: Caller) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    first = call(run.gate, model=GPT, messages=MSGS)
    second = call(run.gate, model=GPT, messages=MSGS)
    assert len(run.sent) == 2
    assert first.model == second.model
    assert second.metadata is not None and second.metadata.cache_hit is True


def test_a_counted_answer_is_cached_with_its_counts_left_out_of_the_hit(call: Caller) -> None:
    def no_usage(provider: str, stream: bool) -> httpx2.Response:
        body = _json_ok(provider)
        body.pop("usage")
        return httpx2.Response(200, json=body, headers=DATE)

    run = Run({OPENAI: [no_usage]})
    first = call(run.gate, model=GPT, messages=MSGS)
    second = call(run.gate, model=GPT, messages=MSGS)
    assert first.metadata is not None and first.metadata.token_counts is not None
    assert second.metadata is not None and second.metadata.token_counts is None
    assert len(run.sent) == 1


def test_a_hit_reaches_the_handler_exactly_once(call: Caller) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    call(run.gate, model=GPT, messages=MSGS)
    call(run.gate, model=GPT, messages=MSGS)
    assert handler.types == ["dispatch_started", "call_succeeded", "cache_hit"]


def test_a_call_that_needed_a_retry_is_cached_once_it_succeeds(
    call: Caller, delays: list[float]
) -> None:
    run = Run(
        {OPENAI: [_status(503), _answer]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
    )
    call(run.gate, model=GPT, messages=MSGS)
    sent = len(run.sent)
    second = call(run.gate, model=GPT, messages=MSGS)
    assert sent == 2 and len(run.sent) == 2
    assert second.metadata is not None and second.metadata.cache_hit is True
