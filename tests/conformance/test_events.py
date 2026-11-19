import asyncio
import logging

import pytest
from conftest import Caller, StreamCaller
from test_fallback_run import CLAUDE, GPT, _fallbacks
from test_retry_routing import (
    B_KEY,
    B_URL,
    MODELS,
    MSGS,
    PRIMARY,
    Run,
    _answer,
    _status,
)

from primagate import PrimaGate
from primagate.adapters.anthropic import AnthropicAdapter
from primagate.schema import errors, events
from primagate.schema.wire import Capabilities

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]


class Recorder:
    def __init__(self) -> None:
        self.seen: list[events.Event] = []

    def __call__(self, event: events.Event) -> None:
        self.seen.append(event)

    @property
    def types(self) -> list[str]:
        return [e.event_type for e in self.seen]


def _logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "primagate"]


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR37")
def test_a_handler_gets_each_event_once_and_the_logger_the_same(
    call: Caller, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    run = Run({PRIMARY[provider]: [_answer]}, event_handler=handler)
    with caplog.at_level(logging.INFO, logger="primagate"):
        call(run.gate, model=MODELS[provider], messages=MSGS)
    assert handler.types == ["dispatch_started", "call_succeeded"]
    lines = _logged(caplog)
    assert len(lines) == 2
    assert [line.split()[0] for line in lines] == handler.types
    assert all(f"request_id={handler.seen[0].request_id}" in line for line in lines)


@pytest.mark.fr("FR37")
def test_events_are_logged_without_a_handler(
    call: Caller, caplog: pytest.LogCaptureFixture
) -> None:
    run = Run({OPENAI: [_answer]})
    with caplog.at_level(logging.INFO, logger="primagate"):
        result = call(run.gate, model=GPT, messages=MSGS)
    assert result.choices and len(_logged(caplog)) == 2


@pytest.mark.fr("FR37")
def test_each_call_has_its_own_id_and_every_event_carries_it(call: Caller) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    call(run.gate, model=GPT, messages=MSGS)
    call(run.gate, model=GPT, messages=MSGS)
    first, second = handler.seen[:2], handler.seen[2:]
    assert len({e.request_id for e in first}) == 1 == len({e.request_id for e in second})
    assert first[0].request_id != second[0].request_id
    assert all(e.request_id.startswith("req-") for e in handler.seen)


@pytest.mark.fr("FR37")
def test_the_driver_stamps_an_iso_utc_timestamp(call: Caller) -> None:
    handler = Recorder()
    call(Run({OPENAI: [_answer]}, event_handler=handler).gate, model=GPT, messages=MSGS)
    for event in handler.seen:
        assert event.timestamp.endswith("+00:00") and "T" in event.timestamp


@pytest.mark.fr("FR37")
def test_retries_and_failover_are_all_traceable(call: Caller, delays: list[float]) -> None:
    handler = Recorder()
    routing = {GPT: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {OPENAI: [_status(503)], "b.test": [_status(503)], ANTHROPIC: [_answer]}
    run = Run(script, provider_routing=routing, event_handler=handler, **_fallbacks(CLAUDE))
    call(run.gate, model=GPT, messages=MSGS)
    failures = [e for e in handler.seen if isinstance(e, events.DispatchFailed)]
    assert [f.next for f in failures] == ["retry", "retry", "next_endpoint"] + [
        "retry",
        "retry",
        "next_attempt",
    ]
    assert handler.types[-1] == "call_succeeded"
    assert len({e.request_id for e in handler.seen}) == 1


@pytest.mark.fr("FR37")
def test_a_total_failure_ends_with_call_failed(call: Caller) -> None:
    handler = Recorder()
    run = Run(
        {OPENAI: [_status(503)], ANTHROPIC: [_status(429)]},
        event_handler=handler,
        **_fallbacks(CLAUDE),
    )
    with pytest.raises(errors.AttemptsExhaustedError):
        call(run.gate, model=GPT, messages=MSGS)
    final = handler.seen[-1]
    assert isinstance(final, events.CallFailed) and final.attempts == 2


@pytest.mark.fr("FR37")
def test_a_call_that_fails_before_dispatch_emits_one_call_failed(call: Caller) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    with pytest.raises(errors.PrimaGateError):
        call(run.gate, model="nope/x", messages=MSGS)
    assert handler.types == ["call_failed"]
    assert run.sent == []


@pytest.mark.fr("FR37")
def test_a_stream_failing_before_its_first_chunk_emits_the_same_events(
    stream_call: StreamCaller,
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_status(503)]}, event_handler=handler, max_retries=0)
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert isinstance(result.call_error, errors.ServiceUnavailableError)
    assert handler.types == ["dispatch_started", "dispatch_failed", "call_failed"]


@pytest.mark.fr("FR37")
def test_a_stream_that_opens_reports_its_start_and_nothing_more(
    stream_call: StreamCaller,
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.chunks and result.read_error is None
    assert handler.types == ["dispatch_started"]


@pytest.mark.fr("FR37")
def test_a_stream_that_is_retried_keeps_one_id(
    stream_call: StreamCaller, delays: list[float]
) -> None:
    handler = Recorder()
    run = Run(
        {OPENAI: [_status(503), _answer]},
        event_handler=handler,
        provider_routing={GPT: [{"base_url": B_URL}]},
    )
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.chunks
    assert handler.types == [
        "dispatch_started",
        "dispatch_failed",
        "retry_scheduled",
        "dispatch_started",
    ]
    assert len({e.request_id for e in handler.seen}) == 1


@pytest.mark.fr("FR10", "FR37")
def test_a_skipped_fallback_is_reported_before_the_next_start(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    handler = Recorder()
    run = Run(
        {OPENAI: [_status(503), _answer]},
        event_handler=handler,
        **_fallbacks(CLAUDE, "gpt-4o-mini"),
    )
    call(run.gate, model=GPT, messages=MSGS, tools=[tool])
    assert handler.types == [
        "dispatch_started",
        "dispatch_failed",
        "attempt_skipped",
        "dispatch_started",
        "call_succeeded",
    ]
    skipped = handler.seen[2]
    assert isinstance(skipped, events.AttemptSkipped) and "tools" in skipped.reason


@pytest.mark.fr("FR37")
def test_a_raising_handler_is_logged_and_the_call_goes_on(
    call: Caller, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    def broken(event: events.Event) -> None:
        calls.append(event.event_type)
        raise ValueError("handler bug")

    run = Run({OPENAI: [_answer]}, event_handler=broken)
    with caplog.at_level(logging.INFO, logger="primagate"):
        result = call(run.gate, model=GPT, messages=MSGS)
    assert result.choices
    assert calls == ["dispatch_started", "call_succeeded"]
    failures = [r for r in caplog.records if r.exc_info and r.name == "primagate"]
    assert len(failures) == 2 and "handler bug" in str(failures[0].exc_info)


@pytest.mark.fr("FR37")
@pytest.mark.mode_independent("task cancellation exists only in async mode")
def test_cancellation_inside_a_handler_propagates() -> None:
    def cancelling(event: events.Event) -> None:
        raise asyncio.CancelledError

    run = Run({OPENAI: [_answer]}, event_handler=cancelling)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run.gate.acompletion(model=GPT, messages=MSGS))


@pytest.mark.fr("FR37")
@pytest.mark.mode_independent("asserts the sync driver; async mode has its own cancellation test")
def test_a_handler_that_raises_a_base_exception_propagates_in_sync_mode() -> None:
    def interrupting(event: events.Event) -> None:
        raise KeyboardInterrupt

    run = Run({OPENAI: [_answer]}, event_handler=interrupting)
    with pytest.raises(KeyboardInterrupt):
        run.gate.completion(model=GPT, messages=MSGS)


@pytest.mark.fr("FR37")
@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_a_handler_must_be_a_plain_function() -> None:
    async def async_handler(event: events.Event) -> None: ...

    with pytest.raises(errors.BadRequestError, match="event_handler"):
        PrimaGate(event_handler="not callable")  # type: ignore[arg-type]
    with pytest.raises(errors.BadRequestError, match="plain function"):
        PrimaGate(event_handler=async_handler)  # type: ignore[arg-type]


def test_no_key_appears_in_any_event_or_log_line(
    call: Caller, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    routing = {GPT: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {OPENAI: [_status(401, "bad key sk-o")], "b.test": [_answer]}
    run = Run(script, provider_routing=routing, event_handler=handler)
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        call(run.gate, model=GPT, messages=MSGS)
    text = repr(handler.seen) + " ".join(_logged(caplog))
    assert "sk-o" not in text and B_KEY not in text
