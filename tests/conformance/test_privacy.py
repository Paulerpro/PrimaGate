import logging
import traceback
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, SSE, _json_ok, _stream_body
from test_events import Recorder
from test_retry_routing import (
    B_KEY,
    B_URL,
    MODELS,
    PRIMARY,
    Run,
    _status,
)

from primagate import PrimaGate
from primagate.schema import errors, events

ASKED, ANSWERED = "SECRET-QUESTION-TEXT", "SECRET-ANSWER-TEXT"
MSGS = [{"role": "user", "content": ASKED}]


def _secret_answer(provider: str, stream: bool) -> httpx2.Response:
    if stream:
        return httpx2.Response(200, content=_stream_body(provider, full=True), headers=SSE)
    body = _json_ok(provider)
    if provider == "anthropic":
        body["content"][0]["text"] = ANSWERED
    else:
        body["choices"][0]["message"]["content"] = ANSWERED
    return httpx2.Response(200, json=body, headers=DATE)


def _seen(handler: Recorder, caplog: pytest.LogCaptureFixture) -> str:
    lines = [r.getMessage() for r in caplog.records if r.name == "primagate"]
    return repr(handler.seen) + "\n".join(lines)


@pytest.mark.parametrize("provider", MODELS)
def test_by_default_no_event_or_log_line_holds_the_conversation(
    call: Caller, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    run = Run({PRIMARY[provider]: [_secret_answer]}, event_handler=handler)
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        result = call(run.gate, model=MODELS[provider], messages=MSGS)
    assert result.choices[0].message.content == ANSWERED
    text = _seen(handler, caplog)
    assert ASKED not in text and ANSWERED not in text
    started, done = handler.seen[0], handler.seen[-1]
    assert isinstance(started, events.DispatchStarted) and started.messages is None
    assert isinstance(done, events.CallSucceeded) and done.reply_messages is None


@pytest.mark.parametrize("provider", MODELS)
def test_opting_in_puts_both_sides_of_the_conversation_in_events_and_logs(
    call: Caller, provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    run = Run({PRIMARY[provider]: [_secret_answer]}, event_handler=handler, payload_retention=True)
    with caplog.at_level(logging.INFO, logger="primagate"):
        call(run.gate, model=MODELS[provider], messages=MSGS)
    started, done = handler.seen[0], handler.seen[-1]
    assert isinstance(started, events.DispatchStarted) and isinstance(done, events.CallSucceeded)
    assert started.messages == ({"role": "user", "content": ASKED},)
    assert done.reply_messages is not None
    assert done.reply_messages[0]["content"] == ANSWERED
    lines = [r.getMessage() for r in caplog.records if r.name == "primagate"]
    assert any(line.startswith("dispatch_started") and ASKED in line for line in lines)
    assert any(line.startswith("call_succeeded") and ANSWERED in line for line in lines)


def test_every_dispatch_carries_the_request_when_retried(
    call: Caller, delays: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    routing = {"gpt-4o": [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {PRIMARY["openai"]: [_status(503)], "b.test": [_secret_answer]}
    run = Run(script, provider_routing=routing, event_handler=handler, payload_retention=True)
    call(run.gate, model="gpt-4o", messages=MSGS)
    started = [e for e in handler.seen if isinstance(e, events.DispatchStarted)]
    assert len(started) > 1 and all(s.messages is not None for s in started)


def test_error_text_is_never_retained_even_when_opted_in(call: Caller) -> None:
    handler = Recorder()
    run = Run(
        {PRIMARY["openai"]: [_status(400, f"you sent {ASKED}")]},
        event_handler=handler,
        payload_retention=True,
    )
    with pytest.raises(errors.BadRequestError):
        call(run.gate, model="gpt-4o", messages=MSGS)
    failures = [e for e in handler.seen if not isinstance(e, events.DispatchStarted)]
    assert failures and ASKED not in repr(failures)


@pytest.mark.parametrize("value", ["yes", 1, 0, None, []])
def test_payload_retention_must_be_a_bool(value: Any) -> None:
    with pytest.raises(errors.BadRequestError, match="payload_retention"):
        PrimaGate(payload_retention=value)


def test_a_stream_follows_the_same_rules(
    stream_call: StreamCaller, caplog: pytest.LogCaptureFixture
) -> None:
    quiet, open_ = Recorder(), Recorder()
    script = {PRIMARY["openai"]: [_secret_answer]}
    with caplog.at_level(logging.INFO, logger="primagate"):
        stream_call(Run(script, event_handler=quiet).gate, model="gpt-4o", messages=MSGS)
        assert ASKED not in "".join(r.getMessage() for r in caplog.records)
        stream_call(
            Run(script, event_handler=open_, payload_retention=True).gate,
            model="gpt-4o",
            messages=MSGS,
        )
    assert ASKED not in repr(quiet.seen)
    assert ASKED in "".join(r.getMessage() for r in caplog.records)
    first = open_.seen[0]
    assert isinstance(first, events.DispatchStarted) and first.messages is not None


def test_credentials_in_an_endpoint_url_never_reach_events_or_logs(
    call: Caller, delays: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    url = "https://user:pw-secret@b.test/v1?api-key=QUERY-SECRET#frag"
    routing = {"gpt-4o": [{"base_url": url}]}
    script = {PRIMARY["openai"]: [_status(503)], "b.test": [_secret_answer]}
    run = Run(script, provider_routing=routing, event_handler=handler)
    with caplog.at_level(logging.INFO, logger="primagate"):
        call(run.gate, model="gpt-4o", messages=MSGS)
    text = _seen(handler, caplog)
    assert "pw-secret" not in text and "QUERY-SECRET" not in text
    endpoints = {e.endpoint for e in handler.seen if isinstance(e, events.DispatchStarted)}
    assert endpoints == {"openai", "https://b.test/v1"}


def _leaks(secret: str, *places: object) -> list[str]:
    return [type(p).__name__ for p in places if secret in str(p) or secret in repr(p)]


@pytest.mark.parametrize("retention", [False, True])
@pytest.mark.parametrize("via", ["argument", "environment"])
def test_a_key_echoed_by_the_provider_appears_nowhere(
    call: Caller,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    delays: list[float],
    retention: bool,
    via: str,
) -> None:
    key = "sk-leak-canary-12345"
    redundant = "sk-redundant-canary-678"
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    if via == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", key)
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        # A server can only echo the credential it was sent.
        sent = request.headers.get("authorization", "") + request.headers.get("x-api-key", "")
        used = redundant if request.url.host == "b.test" else key
        assert used in sent
        echoed = {"error": {"message": f"bad credentials {used}", "type": "x"}}
        status = 401 if request.url.host == "api.openai.com" else 503
        return httpx2.Response(status, json=echoed)

    handler = Recorder()
    gate = PrimaGate(
        api_keys={"openai": key} if via == "argument" else None,
        provider_routing={"gpt-4o": [{"base_url": B_URL, "api_key": redundant}]},
        default_fallbacks=["gpt-4o-mini"],
        event_handler=handler,
        payload_retention=retention,
        transport=httpx2.MockTransport(handle),
    )
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        with pytest.raises(errors.PrimaGateError) as raised:
            call(gate, model="gpt-4o", messages=MSGS)
    error = raised.value
    rendered = "".join(traceback.format_exception(error))
    rows = [d for a in error.attempt_chain for d in a.dispatches]
    places = [
        error,
        rendered,
        gate,
        _seen(handler, caplog),
        [str(r.exc_info) for r in caplog.records],
        rows,
        getattr(error, "last_error", None),
    ]
    for secret in (key, redundant):
        assert not _leaks(secret, *places), secret
    assert seen, "the scenario must have reached the provider"


def test_a_key_never_shows_in_the_instance_repr() -> None:
    gate = PrimaGate(api_keys={"openai": "sk-repr-canary"}, payload_retention=True)
    assert "sk-repr-canary" not in repr(gate) and "sk-repr-canary" not in str(gate)


def test_a_failing_credentialed_endpoint_leaks_nothing_anywhere(
    call: Caller, delays: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    url = "https://user:pw-secret@b.test/v1?api-key=QUERY-SECRET"
    routing = {"gpt-4o": [{"base_url": url}]}
    run = Run(
        {PRIMARY["openai"]: [_status(503)], "b.test": [_status(503)]},
        provider_routing=routing,
        event_handler=handler,
    )
    with caplog.at_level(logging.INFO, logger="primagate"):
        with pytest.raises(errors.ServiceUnavailableError) as raised:
            call(run.gate, model="gpt-4o", messages=MSGS)
    error = raised.value
    rows = [d for a in error.attempt_chain for d in a.dispatches]
    places = [error, "".join(traceback.format_exception(error)), rows, _seen(handler, caplog)]
    for secret in ("pw-secret", "QUERY-SECRET"):
        assert not _leaks(secret, *places), secret
    shown = {e.endpoint for e in handler.seen if hasattr(e, "endpoint")}
    assert shown == {"openai", "https://b.test/v1"}
    assert [d.endpoint for d in rows] == ["openai", "https://b.test/v1"]
    assert any(isinstance(e, events.RetryScheduled) for e in handler.seen)
