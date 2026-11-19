import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import SSE, Dropping, _stream_body
from test_fallback_run import CLAUDE, GPT, MINI, _fallbacks
from test_retry_routing import (
    B_KEY,
    B_URL,
    MSGS,
    PRIMARY,
    Run,
    _answer,
    _status,
)

from primagate.adapters.anthropic import AnthropicAdapter
from primagate.schema import errors
from primagate.schema.wire import Capabilities

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


@pytest.mark.parametrize(
    ("primary", "backup", "primary_host", "backup_host"),
    [(GPT, CLAUDE, OPENAI, ANTHROPIC), (CLAUDE, GPT, ANTHROPIC, OPENAI)],
)
@pytest.mark.fr("FR11")
def test_everything_failing_raises_one_error_holding_every_attempt(
    call: Caller,
    delays: list[float],
    primary: str,
    backup: str,
    primary_host: str,
    backup_host: str,
) -> None:
    routing = {primary: [{"base_url": B_URL, "api_key": B_KEY}]}
    run = Run(
        {primary_host: [_status(503)], "b.test": [_status(503)], backup_host: [_status(429)]},
        provider_routing=routing,
        **_fallbacks(backup),
    )
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=primary, messages=MSGS)
    error = raised.value
    assert [a.model.split("/")[1] for a in error.attempts] == [primary, backup]
    first, second = error.attempts
    assert [d.endpoint for d in first.dispatches] == [first.model.split("/")[0], B_URL]
    original = first.dispatches[0].error
    assert original is not None and original.status_code == 503
    assert isinstance(error.last_error, errors.RateLimitError)
    assert error.status_code == 429 and error.__cause__ is error.last_error
    assert error.attempt_chain == error.last_error.attempt_chain == list(error.attempts)
    assert second.dispatches[0].error is not None
    assert B_KEY not in error.message


@pytest.mark.parametrize("status", [400, 401, 429])
@pytest.mark.fr("FR11")
def test_whatever_class_ends_the_chain_the_aggregate_is_raised(call: Caller, status: int) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_status(status)]}, **_fallbacks(CLAUDE))
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=GPT, messages=MSGS)
    expected = {
        400: errors.BadRequestError,
        401: errors.AuthenticationError,
        429: errors.RateLimitError,
    }[status]
    assert type(raised.value.last_error) is expected
    assert len(raised.value.attempts) == 2


@pytest.mark.fr("FR9", "FR12")
def test_the_outage_scenario_still_gives_an_answer_with_the_chain(
    call: Caller, delays: list[float]
) -> None:
    routing = {GPT: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {OPENAI: [_status(503)], "b.test": [_status(503)], ANTHROPIC: [_answer]}
    run = Run(script, provider_routing=routing, **_fallbacks(CLAUDE))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None
    first, second = result.metadata.attempt_chain
    assert [d.endpoint for d in first.dispatches] == ["openai", B_URL]
    assert all(d.error is not None for d in first.dispatches)
    assert second.dispatches[0].error is None


@pytest.mark.fr("FR11")
def test_a_single_attempt_keeps_its_plain_error_whatever_the_endpoint_count(
    call: Caller, delays: list[float]
) -> None:
    routing = {GPT: [{"base_url": B_URL}]}
    run = Run({OPENAI: [_status(503)], "b.test": [_status(503)]}, provider_routing=routing)
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(run.gate, model=GPT, messages=MSGS)
    assert type(raised.value) is errors.ServiceUnavailableError
    assert len(raised.value.attempt_chain) == 1


@pytest.mark.fr("FR11")
def test_a_lone_raise_class_error_stays_plain_with_fallbacks_declared(call: Caller) -> None:
    run = Run({OPENAI: [_status(400)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    with pytest.raises(errors.BadRequestError) as raised:
        call(run.gate, model=GPT, messages=MSGS)
    assert type(raised.value) is errors.BadRequestError and len(run.sent) == 1


@pytest.mark.fr("FR11")
def test_a_failure_before_any_dispatch_stays_plain_with_an_empty_chain(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_answer]}, **_fallbacks(GPT))
    with pytest.raises(errors.UnsupportedParamsError) as raised:
        call(run.gate, model=CLAUDE, messages=MSGS, tools=[TOOL])
    assert type(raised.value) is errors.UnsupportedParamsError
    assert raised.value.attempt_chain == [] and run.sent == []


@pytest.mark.fr("FR10", "FR11")
def test_one_attempt_then_only_skipped_candidates_gives_the_aggregate(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503)]}, **_fallbacks(CLAUDE))
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    error = raised.value
    assert len(error.attempts) == 1 and len(error.attempt_chain) == 2
    assert error.attempt_chain[1].skipped is not None
    assert isinstance(error.last_error, errors.ServiceUnavailableError)
    assert len(run.sent) == 1


@pytest.mark.fr("FR11")
def test_a_stream_that_fails_to_open_everywhere_raises_the_aggregate(
    stream_call: StreamCaller,
) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_status(503)]}, **_fallbacks(CLAUDE))
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert isinstance(result.call_error, errors.AttemptsExhaustedError)
    assert len(result.call_error.attempts) == 2 and result.chunks == []


@pytest.mark.fr("FR11")
def test_a_stream_that_fails_after_its_first_chunk_ends_with_the_plain_error(
    stream_call: StreamCaller,
) -> None:
    def drops(provider: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(
            200, stream=Dropping(_stream_body(provider, full=False)), headers=SSE
        )

    run = Run(
        {OPENAI: [_status(503), drops], ANTHROPIC: [_status(503)]}, **_fallbacks(CLAUDE, MINI)
    )
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.chunks and type(result.read_error) is errors.APIConnectionError
    assert len(result.read_error.attempt_chain) == 3
