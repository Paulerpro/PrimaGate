import random
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, SSE, Dropping, _event, _json_ok, _stream_body

from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.types import DispatchRecord

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
PRIMARY = {"openai": "api.openai.com", "anthropic": "api.anthropic.com"}
MSGS = [{"role": "user", "content": "hi"}]
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}
B_URL = "https://b.test/v1"
B_KEY = "sk-b"

Reply = Callable[[str, bool], httpx2.Response]


def _answer(provider: str, stream: bool) -> httpx2.Response:
    if stream:
        return httpx2.Response(200, content=_stream_body(provider, full=True), headers=SSE)
    return httpx2.Response(200, json=_json_ok(provider), headers=DATE)


def _status(code: int, message: str = "nope") -> Reply:
    openai = {"error": {"message": message, "type": "server_error", "code": "x"}}
    anthropic = {"type": "error", "error": {"type": "api_error", "message": message}}

    def reply(provider: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(code, json=anthropic if provider == "anthropic" else openai)

    return reply


class Run:
    """A gate over a scripted mock: each host plays its replies in order, the last repeating."""

    def __init__(self, script: dict[str, list[Reply]], **kwargs: Any) -> None:
        self.sent: list[httpx2.Request] = []
        self._script = script
        self._played: dict[str, int] = {}
        self.gate = PrimaGate(api_keys=KEYS, transport=httpx2.MockTransport(self._handle), **kwargs)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(request)
        host = request.url.host
        provider = "anthropic" if request.url.path.endswith("/messages") else "openai"
        replies = self._script[host]
        index = min(self._played.get(host, 0), len(replies) - 1)
        self._played[host] = self._played.get(host, 0) + 1
        stream = b'"stream":true' in request.content.replace(b" ", b"")
        return replies[index](provider, stream)

    @property
    def hosts(self) -> list[str]:
        return [r.url.host for r in self.sent]


def _routed(provider: str) -> dict[str, Any]:
    return {"provider_routing": {MODELS[provider]: [{"base_url": B_URL, "api_key": B_KEY}]}}


@pytest.mark.usefixtures("delays")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_a_retry_that_succeeds_stays_on_the_endpoint(
    call: Caller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(503), _answer]}, **_routed(provider))
    result = call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider]] * 2
    assert delays == [0.5]
    assert result.metadata is not None
    (attempt,) = result.metadata.attempt_chain
    assert attempt.dispatches == [DispatchRecord(provider, 1)]


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12", "FR14")
def test_spent_retries_move_on_to_the_redundant_endpoint(
    call: Caller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(503)], "b.test": [_answer]}, **_routed(provider))
    result = call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider]] * 3 + ["b.test"]
    assert delays == [0.5, 1.0]
    assert result.metadata is not None
    (attempt,) = result.metadata.attempt_chain
    first, second = attempt.dispatches
    assert (first.endpoint, first.retries) == (provider, 2)
    assert first.error is not None and first.error.status_code == 503
    assert (second.endpoint, second.retries, second.error) == (B_URL, 0, None)


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_when_everything_fails_the_last_error_is_raised_with_both_rows(
    call: Caller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(503)], "b.test": [_status(429)]}, **_routed(provider))
    with pytest.raises(errors.RateLimitError) as raised:
        call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider]] * 3 + ["b.test"] * 3
    assert delays == [0.5, 1.0, 0.5, 1.0]
    (attempt,) = raised.value.attempt_chain
    assert [d.endpoint for d in attempt.dispatches] == [provider, B_URL]
    assert [d.retries for d in attempt.dispatches] == [2, 2]


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_an_advance_error_skips_the_retries(
    call: Caller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(401)], "b.test": [_answer]}, **_routed(provider))
    call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider], "b.test"]
    assert delays == []


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_a_raise_error_never_reaches_the_redundant_endpoint(
    call: Caller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(400)], "b.test": [_answer]}, **_routed(provider))
    with pytest.raises(errors.BadRequestError):
        call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider]]
    assert delays == []


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR13")
def test_routing_off_makes_one_request_without_sleep_or_randomness(
    call: Caller, provider: str, delays: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden() -> float:
        raise AssertionError("random source read with Provider Routing off")

    monkeypatch.setattr(random, "random", forbidden)
    run = Run({PRIMARY[provider]: [_status(503)]})
    with pytest.raises(errors.ServiceUnavailableError):
        call(run.gate, model=MODELS[provider], messages=MSGS)
    assert len(run.sent) == 1 and delays == []


@pytest.mark.fr("FR12")
def test_delays_follow_the_configured_settings(call: Caller, delays: list[float]) -> None:
    run = Run(
        {PRIMARY["openai"]: [_status(503)], "b.test": [_answer]},
        max_retries=4,
        backoff_base=0.5,
        backoff_max=1.5,
        **_routed("openai"),
    )
    call(run.gate, model="gpt-4o", messages=MSGS)
    assert delays == [0.5, 1.0, 1.5, 1.5]


@pytest.mark.fr("FR12")
def test_jitter_scales_each_delay(
    call: Caller, delays: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(random, "random", lambda: 0.0)
    run = Run({PRIMARY["openai"]: [_status(503)], "b.test": [_answer]}, **_routed("openai"))
    call(run.gate, model="gpt-4o", messages=MSGS)
    assert delays == [0.25, 0.5]


@pytest.mark.fr("FR12")
def test_zero_retries_goes_straight_to_the_redundant_endpoint(
    call: Caller, delays: list[float]
) -> None:
    run = Run(
        {PRIMARY["openai"]: [_status(503)], "b.test": [_answer]},
        max_retries=0,
        **_routed("openai"),
    )
    call(run.gate, model="gpt-4o", messages=MSGS)
    assert run.hosts == [PRIMARY["openai"], "b.test"] and delays == []


@pytest.mark.usefixtures("delays")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR14")
def test_the_redundant_endpoint_uses_its_own_key_and_never_leaks_it(
    call: Caller, provider: str
) -> None:
    echo = _status(400, f"bad key {B_KEY}")
    run = Run({PRIMARY[provider]: [_status(401)], "b.test": [echo]}, **_routed(provider))
    with pytest.raises(errors.BadRequestError) as raised:
        call(run.gate, model=MODELS[provider], messages=MSGS)
    assert B_KEY in " ".join(run.sent[1].headers.values())
    assert B_KEY not in raised.value.message
    texts = [d.error.message for a in raised.value.attempt_chain for d in a.dispatches if d.error]
    assert not any(B_KEY in t for t in texts)


@pytest.mark.usefixtures("delays")
@pytest.mark.fr("FR14")
def test_an_endpoint_without_a_key_uses_the_providers_key(call: Caller) -> None:
    routing = {"provider_routing": {"gpt-4o": [{"base_url": B_URL, "api_key": ""}]}}
    run = Run({PRIMARY["openai"]: [_status(401)], "b.test": [_answer]}, **routing)
    call(run.gate, model="gpt-4o", messages=MSGS)
    assert "sk-o" in " ".join(run.sent[1].headers.values())


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_a_stream_that_fails_to_open_is_retried_then_failed_over(
    stream_call: StreamCaller, provider: str, delays: list[float]
) -> None:
    run = Run({PRIMARY[provider]: [_status(503)], "b.test": [_answer]}, **_routed(provider))
    result = stream_call(run.gate, model=MODELS[provider], messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.hosts == [PRIMARY[provider]] * 3 + ["b.test"]
    assert delays == [0.5, 1.0]
    final = result.chunks[-1]
    assert final.metadata is not None
    first, second = final.metadata.attempt_chain[0].dispatches
    assert (first.retries, second.endpoint) == (2, B_URL)


@pytest.mark.usefixtures("delays")
@pytest.mark.fr("FR12")
def test_an_error_event_before_the_first_chunk_is_retried(stream_call: StreamCaller) -> None:
    def broken(provider: str, stream: bool) -> httpx2.Response:
        body = _event(None, {"error": {"message": "boom", "type": "server_error"}})
        return httpx2.Response(200, content=body.encode(), headers=SSE)

    run = Run({PRIMARY["openai"]: [broken, _answer]}, **_routed("openai"))
    result = stream_call(run.gate, model="gpt-4o", messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.hosts == [PRIMARY["openai"]] * 2


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_a_stream_is_never_retried_after_its_first_chunk(
    stream_call: StreamCaller, provider: str, delays: list[float]
) -> None:
    def drops(p: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(200, stream=Dropping(_stream_body(p, full=False)), headers=SSE)

    run = Run({PRIMARY[provider]: [drops], "b.test": [_answer]}, **_routed(provider))
    result = stream_call(run.gate, model=MODELS[provider], messages=MSGS)
    assert result.chunks and isinstance(result.read_error, errors.APIConnectionError)
    assert len(run.sent) == 1 and delays == []
    (attempt,) = result.read_error.attempt_chain
    assert len(attempt.dispatches) == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"max_retries": True},
        {"backoff_base": 0},
        {"backoff_base": -1.0},
        {"backoff_base": 2.0, "backoff_max": 1.0},
        {"backoff_max": "8"},
        {"backoff_base": float("nan")},
        {"backoff_max": float("inf")},
    ],
)
@pytest.mark.fr("FR12")
@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_bad_retry_settings_are_rejected_at_setup(settings: dict[str, Any]) -> None:
    with pytest.raises(errors.BadRequestError, match=next(iter(settings))):
        PrimaGate(api_keys=KEYS, **settings)


def _in_band_error(message: str) -> Reply:
    def reply(provider: str, stream: bool) -> httpx2.Response:
        if provider == "anthropic":
            body = _event(
                "error", {"type": "error", "error": {"type": "api_error", "message": message}}
            )
        else:
            body = _event(None, {"error": {"message": message, "type": "server_error"}})
        return httpx2.Response(200, content=body.encode(), headers=SSE)

    return reply


@pytest.mark.usefixtures("delays")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.parametrize("failure", ["status", "first_event"])
def test_the_redundant_key_never_leaks_from_a_stream(
    stream_call: StreamCaller, provider: str, failure: str
) -> None:
    echo = f"bad key {B_KEY}"
    reply = _status(400, echo) if failure == "status" else _in_band_error(echo)
    run = Run({PRIMARY[provider]: [_status(401)], "b.test": [reply]}, **_routed(provider))
    result = stream_call(run.gate, model=MODELS[provider], messages=MSGS)
    error = result.call_error
    assert isinstance(error, errors.PrimaGateError)
    assert B_KEY not in error.message
    texts = [d.error.message for a in error.attempt_chain for d in a.dispatches if d.error]
    assert len(texts) == 2 and not any(B_KEY in t for t in texts)


@pytest.mark.usefixtures("delays")
@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12")
def test_a_drop_after_failover_reports_every_endpoint_tried(
    stream_call: StreamCaller, provider: str
) -> None:
    def drops(p: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(200, stream=Dropping(_stream_body(p, full=False)), headers=SSE)

    run = Run({PRIMARY[provider]: [_status(503)], "b.test": [drops]}, **_routed(provider))
    result = stream_call(run.gate, model=MODELS[provider], messages=MSGS)
    assert isinstance(result.read_error, errors.APIConnectionError)
    (attempt,) = result.read_error.attempt_chain
    assert [d.endpoint for d in attempt.dispatches] == [provider, B_URL]
    assert run.hosts.count("b.test") == 1


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR12", "FR14")
def test_several_redundant_endpoints_are_tried_in_order(
    call: Caller, provider: str, delays: list[float]
) -> None:
    routing = {
        "provider_routing": {
            MODELS[provider]: [
                {"base_url": B_URL, "api_key": B_KEY},
                {"base_url": "https://c.test/v1", "api_key": "sk-c"},
            ]
        }
    }
    script = {PRIMARY[provider]: [_status(401)], "b.test": [_status(503)], "c.test": [_answer]}
    run = Run(script, max_retries=1, **routing)
    result = call(run.gate, model=MODELS[provider], messages=MSGS)
    assert run.hosts == [PRIMARY[provider], "b.test", "b.test", "c.test"]
    assert delays == [0.5]
    assert result.metadata is not None
    rows = result.metadata.attempt_chain[0].dispatches
    assert [(d.endpoint, d.retries) for d in rows] == [
        (provider, 0),
        (B_URL, 1),
        ("https://c.test/v1", 0),
    ]
