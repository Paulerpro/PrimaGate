import json
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import SSE, Dropping, _stream_body
from test_retry_routing import (
    B_KEY,
    B_URL,
    MODELS,
    MSGS,
    PRIMARY,
    Run,
    _answer,
    _in_band_error,
    _status,
)

from primagate.adapters.anthropic import AnthropicAdapter
from primagate.schema import errors
from primagate.schema.wire import Capabilities

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]
CLAUDE, GPT, MINI, FOUR = "claude-sonnet-5-5", "gpt-4o", "gpt-4o-mini", "gpt-4.1"
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _models(run: Run) -> list[str]:
    return [json.loads(r.content)["model"] for r in run.sent]


def _fallbacks(*names: str) -> dict[str, Any]:
    return {"default_fallbacks": list(names)}


@pytest.mark.parametrize(
    ("primary", "backup", "primary_host", "backup_host"),
    [(GPT, CLAUDE, OPENAI, ANTHROPIC), (CLAUDE, GPT, ANTHROPIC, OPENAI)],
)
def test_a_backup_answers_when_the_primary_fails(
    call: Caller, primary: str, backup: str, primary_host: str, backup_host: str
) -> None:
    run = Run({primary_host: [_status(503)], backup_host: [_answer]}, **_fallbacks(backup))
    result = call(run.gate, model=primary, messages=MSGS)
    assert run.hosts == [primary_host, backup_host]
    assert _models(run) == [primary, backup]
    assert result.metadata is not None
    first, second = result.metadata.attempt_chain
    assert first.model.endswith(primary) and second.model.endswith(backup)
    assert first.dispatches[0].error is not None and second.dispatches[0].error is None


def test_the_backup_is_sent_to_its_own_provider_with_its_own_key(call: Caller) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    call(run.gate, model=GPT, messages=MSGS)
    assert "sk-o" in run.sent[0].headers["authorization"]
    assert run.sent[1].headers["x-api-key"] == "sk-a"
    assert str(run.sent[1].url).endswith("/messages")


@pytest.mark.usefixtures("delays")
def test_entries_are_tried_strictly_in_order(call: Caller) -> None:
    # Scripts are per host and replayed in call order, so the OpenAI host answers the
    # primary, then gpt-4o-mini, then gpt-4.1, in that order.
    script = {OPENAI: [_status(503), _status(503), _answer], ANTHROPIC: [_status(503)]}
    run = Run(script, **_fallbacks(CLAUDE, MINI, FOUR))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert _models(run) == [GPT, CLAUDE, MINI, FOUR]
    assert result.metadata is not None and len(result.metadata.attempt_chain) == 4


def test_a_per_model_list_replaces_the_default(call: Caller) -> None:
    run = Run(
        {OPENAI: [_status(503), _answer], ANTHROPIC: [_answer]},
        fallbacks={GPT: [MINI]},
        default_fallbacks=[CLAUDE],
    )
    call(run.gate, model=GPT, messages=MSGS)
    assert _models(run) == [GPT, MINI]


def test_the_primarys_routing_is_exhausted_before_the_first_fallback(
    call: Caller, delays: list[float]
) -> None:
    routing = {GPT: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {OPENAI: [_status(503)], "b.test": [_status(503)], ANTHROPIC: [_answer]}
    run = Run(script, provider_routing=routing, **_fallbacks(CLAUDE))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI] * 3 + ["b.test"] * 3 + [ANTHROPIC]
    assert delays == [0.5, 1.0, 0.5, 1.0]
    assert result.metadata is not None
    first, second = result.metadata.attempt_chain
    assert [d.endpoint for d in first.dispatches] == ["openai", B_URL]
    assert [d.endpoint for d in second.dispatches] == ["anthropic"]


def test_a_fallback_has_its_own_routing_and_retries_before_the_next_entry(
    call: Caller, delays: list[float]
) -> None:
    routing = {CLAUDE: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {
        OPENAI: [_status(503), _answer],
        ANTHROPIC: [_status(503)],
        "b.test": [_status(503)],
    }
    run = Run(script, provider_routing=routing, **_fallbacks(CLAUDE, MINI))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI, *[ANTHROPIC] * 3, *["b.test"] * 3, OPENAI]
    assert delays == [0.5, 1.0, 0.5, 1.0]
    assert result.metadata is not None
    assert [len(a.dispatches) for a in result.metadata.attempt_chain] == [1, 2, 1]


def test_the_primary_is_not_retried_just_because_a_fallback_is_routed(
    call: Caller, delays: list[float]
) -> None:
    routing = {CLAUDE: [{"base_url": B_URL}]}
    run = Run(
        {OPENAI: [_status(503)], ANTHROPIC: [_answer]},
        provider_routing=routing,
        **_fallbacks(CLAUDE),
    )
    call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI, ANTHROPIC] and delays == []


@pytest.mark.parametrize("failing", ["primary", "fallback"])
def test_a_raise_error_ends_the_chain(call: Caller, failing: str) -> None:
    openai = [_status(400)] if failing == "primary" else [_status(503)]
    anthropic = [_status(400)] if failing == "fallback" else [_answer]
    run = Run({OPENAI: openai, ANTHROPIC: anthropic}, **_fallbacks(CLAUDE, MINI))
    with pytest.raises(errors.BadRequestError) as raised:
        call(run.gate, model=GPT, messages=MSGS)
    assert _models(run) == ([GPT] if failing == "primary" else [GPT, CLAUDE])
    assert len(raised.value.attempt_chain) == len(run.sent)


def test_an_advance_error_moves_to_the_fallback_without_a_delay(
    call: Caller, delays: list[float]
) -> None:
    run = Run({OPENAI: [_status(401)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI, ANTHROPIC] and delays == []


@pytest.mark.usefixtures("delays")
@pytest.mark.parametrize("provider", MODELS)
def test_when_everything_fails_the_last_error_is_raised_with_every_attempt(
    call: Caller, provider: str
) -> None:
    other = "anthropic" if provider == "openai" else "openai"
    run = Run(
        {PRIMARY[provider]: [_status(503)], PRIMARY[other]: [_status(429)]},
        **_fallbacks(MODELS[other]),
    )
    with pytest.raises(errors.RateLimitError) as raised:
        call(run.gate, model=MODELS[provider], messages=MSGS)
    chain = raised.value.attempt_chain
    assert [a.model.split("/")[0] for a in chain] == [provider, other]
    assert chain[0].dispatches[0].error is not None
    assert chain[1].dispatches[0].error is not None
    assert chain[1].dispatches[0].error.type == "RateLimitError"


def test_a_fallback_that_lacks_a_needed_feature_is_passed_over(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503), _answer]}, **_fallbacks(CLAUDE, MINI))
    result = call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert _models(run) == [GPT, MINI]
    assert result.metadata is not None
    chain = result.metadata.attempt_chain
    assert [a.skipped is None for a in chain] == [True, False, True]
    assert chain[1].model == "anthropic/claude-sonnet-5-5" and chain[1].dispatches == []
    reason = chain[1].skipped
    assert reason is not None and "tools" in reason and "anthropic" in reason
    wire = result.to_dict()["metadata"]["attempt_chain"]
    assert "skipped" not in wire[0] and wire[1]["skipped"] == reason


def test_every_incapable_fallback_leaves_a_row_and_the_primarys_error_is_raised(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503)]}, **_fallbacks(CLAUDE))
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert len(run.sent) == 1
    chain = raised.value.attempt_chain
    assert [a.skipped is None for a in chain] == [True, False]


def test_drop_params_lets_an_incapable_fallback_run_instead_of_being_skipped(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    result = call(run.gate, model=GPT, messages=MSGS, tools=[TOOL], drop_params=["tools"])
    assert _models(run) == [GPT, CLAUDE]
    assert "tools" not in json.loads(run.sent[1].content)
    assert result.metadata is not None
    assert result.metadata.dropped_params == ["tools"]
    assert all(a.skipped is None for a in result.metadata.attempt_chain)


def test_a_skipped_row_reaches_a_streams_final_chunk(
    stream_call: StreamCaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503), _answer]}, **_fallbacks(CLAUDE, MINI))
    result = stream_call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert result.call_error is None and result.read_error is None
    final = result.chunks[-1]
    assert final.metadata is not None
    assert [a.skipped is None for a in final.metadata.attempt_chain] == [True, False, True]


def test_a_skipped_row_reaches_a_stream_that_fails_to_open(
    stream_call: StreamCaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503)]}, **_fallbacks(CLAUDE))
    result = stream_call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert isinstance(result.call_error, errors.ServiceUnavailableError)
    assert [a.skipped is None for a in result.call_error.attempt_chain] == [True, False]


def test_a_primary_that_lacks_the_feature_raises_first_with_an_empty_chain(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_answer], ANTHROPIC: [_answer]}, **_fallbacks(GPT))
    with pytest.raises(errors.UnsupportedParamsError) as raised:
        call(run.gate, model=CLAUDE, messages=MSGS, tools=[TOOL])
    assert run.sent == [] and raised.value.attempt_chain == []


def test_the_fallbacks_provider_decides_which_params_are_dropped(call: Caller) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    result = call(run.gate, model=GPT, messages=MSGS, seed=3, drop_params=True)
    assert "seed" in json.loads(run.sent[0].content)
    assert "seed" not in json.loads(run.sent[1].content)
    assert result.metadata is not None and result.metadata.dropped_params == ["seed"]


@pytest.mark.parametrize("failure", ["status", "first_event"])
def test_a_stream_that_fails_before_its_first_chunk_moves_to_the_fallback(
    stream_call: StreamCaller, failure: str
) -> None:
    bad = _status(503) if failure == "status" else _in_band_error("boom")
    run = Run({OPENAI: [bad], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.hosts == [OPENAI, ANTHROPIC]
    final = result.chunks[-1]
    assert final.metadata is not None and len(final.metadata.attempt_chain) == 2


def test_a_stream_is_never_moved_to_a_fallback_after_its_first_chunk(
    stream_call: StreamCaller,
) -> None:
    def drops(provider: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(
            200, stream=Dropping(_stream_body(provider, full=False)), headers=SSE
        )

    run = Run({OPENAI: [drops], ANTHROPIC: [_answer]}, **_fallbacks(CLAUDE))
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.chunks and isinstance(result.read_error, errors.APIConnectionError)
    assert run.hosts == [OPENAI]
    assert len(result.read_error.attempt_chain) == 1


def test_a_second_fallback_stream_that_fails_early_moves_on_again(
    stream_call: StreamCaller,
) -> None:
    run = Run(
        {OPENAI: [_status(503), _answer], ANTHROPIC: [_in_band_error("boom")]},
        **_fallbacks(CLAUDE, MINI),
    )
    result = stream_call(run.gate, model=GPT, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert _models(run) == [GPT, CLAUDE, MINI]
    final = result.chunks[-1]
    assert final.metadata is not None
    assert [a.model for a in final.metadata.attempt_chain] == [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-5-5",
        "openai/gpt-4o-mini",
    ]


@pytest.mark.parametrize("status", [503, 400])
def test_a_skipped_row_stays_in_place_when_the_next_fallback_fails(
    call: Caller, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    run = Run({OPENAI: [_status(503), _status(status)]}, **_fallbacks(CLAUDE, MINI))
    with pytest.raises(errors.PrimaGateError) as raised:
        call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert _models(run) == [GPT, MINI]
    chain = raised.value.attempt_chain
    assert [a.skipped is None for a in chain] == [True, False, True]
    assert chain[1].model == "anthropic/claude-sonnet-5-5"


def test_a_skipped_row_stays_in_place_when_a_stream_fails_after_its_first_chunk(
    stream_call: StreamCaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())

    def drops(provider: str, stream: bool) -> httpx2.Response:
        return httpx2.Response(
            200, stream=Dropping(_stream_body(provider, full=False)), headers=SSE
        )

    run = Run({OPENAI: [_status(503), drops]}, **_fallbacks(CLAUDE, MINI))
    result = stream_call(run.gate, model=GPT, messages=MSGS, tools=[TOOL])
    assert result.chunks and isinstance(result.read_error, errors.APIConnectionError)
    chain = result.read_error.attempt_chain
    assert [a.skipped is None for a in chain] == [True, False, True]
