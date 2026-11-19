import logging
import sys
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, SSE, _json_ok, _stream_body
from test_events import Recorder
from test_fallback_run import CLAUDE, GPT, MINI, _fallbacks
from test_retry_routing import MODELS, MSGS, PRIMARY, Run, _status

from primagate import PrimaGate
from primagate.schema import errors, events

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]


def _with_usage(provider: str, model: str | None = None) -> httpx2.Response:
    body = _json_ok(provider)
    if provider == "anthropic":
        body["usage"] = {"input_tokens": 1000, "output_tokens": 500}
    else:
        body["usage"] = {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}
    body["model"] = model or body["model"]
    return httpx2.Response(200, json=body, headers=DATE)


def _answer_with(model: str | None = None) -> Any:
    return lambda provider, stream: _with_usage(provider, model)


def _no_usage(provider: str, stream: bool) -> httpx2.Response:
    body = _json_ok(provider)
    body.pop("usage")
    return httpx2.Response(200, json=body, headers=DATE)


@pytest.mark.parametrize(("provider", "total"), [("openai", 0.0075), ("anthropic", 0.007)])
@pytest.mark.fr("FR33")
def test_a_known_model_reports_counts_and_cost(call: Caller, provider: str, total: float) -> None:
    run = Run({PRIMARY[provider]: [_answer_with()]})
    result = call(run.gate, model=MODELS[provider], messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.unknown_model is False
    assert meta.token_counts is not None and meta.token_counts.total_tokens == 1500
    assert meta.cost is not None and meta.cost.total == pytest.approx(total)
    wire = result.to_dict()["metadata"]
    assert wire["token_counts"]["prompt_tokens"] == 1000 and wire["cost"]["currency"] == "USD"


@pytest.mark.fr("FR33")
def test_a_caller_override_replaces_the_bundled_price(call: Caller) -> None:
    run = Run(
        {OPENAI: [_answer_with()]},
        price_overrides={"gpt-4o": {"input": 100.0, "output": 200.0}},
    )
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.1 + 0.1)


@pytest.mark.fr("FR33")
def test_an_override_prices_a_model_the_bundle_does_not_know(call: Caller) -> None:
    run = Run(
        {OPENAI: [_answer_with()]}, price_overrides={"openai/mystery-1": {"input": 1, "output": 1}}
    )
    result = call(run.gate, model="mystery-1", messages=MSGS)
    assert result.metadata is not None
    assert result.metadata.unknown_model is False and result.metadata.cost is not None


@pytest.mark.fr("FR33")
def test_an_unknown_model_still_answers_and_warns(
    call: Caller, caplog: pytest.LogCaptureFixture
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer_with()]}, event_handler=handler)
    with caplog.at_level(logging.INFO, logger="primagate"):
        result = call(run.gate, model="mystery-1", messages=MSGS)
    meta = result.metadata
    assert result.choices and meta is not None
    assert meta.unknown_model is True and meta.cost is None
    assert meta.token_counts is not None and meta.token_counts.total_tokens == 1500
    assert handler.types == ["dispatch_started", "cost_unknown", "call_succeeded"]
    unknown = next(e for e in handler.seen if isinstance(e, events.CostUnknown))
    assert unknown.model == "openai/mystery-1"
    warned = [r for r in caplog.records if r.name == "primagate" and r.levelno == logging.WARNING]
    assert len(warned) == 1 and warned[0].getMessage().startswith("cost_unknown")


@pytest.mark.fr("FR33")
def test_a_dated_snapshot_is_priced_as_its_base(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with("gpt-4o-2024-08-06")]})
    result = call(run.gate, model="gpt-4o-2024-08-06", messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.0075)


@pytest.mark.fr("FR33")
def test_a_near_name_gets_its_own_price(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with()]})
    result = call(run.gate, model=MINI, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.00045)


@pytest.mark.fr("FR33")
def test_a_fallback_answer_is_priced_as_the_fallback(call: Caller) -> None:
    run = Run({OPENAI: [_status(503)], ANTHROPIC: [_answer_with()]}, **_fallbacks(CLAUDE))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.007)


@pytest.mark.fr("FR33", "FR34")
def test_a_response_without_usage_is_counted_by_the_heuristic_and_priced(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    run = Run({OPENAI: [_no_usage]})
    result = call(run.gate, model=GPT, messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.token_counts is not None and meta.cost is not None
    assert meta.token_counts.heuristic is True and meta.unknown_model is False


@pytest.mark.fr("FR33")
def test_an_estimate_never_changes_what_happens(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with()]}, price_overrides={"gpt-4o": {"input": 1e9, "output": 1e9}})
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.choices and len(run.sent) == 1


def _streamed(provider: str, stream: bool) -> httpx2.Response:
    return httpx2.Response(200, content=_stream_body(provider, full=True), headers=SSE)


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.fr("FR33")
def test_a_stream_carries_the_estimate_on_its_final_chunk(
    stream_call: StreamCaller, provider: str
) -> None:
    run = Run({PRIMARY[provider]: [_streamed]})
    result = stream_call(run.gate, model=MODELS[provider], messages=MSGS)
    assert result.read_error is None and result.chunks
    final = result.chunks[-1]
    assert final.metadata is not None
    counts, cost = final.metadata.token_counts, final.metadata.cost
    assert counts is not None and cost is not None
    assert (counts.prompt_tokens, counts.completion_tokens, counts.total_tokens) == (1, 2, 3)
    # One prompt and two completion tokens at the bundled per-million prices.
    assert cost.total == pytest.approx({"openai": 22.5e-6, "anthropic": 22e-6}[provider])


@pytest.mark.fr("FR33")
def test_a_stream_of_an_unpriced_model_is_flagged_but_raises_no_event(
    stream_call: StreamCaller,
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_streamed]}, event_handler=handler)
    result = stream_call(run.gate, model="mystery-1", messages=MSGS)
    assert result.read_error is None
    final = result.chunks[-1]
    assert final.metadata is not None
    assert final.metadata.unknown_model is True and final.metadata.cost is None
    assert "cost_unknown" not in handler.types


@pytest.mark.fr("FR33")
def test_a_priced_model_raises_no_cost_unknown(call: Caller) -> None:
    handler = Recorder()
    call(Run({OPENAI: [_answer_with()]}, event_handler=handler).gate, model=GPT, messages=MSGS)
    assert "cost_unknown" not in handler.types


@pytest.mark.fr("FR33")
def test_an_override_survives_a_fallback(call: Caller) -> None:
    run = Run(
        {OPENAI: [_status(503)], ANTHROPIC: [_answer_with()]},
        price_overrides={CLAUDE: {"input": 100.0, "output": 200.0}},
        **_fallbacks(CLAUDE),
    )
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.2)


@pytest.mark.fr("FR33")
def test_an_override_survives_a_retry(call: Caller, delays: list[float]) -> None:
    run = Run(
        {OPENAI: [_status(503), _answer_with()]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
        price_overrides={GPT: {"input": 100.0, "output": 200.0}},
    )
    result = call(run.gate, model=GPT, messages=MSGS)
    assert delays
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.2)


@pytest.mark.fr("FR33")
@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_overrides_are_checked_when_the_instance_is_made() -> None:
    with pytest.raises(errors.BadRequestError, match="price_overrides"):
        PrimaGate(price_overrides={"gpt-4o": {"input": -1, "output": 1}})


def _bound_price(model: str, **price: float) -> dict[str, Any]:
    return {"tier_bindings": {"SMALL": {"model": model, "pricing": price}}}


@pytest.mark.fr("FR33")
def test_a_binding_price_replaces_the_bundled_price(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with()]}, **_bound_price(GPT, input=10.0, output=20.0))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.01 + 0.01)


@pytest.mark.fr("FR33")
def test_a_caller_override_beats_a_binding_price(call: Caller) -> None:
    run = Run(
        {OPENAI: [_answer_with()]},
        price_overrides={GPT: {"input": 100.0, "output": 200.0}},
        **_bound_price(GPT, input=10.0, output=20.0),
    )
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.2)


@pytest.mark.fr("FR33")
def test_a_binding_prices_a_model_the_bundle_does_not_know(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with()]}, **_bound_price("mystery-1", input=1.0, output=1.0))
    result = call(run.gate, model="mystery-1", messages=MSGS)
    assert result.metadata is not None
    assert result.metadata.unknown_model is False and result.metadata.cost is not None


@pytest.mark.fr("FR33")
def test_a_binding_with_an_unknown_price_leaves_the_bundled_price(call: Caller) -> None:
    run = Run({OPENAI: [_answer_with()]}, **_bound_price(GPT, input=10.0))
    result = call(run.gate, model=GPT, messages=MSGS)
    assert result.metadata is not None and result.metadata.cost is not None
    assert result.metadata.cost.total == pytest.approx(0.0075)
