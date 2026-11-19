from typing import Any

import pytest
from conftest import Caller
from test_retry_routing import B_KEY, B_URL, _answer, _status
from test_tier_failover import FOUR, GPT, MINI, POLICY, THREE, TOOL, Run
from test_tier_routing import MSGS

from primagate.adapters.anthropic import AnthropicAdapter
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.wire import Capabilities

ALIAS = TIER_ROUTING_ALIAS
F, D, CLAUDE = "gpt-4.1-mini", "o3", "claude-sonnet-5-5"
NO_MEDIUM = {**POLICY, "enabled_tiers": ["SMALL", "POWERFUL"], "interim_binding": "SMALL"}


def _rows(chain: list[Any]) -> list[tuple[str, bool]]:
    """Each row as (model, was it sent)."""
    return [(row.model, row.skipped is None) for row in chain]


@pytest.mark.fr("FR30")
def test_the_chain_runs_resolved_tier_then_medium_then_the_fallback_list(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F] and len(run.analyzed) == 1
    meta = result.metadata
    assert meta is not None and meta.tier is None
    assert _rows(meta.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", True),
        (f"openai/{F}", True),
    ]
    assert [e.next for e in run.of(events.DispatchFailed)] == ["next_attempt"] * 2
    assert [s.model for s in run.of(events.DispatchStarted)] == [
        f"openai/{MINI}",
        f"openai/{GPT}",
        f"openai/{F}",
    ]
    (done,) = run.of(events.CallSucceeded)
    assert done.model == f"openai/{F}" and done.attempts == 3
    assert len(run.decisions) == 1


@pytest.mark.fr("FR30")
def test_a_failed_medium_choice_goes_straight_to_the_fallback_list(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"], {GPT: [_status(503)]}, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [GPT, F]
    assert result.metadata is not None and result.metadata.tier is None
    assert run.of(events.AttemptSkipped) == []


@pytest.mark.fr("FR30")
def test_a_skipped_medium_is_followed_by_the_fallback_list(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, policy=NO_MEDIUM, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, F]
    assert result.metadata is not None
    assert _rows(result.metadata.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", False),
        (f"openai/{F}", True),
    ]
    (skip,) = run.of(events.AttemptSkipped)
    assert skip.reason == "MEDIUM is not in enabled_tiers"


@pytest.mark.fr("FR30")
def test_a_listed_model_that_already_ran_as_the_failover_is_not_run_again(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[GPT, F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F]
    assert result.metadata is not None
    assert all(sent for _, sent in _rows(result.metadata.attempt_chain))
    assert run.of(events.AttemptSkipped) == []


@pytest.mark.parametrize("listed", [MINI, f"openai/{MINI}"])
@pytest.mark.fr("FR30")
def test_a_listed_model_that_is_the_resolved_tier_is_not_run_again(
    call: Caller, listed: str
) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[listed, F])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F]
    assert run.of(events.AttemptSkipped) == []


@pytest.mark.fr("FR30")
def test_a_listed_medium_model_still_runs_when_medium_was_only_skipped(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, policy=NO_MEDIUM, default_fallbacks=[GPT])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    meta = result.metadata
    assert meta is not None and meta.tier is None
    assert _rows(meta.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", False),
        (f"openai/{GPT}", True),
    ]


@pytest.mark.fr("FR10", "FR25")
def test_a_fallback_is_judged_by_its_providers_capabilities_not_a_tiers_limits(
    call: Caller,
) -> None:
    limited = {**THREE, "MEDIUM": {"model": GPT, "limits": {"supported_features": ["streaming"]}}}
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, bindings=limited, default_fallbacks=[GPT])
    result = call(run.gate, model=ALIAS, messages=MSGS, tools=[TOOL], drop_params=False)
    assert run.models == [MINI, GPT]
    # The Tier's limits removed MEDIUM from failover; the same model as a fallback keeps tools.
    assert "tools" in run.workers[1]
    assert result.metadata is not None
    _, passed, answered = result.metadata.attempt_chain
    assert passed.skipped is not None and "tools" in passed.skipped
    assert answered.skipped is None and result.metadata.dropped_params == []


@pytest.mark.fr("FR10")
def test_an_incapable_fallback_is_a_skipped_row_and_the_next_is_tried(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[CLAUDE, F])
    result = call(run.gate, model=ALIAS, messages=MSGS, tools=[TOOL])
    assert run.models == [MINI, GPT, F]
    assert result.metadata is not None
    assert _rows(result.metadata.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", True),
        (f"anthropic/{CLAUDE}", False),
        (f"openai/{F}", True),
    ]
    (skip,) = run.of(events.AttemptSkipped)
    assert skip.model == f"anthropic/{CLAUDE}" and "tools" in skip.reason


@pytest.mark.fr("FR11", "FR30")
def test_when_every_layer_fails_the_error_holds_the_whole_chain_in_order(call: Caller) -> None:
    down = [_status(503)]
    script = {MINI: down, GPT: down, F: down, D: down}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F, D])
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    # The worst case: one resolved-Tier Attempt, one MEDIUM Attempt and the declared entries.
    assert run.models == [MINI, GPT, F, D]
    assert isinstance(raised.value.last_error, errors.ServiceUnavailableError)
    chain = raised.value.attempt_chain
    assert _rows(chain) == [(f"openai/{m}", True) for m in (MINI, GPT, F, D)]
    assert all(row.dispatches[0].error is not None for row in chain)
    (failed,) = run.of(events.CallFailed)
    assert failed.attempts == 4


@pytest.mark.fr("FR10", "FR11")
def test_the_chain_with_skips_keeps_them_in_place_when_everything_fails(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AnthropicAdapter, "capabilities", lambda self: Capabilities())
    down = [_status(503)]
    run = Run(
        ["SMALL 0.9"],
        {MINI: down, F: down},
        policy=NO_MEDIUM,
        default_fallbacks=[CLAUDE, F],
    )
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS, tools=[TOOL])
    assert run.models == [MINI, F]
    assert _rows(raised.value.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", False),
        (f"anthropic/{CLAUDE}", False),
        (f"openai/{F}", True),
    ]
    assert len(raised.value.attempts) == 2


@pytest.mark.fr("FR30")
def test_a_raise_class_error_on_medium_stops_before_the_fallback_list(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(400)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F])
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    assert type(raised.value.last_error) is errors.BadRequestError
    assert len(raised.value.attempt_chain) == 2


@pytest.mark.fr("FR30")
def test_a_raise_class_error_on_the_resolved_tier_stops_everything(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(400)]}, default_fallbacks=[F])
    with pytest.raises(errors.BadRequestError):
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI]


@pytest.mark.fr("FR7")
def test_the_resolved_tiers_own_list_replaces_the_default(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, fallbacks={MINI: [F]}, default_fallbacks=[D])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F]


@pytest.mark.fr("FR7")
def test_another_models_list_is_not_used_so_the_default_applies(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    # A list keyed by MEDIUM's model belongs to MEDIUM as a plain call, not to this chain.
    run = Run(["SMALL 0.9"], script, fallbacks={GPT: [F]}, default_fallbacks=[D])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, D]


@pytest.mark.fr("FR22", "FR31")
def test_a_degraded_call_uses_the_interim_tiers_list_and_its_answer_is_not_cached(
    call: Caller,
) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(
        ["nonsense"],
        script,
        policy={**POLICY, "interim_binding": "SMALL"},
        fallbacks={MINI: [F]},
        default_fallbacks=[D],
    )
    first = call(run.gate, model=ALIAS, messages=MSGS)
    assert [e.reason for e in run.of(events.RoutingDegraded)] == ["malformed_output"]
    assert run.models == [MINI, GPT, F]
    assert first.metadata is not None and first.metadata.tier is None
    again = call(run.gate, model=ALIAS, messages=MSGS)
    assert again.metadata is not None and not again.metadata.cache_hit
    assert run.models == [MINI, GPT, F] * 2 and len(run.analyzed) == 2


@pytest.mark.fr("FR31")
def test_a_routed_fallback_answer_is_cached_like_any_routed_answer(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F])
    call(run.gate, model=ALIAS, messages=MSGS)
    again = call(run.gate, model=ALIAS, messages=MSGS)
    assert again.metadata is not None and again.metadata.cache_hit
    assert run.models == [MINI, GPT, F] and len(run.analyzed) == 1


@pytest.mark.fr("FR35")
def test_the_spend_counts_the_fallback_answer_and_the_one_analyzer_cost(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    policy = {**POLICY, "spending_ceiling": 1.0}
    run = Run(["SMALL 0.9"], script, policy=policy, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.tier is None and meta.cost is not None
    assert meta.analyzer is not None and meta.analyzer.cost is not None
    assert run.gate._spend.total == pytest.approx(meta.cost.total + meta.analyzer.cost.total)


@pytest.mark.fr("FR35")
def test_a_plain_call_with_a_fallback_list_still_counts_no_spend(call: Caller) -> None:
    policy = {**POLICY, "spending_ceiling": 1.0}
    run = Run(["SMALL 0.9"], {GPT: [_status(503)]}, policy=policy, default_fallbacks=[F])
    call(run.gate, model=GPT, messages=MSGS)
    assert run.models == [GPT, F] and run.analyzed == []
    assert run.gate._spend.total == 0


@pytest.mark.usefixtures("delays")
@pytest.mark.fr("FR9")
def test_each_fallback_gets_its_own_provider_routing_and_retries(call: Caller) -> None:
    routing = {F: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {MINI: [_status(503)], GPT: [_status(503)], F: [_status(503)] * 3 + [_answer]}
    run = Run(["SMALL 0.9"], script, provider_routing=routing, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT] + [F] * 4
    assert run.hosts[-1] == "b.test"
    assert result.metadata is not None
    *_, fallback = result.metadata.attempt_chain
    assert [d.endpoint for d in fallback.dispatches] == ["openai", B_URL]
    assert [d.retries for d in fallback.dispatches] == [2, 0]


@pytest.mark.fr("FR30")
def test_a_powerful_tier_chain_has_the_same_order(call: Caller) -> None:
    script = {FOUR: [_status(401)], GPT: [_status(503)]}
    run = Run(["POWERFUL 0.9"], script, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [FOUR, GPT, F]
    assert result.metadata is not None and result.metadata.tier is None


@pytest.mark.fr("FR35")
def test_a_fallback_answer_that_crosses_the_ceiling_sends_the_warning(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    policy = {**POLICY, "spending_ceiling": 0.000001}
    run = Run(["SMALL 0.9"], script, policy=policy, default_fallbacks=[F])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F]
    meta = result.metadata
    assert meta is not None and meta.cost is not None
    assert meta.analyzer is not None and meta.analyzer.cost is not None
    (warning,) = run.of(events.SpendingCeilingExceeded)
    assert warning.ceiling == 0.000001
    assert warning.total == pytest.approx(meta.cost.total + meta.analyzer.cost.total)


@pytest.mark.fr("FR7")
def test_an_empty_list_for_the_resolved_model_keeps_the_default_off(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = Run(["SMALL 0.9"], script, fallbacks={MINI: []}, default_fallbacks=[D])
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    assert _rows(raised.value.attempt_chain) == [(f"openai/{MINI}", True), (f"openai/{GPT}", True)]


@pytest.mark.fr("FR7", "FR11")
def test_a_raise_class_error_on_a_fallback_stops_before_the_next_entry(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)], F: [_status(400)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F, D])
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT, F]
    assert type(raised.value.last_error) is errors.BadRequestError
    assert [row.model for row in raised.value.attempt_chain] == [
        f"openai/{MINI}",
        f"openai/{GPT}",
        f"openai/{F}",
    ]


@pytest.mark.fr("FR22", "FR30")
def test_a_degraded_call_on_an_interim_medium_uses_mediums_own_list(call: Caller) -> None:
    run = Run(["nonsense"], {GPT: [_status(503)]}, fallbacks={GPT: [F]}, default_fallbacks=[D])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert [e.reason for e in run.of(events.RoutingDegraded)] == ["malformed_output"]
    assert run.models == [GPT, F] and run.of(events.AttemptSkipped) == []
    assert result.metadata is not None and result.metadata.tier is None
