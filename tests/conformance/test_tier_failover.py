import json
from typing import Any

import httpx2
import pytest
from conftest import Caller
from test_retry_routing import B_KEY, B_URL, Reply, _answer, _status
from test_tier_routing import IMAGE_MSGS, MSGS
from test_tier_routing import Run as TierRun

from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS

ALIAS = TIER_ROUTING_ALIAS
MINI, GPT, FOUR = "gpt-4o-mini", "gpt-4o", "gpt-4.1"
THREE = {"SMALL": {"model": MINI}, "MEDIUM": {"model": GPT}, "POWERFUL": {"model": FOUR}}
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
POLICY: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
    "ambiguous_context_default": "SMALL",
}


class Run(TierRun):
    """A routed gate whose worker calls play a script per model, the last reply repeating."""

    def __init__(
        self,
        verdicts: list[Any],
        workers: dict[str, list[Reply]],
        policy: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.script = workers
        self.played: dict[str, int] = {}
        self.hosts: list[str] = []
        super().__init__(verdicts, policy=policy or POLICY, bindings=bindings or THREE, **kwargs)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        system = str(body.get("messages", [{}])[0].get("content", ""))
        if "Prompt version: analyzer" in system:
            return super()._handle(request)
        self.workers.append(body)
        self.hosts.append(request.url.host)
        model = body["model"]
        replies = self.script.get(model, [_answer])
        index = min(self.played.get(model, 0), len(replies) - 1)
        self.played[model] = self.played.get(model, 0) + 1
        return replies[index]("openai", bool(body.get("stream")))

    @property
    def models(self) -> list[str]:
        return [w["model"] for w in self.workers]


def _skips(run: Run) -> list[events.AttemptSkipped]:
    return run.of(events.AttemptSkipped)


@pytest.mark.usefixtures("delays")
@pytest.mark.fr("FR30")
def test_a_failed_small_tier_is_answered_by_medium_after_its_own_routing(call: Caller) -> None:
    routing = {MINI: [{"base_url": B_URL, "api_key": B_KEY}]}
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, provider_routing=routing)
    result = call(run.gate, model=ALIAS, messages=MSGS)
    # Three tries on the primary, three on the redundant endpoint, then one MEDIUM Attempt.
    assert run.models == [MINI] * 6 + [GPT]
    assert len(run.analyzed) == 1
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    first, second = result.metadata.attempt_chain
    assert first.model == f"openai/{MINI}" and second.model == f"openai/{GPT}"
    assert [d.endpoint for d in first.dispatches] == ["openai", B_URL]
    assert second.dispatches[0].error is None
    assert _skips(run) == []


@pytest.mark.fr("FR30")
def test_a_powerful_tier_that_fails_with_an_advance_error_moves_to_medium(call: Caller) -> None:
    run = Run(["POWERFUL 0.9"], {FOUR: [_status(401)]})
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [FOUR, GPT] and len(run.analyzed) == 1
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert [a.model for a in result.metadata.attempt_chain] == [f"openai/{FOUR}", f"openai/{GPT}"]


@pytest.mark.fr("FR30")
def test_a_failed_medium_tier_has_no_failover(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"], {GPT: [_status(503)]})
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [GPT] and len(raised.value.attempt_chain) == 1
    assert _skips(run) == []


@pytest.mark.fr("FR30")
def test_medium_not_enabled_leaves_a_skipped_row(call: Caller) -> None:
    policy = {
        **POLICY,
        "enabled_tiers": ["SMALL", "POWERFUL"],
        "interim_binding": "SMALL",
    }
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, policy=policy)
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI]
    assert isinstance(raised.value.last_error, errors.ServiceUnavailableError)
    tried, passed = raised.value.attempt_chain
    assert tried.skipped is None and passed.dispatches == []
    assert passed.model == f"openai/{GPT}" and passed.skipped is not None
    assert "enabled_tiers" in passed.skipped
    (event,) = _skips(run)
    assert (event.model, event.reason) == (passed.model, passed.skipped)


@pytest.mark.fr("FR25", "FR30")
def test_a_medium_tier_the_filter_removed_leaves_its_filter_reason(call: Caller) -> None:
    bindings = {**THREE, "MEDIUM": {"model": "claude-sonnet-5-5"}}
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, bindings=bindings)
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=IMAGE_MSGS, drop_params=False)
    assert run.models == [MINI]
    _, passed = raised.value.attempt_chain
    assert passed.model == "anthropic/claude-sonnet-5-5" and passed.skipped is not None
    assert "anthropic" in passed.skipped and "image" in passed.skipped
    assert [e.reason for e in _skips(run)] == [passed.skipped]


@pytest.mark.fr("FR30")
def test_a_medium_tier_bound_to_the_failed_model_is_not_run_again(call: Caller) -> None:
    bindings = {**THREE, "MEDIUM": {"model": MINI}}
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, bindings=bindings)
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI]
    _, passed = raised.value.attempt_chain
    assert passed.model == f"openai/{MINI}"
    assert passed.skipped == "same model as the resolved Tier"


@pytest.mark.fr("FR11", "FR30")
def test_when_both_tiers_fail_the_aggregate_error_holds_both_attempts(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)], GPT: [_status(429)]})
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT] and len(run.analyzed) == 1
    assert isinstance(raised.value.last_error, errors.RateLimitError)
    chain = raised.value.attempt_chain
    assert [a.model for a in chain] == [f"openai/{MINI}", f"openai/{GPT}"]
    assert all(a.skipped is None and a.dispatches[0].error for a in chain)
    (failed,) = run.of(events.CallFailed)
    assert failed.attempts == 2


@pytest.mark.fr("FR12", "FR30")
def test_the_medium_attempt_has_its_own_retries_and_redundant_endpoints(
    call: Caller, delays: list[float]
) -> None:
    routing = {GPT: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {MINI: [_status(503)], GPT: [_status(503)] * 3 + [_answer]}
    run = Run(["SMALL 0.9"], script, provider_routing=routing)
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI] + [GPT] * 4
    assert run.hosts == ["api.openai.com"] * 4 + ["b.test"]
    assert delays == [0.5, 1.0]
    assert result.metadata is not None
    _, medium = result.metadata.attempt_chain
    assert [d.endpoint for d in medium.dispatches] == ["openai", B_URL]
    assert [d.retries for d in medium.dispatches] == [2, 0]


@pytest.mark.fr("FR30")
def test_a_raise_error_on_the_resolved_tier_never_fails_over(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(400)]})
    with pytest.raises(errors.BadRequestError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI] and len(raised.value.attempt_chain) == 1
    assert _skips(run) == []


@pytest.mark.fr("FR30", "FR31")
def test_a_failover_from_the_interim_binding_is_never_cached(call: Caller) -> None:
    run = Run(["nonsense"], {MINI: [_status(503)]}, policy={**POLICY, "interim_binding": "SMALL"})
    first = call(run.gate, model=ALIAS, messages=MSGS)
    assert first.metadata is not None and first.metadata.tier == "MEDIUM"
    assert [e.reason for e in run.of(events.RoutingDegraded)] == ["malformed_output"]
    second = call(run.gate, model=ALIAS, messages=MSGS)
    assert second.metadata is not None and not second.metadata.cache_hit
    assert run.models == [MINI, GPT, MINI, GPT] and len(run.analyzed) == 2


@pytest.mark.fr("FR31")
def test_a_routed_failover_answer_is_cached_like_any_routed_answer(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]})
    call(run.gate, model=ALIAS, messages=MSGS)
    again = call(run.gate, model=ALIAS, messages=MSGS)
    assert again.metadata is not None and again.metadata.cache_hit
    assert run.models == [MINI, GPT] and len(run.analyzed) == 1


@pytest.mark.fr("FR35")
def test_the_spend_counts_the_medium_answer_and_the_one_analyzer_cost(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, policy={**POLICY, "spending_ceiling": 1.0})
    result = call(run.gate, model=ALIAS, messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.cost is not None
    assert meta.analyzer is not None and meta.analyzer.cost is not None
    assert len(run.analyzed) == 1
    assert run.gate._spend.total == pytest.approx(meta.cost.total + meta.analyzer.cost.total)


@pytest.mark.fr("FR30")
def test_a_plain_model_call_never_fails_over(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]})
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(run.gate, model=MINI, messages=MSGS)
    assert run.models == [MINI] and run.analyzed == []
    assert len(raised.value.attempt_chain) == 1


@pytest.mark.fr("FR30")
def test_the_failover_is_announced_as_the_next_attempt_without_a_new_decision(
    call: Caller,
) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]})
    call(run.gate, model=ALIAS, messages=MSGS)
    (failed,) = run.of(events.DispatchFailed)
    assert failed.next == "next_attempt" and failed.model == f"openai/{MINI}"
    assert len(run.decisions) == 1
    started = run.of(events.DispatchStarted)
    assert [s.model for s in started] == [f"openai/{MINI}", f"openai/{GPT}"]
    (done,) = run.of(events.CallSucceeded)
    assert done.model == f"openai/{GPT}" and done.attempts == 2


@pytest.mark.fr("FR30")
def test_a_medium_tier_that_is_not_bound_at_all_is_named_medium_in_its_row(call: Caller) -> None:
    policy = {**POLICY, "enabled_tiers": ["SMALL"], "interim_binding": "SMALL"}
    run = Run(
        ["SMALL 0.9"], {MINI: [_status(503)]}, policy=policy, bindings={"SMALL": {"model": MINI}}
    )
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI]
    assert isinstance(raised.value.last_error, errors.ServiceUnavailableError)
    passed = raised.value.attempt_chain[1]
    assert passed.model == "MEDIUM" and passed.skipped == "MEDIUM is not in enabled_tiers"
    (event,) = _skips(run)
    assert (event.model, event.reason) == ("MEDIUM", "MEDIUM is not in enabled_tiers")


@pytest.mark.fr("FR30")
def test_the_medium_attempt_drops_what_its_own_limits_cannot_take(call: Caller) -> None:
    bindings = {**THREE, "MEDIUM": {"model": GPT, "limits": {"supported_features": ["streaming"]}}}
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, bindings=bindings)
    result = call(run.gate, model=ALIAS, messages=MSGS, tools=[TOOL], drop_params=["tools"])
    assert run.models == [MINI, GPT]
    assert "tools" in run.workers[0] and "tools" not in run.workers[1]
    assert result.metadata is not None and result.metadata.dropped_params == ["tools"]
    assert result.metadata.tier == "MEDIUM"


def _too_long(provider: str, stream: bool) -> httpx2.Response:
    body = {
        "error": {
            "message": "too long",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    return httpx2.Response(400, json=body)


@pytest.mark.fr("FR30")
def test_a_context_window_error_on_the_resolved_tier_moves_to_medium(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_too_long]})
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    first = result.metadata.attempt_chain[0].dispatches[0]
    assert first.error is not None and first.error.type == "ContextWindowExceededError"


@pytest.mark.fr("FR30")
def test_a_raise_error_on_medium_ends_the_chain_holding_both_attempts(call: Caller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)], GPT: [_status(400)]})
    with pytest.raises(errors.AttemptsExhaustedError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    # Two models were tried, so the RAISE-class error arrives as the aggregate's last error.
    assert type(raised.value.last_error) is errors.BadRequestError
    chain = raised.value.attempt_chain
    assert [a.model for a in chain] == [f"openai/{MINI}", f"openai/{GPT}"]
    assert all(a.skipped is None for a in chain)


@pytest.mark.fr("FR30")
def test_an_interim_binding_of_medium_that_fails_has_no_failover(call: Caller) -> None:
    run = Run(["nonsense"], {GPT: [_status(503)]})
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(run.gate, model=ALIAS, messages=MSGS)
    assert [e.reason for e in run.of(events.RoutingDegraded)] == ["malformed_output"]
    assert run.models == [GPT] and len(raised.value.attempt_chain) == 1
    assert _skips(run) == []
