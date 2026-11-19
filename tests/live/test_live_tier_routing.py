"""Live: Tier Routing with a real Analyzer, and the Interim Binding when the Analyzer fails."""

from typing import Any

import pytest

from primagate import (
    TIER_ROUTING_ALIAS,
    LowConfidenceDowngrade,
    PremiumRoutingSelected,
    PrimaGate,
    RoutingDecision,
    RoutingDegraded,
    SpendingCeilingExceeded,
)
from primagate.core import prices
from primagate.core.analyzer import ANALYZER_PROMPT_VERSION

from .support import ASK, Live, LiveCaller, LiveOutage

pytestmark = pytest.mark.live

# Model names no provider serves, so the Analyzer's question is refused without spending tokens.
MISSING_ANALYZER = {
    "openai": "openai/primagate-missing-analyzer",
    "anthropic": "claude-primagate-missing-analyzer",
}
POLICY: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.5,
    "context_token_budget": 1000,
    # SMALL, so a degraded call is answered by the cheap model.
    "interim_binding": "SMALL",
    "ambiguous_context_default": "SMALL",
}


def _models(live: Live) -> dict[str, str]:
    return {"SMALL": live.model, "MEDIUM": live.medium_model}


def _routed(live: Live, analyzer: str | None = None) -> PrimaGate:
    bindings = {tier: {"model": model} for tier, model in _models(live).items()}
    return live.gate(tier_bindings=bindings, policy=POLICY, analyzer=analyzer or live.model)


@pytest.mark.fr("FR20", "FR21", "FR36")
def test_a_real_analyzer_routes_the_call_and_the_decision_is_logged(
    call: LiveCaller, live: Live
) -> None:
    result = call(_routed(live), model=TIER_ROUTING_ALIAS, messages=ASK)
    assert result.choices[0].message.content
    meta = result.metadata
    assert meta is not None and meta.tier in POLICY["enabled_tiers"]

    (decision,) = live.events.of(RoutingDecision)
    assert decision.binding_id == f"{meta.tier}:{live.provider}/{_models(live)[meta.tier]}"
    assert decision.analyzer_prompt_version == ANALYZER_PROMPT_VERSION
    degraded = live.events.of(RoutingDegraded)
    if degraded:
        (event,) = degraded
        if event.reason == "provider_error":
            # Routing absorbed the Analyzer's failure, so the provider is what failed.
            raise LiveOutage("the Analyzer's provider could not answer")
        # Any other reason (an answer that is not exactly "<TIER> <confidence>", or a Tier that
        # is not enabled) is never read for a label: the Interim Binding serves the call.
        assert meta.tier == POLICY["interim_binding"]
    elif live.events.of(LowConfidenceDowngrade):
        assert meta.tier == POLICY["ambiguous_context_default"]
    premium = live.events.of(PremiumRoutingSelected)
    assert bool(premium) == (meta.tier == "MEDIUM")

    # The decision is made and logged before the one worker dispatch.
    types = live.events.types
    assert types.index("routing_decision") < types.index("dispatch_started")
    assert types.count("dispatch_started") == 1
    if not degraded:
        line = meta.analyzer
        assert line is not None and line.provider == live.provider
        assert line.token_counts is not None and line.token_counts.total_tokens > 0


@pytest.mark.fr("FR22", "FR36")
def test_an_analyzer_that_cannot_answer_degrades_to_the_interim_binding(
    call: LiveCaller, live: Live
) -> None:
    gate = _routed(live, analyzer=MISSING_ANALYZER[live.provider])
    result = call(gate, model=TIER_ROUTING_ALIAS, messages=ASK)
    assert result.choices[0].message.content
    assert result.metadata is not None and result.metadata.tier == POLICY["interim_binding"]
    (event,) = live.events.of(RoutingDegraded)
    assert event.reason == "provider_error"
    assert event.binding_id == f"SMALL:{live.provider}/{live.model}"
    assert live.events.of(PremiumRoutingSelected) == []
    types = live.events.types
    assert types.index("routing_degraded") < types.index("dispatch_started")
    assert types.count("dispatch_started") == 1


@pytest.mark.fr("FR35")
def test_a_real_call_that_crosses_the_ceiling_is_reported_and_still_answered(
    call: LiveCaller, live: Live
) -> None:
    # One Tier needs no Analyzer, so this is one request.
    bundled = prices.lookup(prices.BUNDLED, live.provider, live.model)
    overrides = None if bundled else {live.model: {"input": 1.0, "output": 5.0}}
    # Below what any real call costs, so the first answer crosses it.
    policy = {**POLICY, "enabled_tiers": ["SMALL"], "spending_ceiling": 1e-12}
    gate = live.gate(
        tier_bindings={"SMALL": {"model": live.model}}, policy=policy, price_overrides=overrides
    )
    result = call(gate, model=TIER_ROUTING_ALIAS, messages=ASK)
    assert result.choices[0].message.content
    assert result.metadata is not None and result.metadata.cost is not None
    (event,) = live.events.of(SpendingCeilingExceeded)
    assert event.ceiling == 1e-12 and event.total == pytest.approx(result.metadata.cost.total)
    assert live.events.types[-2:] == ["call_succeeded", "spending_ceiling_exceeded"]
