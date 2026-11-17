import pytest

from primagate.core import steps
from primagate.core.analyzer import decision_events, resolve_tier
from primagate.core.call import advance, start
from primagate.core.policy import build_policy
from primagate.core.steps import Action
from primagate.drivers.providers import parse_tier_bindings
from primagate.schema import events
from primagate.schema.analyzer import Analysis, Verdict
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.tiers import Tier
from primagate.schema.types import ChatCompletionRequest, Message
from primagate.schema.wire import Capabilities

S, M, P = Tier.SMALL, Tier.MEDIUM, Tier.POWERFUL


@pytest.mark.parametrize(
    ("chosen", "eligible", "expected"),
    [
        (M, [S, M, P], M),
        (S, [M, P], M),
        (S, [P], P),
        (P, [S, M], M),
        (M, [S, P], P),
        (P, [S], S),
    ],
)
def test_a_tier_that_cannot_serve_moves_to_the_nearest_able_one_stronger_first(
    chosen: Tier, eligible: list[Tier], expected: Tier
) -> None:
    assert resolve_tier(chosen, eligible) is expected


def _analysis(tier: Tier, confidence: float, downgraded: bool) -> Analysis:
    reason = "ambiguous_context_default" if downgraded else "accepted"
    return Analysis(Verdict(tier, confidence, reason, downgraded), None)


def test_every_decision_has_the_five_trace_fields_and_nothing_else() -> None:
    (only,) = decision_events("req-1", S, "SMALL:openai/gpt-4o-mini", [S], None)
    assert isinstance(only, events.RoutingDecision)
    assert only.to_dict() == {
        "event_type": "routing_decision",
        "request_id": "req-1",
        "profile_id": "unknown",
        "input_fingerprint": "unknown",
        "prep_prompt_version": "unknown",
        "analyzer_prompt_version": "unknown",
        "binding_id": "SMALL:openai/gpt-4o-mini",
        # Stamped by the driver on delivery, since core reads no clock.
        "timestamp": "",
    }


def test_the_extra_events_follow_the_decision_in_a_fixed_order() -> None:
    made = decision_events("r", M, "MEDIUM:openai/gpt-4o", [S, M], _analysis(S, 0.1, True))
    assert [type(e) for e in made] == [
        events.RoutingDecision,
        events.LowConfidenceDowngrade,
        events.PremiumRoutingSelected,
    ]
    assert all(e.analyzer_prompt_version == "analyzer-1" for e in made)  # type: ignore[attr-defined]


def test_a_confident_cheapest_choice_is_only_a_decision() -> None:
    made = decision_events("r", S, "SMALL:openai/m", [S, M], _analysis(S, 0.9, False))
    assert [type(e) for e in made] == [events.RoutingDecision]


def _drive(tiers: TierRouting, caps: dict[str, Capabilities]) -> list[object]:
    """Run `start` by hand, answering each ask, and return what core emitted and dispatched."""
    seen: list[object] = []
    action: Action = start(
        ChatCompletionRequest(
            model=TIER_ROUTING_ALIAS, messages=[Message(role="user", content="hi")]
        ),
        caps,
        tier_routing=tiers,
        cache_salt=b"s",
    )
    while True:
        seen.append(action)
        if isinstance(action, steps.Dispatch | steps.Raise | steps.Return):
            return seen
        if isinstance(action, steps.NewId):
            outcome: steps.Outcome = steps.IdOutcome("req-1")
        elif isinstance(action, steps.CacheRead):
            outcome = steps.CacheReadOutcome(None)
        elif isinstance(action, steps.Emit):
            outcome = steps.EmitOutcome()
        else:
            raise AssertionError(f"unexpected {type(action).__name__}")
        action = advance(action, outcome)


def _routing(analyzer: str | None) -> TierRouting:
    bindings = parse_tier_bindings(
        {"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}}
    )
    policy, _ = build_policy(
        {
            "enabled_tiers": ["SMALL", "MEDIUM"],
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            "interim_binding": "MEDIUM",
        },
        bindings,
    )
    return TierRouting(policy, bindings, analyzer)


CAPS = {"openai": Capabilities(), "anthropic": Capabilities()}


@pytest.mark.parametrize(
    ("analyzer", "caps", "reason"),
    [
        (None, CAPS, "no_analyzer"),
        # The Tiers' own provider is registered but the Analyzer's is not.
        ("claude-sonnet-5-5", {"openai": Capabilities()}, "analyzer_unavailable"),
    ],
)
def test_causes_setup_normally_prevents_still_degrade_instead_of_raising(
    analyzer: str | None, caps: dict[str, Capabilities], reason: str
) -> None:
    seen = _drive(_routing(analyzer), caps)
    degraded = [
        e
        for a in seen
        if isinstance(a, steps.Emit)
        for e in a.events
        if isinstance(e, events.RoutingDegraded)
    ]
    assert [e.reason for e in degraded] == [reason]
    last = seen[-1]
    assert isinstance(last, steps.Dispatch) and last.no_cache and last.request.model == "gpt-4o"
    assert not any(isinstance(a, steps.Analyze) for a in seen)
