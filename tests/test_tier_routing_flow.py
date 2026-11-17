import pytest

from primagate.core.analyzer import decision_events, resolve_tier
from primagate.schema import events
from primagate.schema.analyzer import Analysis, Verdict
from primagate.schema.tiers import Tier

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
