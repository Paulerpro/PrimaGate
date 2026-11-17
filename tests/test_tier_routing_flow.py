import pytest

from primagate.core import steps
from primagate.core.analyzer import decision_events, resolve_tier
from primagate.core.call import advance, start
from primagate.core.policy import build_policy
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.core.steps import Action
from primagate.drivers.providers import parse_tier_bindings
from primagate.schema import events
from primagate.schema.analyzer import Analysis, Verdict
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.profile import (
    PROFILE_SCHEMA_VERSION,
    PrepState,
    Profile,
    ProfileContent,
    ProfileSources,
)
from primagate.schema.tiers import Tier
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Choice, Message
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


# --- the Profile a call pins when it starts ---

FP = "fp-ready"
PREPARED = ProfileContent(
    purpose="Answers questions about orders.",
    interactions=("order status questions",),
    unknowns=("refund policy",),
)


def _profile(source_ref: str = "prep-1;gpt-4o-mini") -> Profile:
    return Profile(
        "prep-abc", FP, PROFILE_SCHEMA_VERSION, "2026-11-17T10:00:00+00:00", PREPARED, source_ref
    )


def _with_sources() -> TierRouting:
    plain = _routing("gpt-4o-mini")
    return TierRouting(plain.policy, plain.bindings, plain.analyzer, ProfileSources("x"), FP)


def _verdict(text: str) -> steps.DispatchOutcome:
    return steps.DispatchOutcome(
        ChatCompletion(
            id="a",
            created=1,
            model="gpt-4o-mini",
            choices=[Choice(index=0, message=Message(role="assistant", content=text))],
        )
    )


def _pinned(
    outcome: steps.StartOutcome,
    analyzer: steps.DispatchOutcome | steps.Declined | None = None,
) -> list[object]:
    """Run a routed call whose preparation answer is `outcome`; the Analyzer says SMALL unless
    `analyzer` says what it answers instead."""
    seen: list[object] = []
    action: Action = start(
        ChatCompletionRequest(
            model=TIER_ROUTING_ALIAS, messages=[Message(role="user", content="hi")]
        ),
        CAPS,
        tier_routing=_with_sources(),
        cache_salt=b"s",
    )
    while True:
        seen.append(action)
        if isinstance(action, steps.Dispatch | steps.Raise | steps.Return):
            return seen
        if isinstance(action, steps.NewId):
            answer: steps.Outcome = steps.IdOutcome("req-1")
        elif isinstance(action, steps.CacheRead):
            answer = steps.CacheReadOutcome(None)
        elif isinstance(action, steps.Emit):
            answer = steps.EmitOutcome()
        elif isinstance(action, steps.StartPreparation):
            answer = outcome
        elif isinstance(action, steps.Analyze):
            answer = _verdict("SMALL 0.9") if analyzer is None else analyzer
        else:
            raise AssertionError(f"unexpected {type(action).__name__}")
        action = advance(action, answer)


def _emitted(seen: list[object]) -> list[events.Event]:
    return [e for a in seen if isinstance(a, steps.Emit) for e in a.events]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (steps.StartOutcome(True, PrepState.PREPARING, None), "preparing"),
        (steps.StartOutcome(False, PrepState.PREPARING, None), "preparing"),
        (steps.StartOutcome(False, PrepState.FAILED, None), "preparation_failed"),
    ],
)
def test_without_a_ready_profile_the_call_goes_to_the_interim_binding_unasked(
    outcome: steps.StartOutcome, reason: str
) -> None:
    seen = _pinned(outcome)
    assert not any(isinstance(a, steps.Analyze) for a in seen)
    last = seen[-1]
    assert isinstance(last, steps.Dispatch) and last.no_cache and last.request.model == "gpt-4o"
    degraded = [e for e in _emitted(seen) if isinstance(e, events.RoutingDegraded)]
    assert [e.reason for e in degraded] == [reason]
    assert all(e.profile_id == "unknown" for e in degraded)


def test_a_ready_profile_is_handed_to_the_analyzer_and_named_in_the_decision() -> None:
    seen = _pinned(steps.StartOutcome(False, PrepState.READY, _profile()))
    (asked,) = [a for a in seen if isinstance(a, steps.Analyze)]
    system = asked.dispatch.request.messages[0].content
    assert isinstance(system, str)
    assert "Purpose: Answers questions about orders." in system
    assert "Not known about the app:\n- refund policy" in system
    last = seen[-1]
    assert isinstance(last, steps.Dispatch) and not last.no_cache
    assert last.request.model == "gpt-4o-mini"
    (decision,) = [e for e in _emitted(seen) if isinstance(e, events.RoutingDecision)]
    assert (decision.profile_id, decision.input_fingerprint, decision.prep_prompt_version) == (
        "prep-abc",
        FP,
        "prep-1",
    )
    assert not any(isinstance(e, events.RoutingDegraded) for e in _emitted(seen))


@pytest.mark.parametrize(
    ("source_ref", "expected"),
    [
        ("prep-7;gpt-4o-mini", "prep-7"),
        ("earlier", PREP_PROMPT_VERSION),
        (";gpt-4o-mini", PREP_PROMPT_VERSION),
        ("", PREP_PROMPT_VERSION),
    ],
)
def test_the_prep_prompt_version_is_read_from_the_source_ref(
    source_ref: str, expected: str
) -> None:
    (decision,) = decision_events("r", S, "SMALL:openai/m", [S], None, profile=_profile(source_ref))
    assert isinstance(decision, events.RoutingDecision)
    assert decision.profile_id == "prep-abc" and decision.input_fingerprint == FP
    assert decision.prep_prompt_version == expected


@pytest.mark.parametrize(
    ("analyzer", "reason"),
    [
        (_verdict("probably SMALL"), "malformed_output"),
        (steps.Declined("over_capacity"), "over_capacity"),
    ],
)
def test_a_ready_call_that_degrades_still_names_its_pinned_profile(
    analyzer: steps.DispatchOutcome | steps.Declined, reason: str
) -> None:
    seen = _pinned(steps.StartOutcome(False, PrepState.READY, _profile()), analyzer)
    last = seen[-1]
    assert isinstance(last, steps.Dispatch) and last.no_cache and last.request.model == "gpt-4o"
    emitted = _emitted(seen)
    (decision,) = [e for e in emitted if isinstance(e, events.RoutingDecision)]
    (degraded,) = [e for e in emitted if isinstance(e, events.RoutingDegraded)]
    assert degraded.reason == reason
    for event in (decision, degraded):
        assert (event.profile_id, event.input_fingerprint, event.prep_prompt_version) == (
            "prep-abc",
            FP,
            "prep-1",
        )
