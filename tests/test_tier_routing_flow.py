from typing import Any

import pytest

from primagate.core import steps
from primagate.core.analyzer import decision_events, resolve_tier
from primagate.core.call import advance, start
from primagate.core.policy import build_policy
from primagate.core.prices import estimate
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.core.steps import Action
from primagate.drivers.providers import parse_fallbacks, parse_tier_bindings
from primagate.schema import errors, events
from primagate.schema.analyzer import Analysis, Verdict
from primagate.schema.errors import PrimaGateError
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.profile import (
    PROFILE_SCHEMA_VERSION,
    PrepState,
    Profile,
    ProfileContent,
    ProfileSources,
)
from primagate.schema.tiers import Tier
from primagate.schema.types import (
    Attempt,
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    Message,
    Usage,
)
from primagate.schema.wire import NO_FALLBACKS, Capabilities, Fallbacks

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


def _verdict(text: str, usage: Usage | None = None) -> steps.DispatchOutcome:
    return steps.DispatchOutcome(
        ChatCompletion(
            id="a",
            created=1,
            model="gpt-4o-mini",
            choices=[Choice(index=0, message=Message(role="assistant", content=text))],
            usage=usage,
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


# --- MEDIUM-only Tier Failover ---

THREE = {
    "SMALL": {"model": "gpt-4o-mini"},
    "MEDIUM": {"model": "gpt-4o"},
    "POWERFUL": {"model": "gpt-4.1"},
}


def _three(bindings: dict[str, Any] | None = None, enabled: list[str] | None = None) -> TierRouting:
    parsed = parse_tier_bindings(bindings or THREE)
    names = enabled or ["SMALL", "MEDIUM", "POWERFUL"]
    policy, _ = build_policy(
        {
            "enabled_tiers": names,
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            "interim_binding": names[0],
            "ambiguous_context_default": names[0],
            "spending_ceiling": 2.0,
        },
        parsed,
    )
    return TierRouting(policy, parsed, "gpt-4o-mini")


def _failed_over(
    tiers: TierRouting,
    verdict: str,
    error: PrimaGateError | None = None,
    fallbacks: Fallbacks = NO_FALLBACKS,
    analyzer_usage: Usage | None = None,
) -> tuple[steps.Dispatch, list[object]]:
    """Route with the Analyzer saying `verdict`, fail the chosen dispatch, and return it with
    every action core took after the failure up to the next dispatch or the end."""
    action: Action = start(
        ChatCompletionRequest(
            model=TIER_ROUTING_ALIAS, messages=[Message(role="user", content="hi")]
        ),
        CAPS,
        fallbacks=fallbacks,
        tier_routing=tiers,
        cache_salt=b"s",
    )
    asked = 0

    def answer(action: Action) -> steps.Outcome:
        nonlocal asked
        if isinstance(action, steps.NewId):
            return steps.IdOutcome("req-1")
        if isinstance(action, steps.CacheRead):
            return steps.CacheReadOutcome(None)
        if isinstance(action, steps.Emit):
            return steps.EmitOutcome()
        if isinstance(action, steps.Analyze):
            asked += 1
            return _verdict(verdict, analyzer_usage)
        raise AssertionError(f"unexpected {type(action).__name__}")

    while not isinstance(action, steps.Dispatch):
        action = advance(action, answer(action))
    first = action
    failure = error or errors.ServiceUnavailableError("down", status_code=503)
    after: list[object] = []
    following = advance(first, steps.DispatchOutcome(error=failure))
    while True:
        after.append(following)
        if isinstance(following, steps.Dispatch | steps.Raise):
            # The Analyzer is asked once per call, never again after a failure.
            assert asked == 1
            return first, after
        following = advance(following, answer(following))


@pytest.mark.parametrize("verdict", ["SMALL 0.9", "POWERFUL 0.9"])
def test_a_failed_tier_fails_over_to_medium_keeping_what_the_call_carries(verdict: str) -> None:
    first, after = _failed_over(_three(), verdict)
    medium = after[-1]
    assert isinstance(medium, steps.Dispatch) and medium.request.model == "gpt-4o"
    assert medium.tier is M and first.tier is not M
    for name in (
        "request_id",
        "payload_retention",
        "prices",
        "cache_key",
        "no_cache",
        "analyzer",
        "spending_ceiling",
    ):
        assert getattr(medium, name) == getattr(first, name), name
    assert first.analyzer is not None and first.cache_key
    assert [a.model for a in medium.past] == [f"openai/{first.request.model}"]
    assert medium.plan is None and medium.retries == 0 and medium.history == ()
    assert not any(isinstance(a, steps.Analyze) for a in after)


def test_a_medium_choice_carries_no_failover() -> None:
    first, after = _failed_over(_three(), "MEDIUM 0.9")
    assert first.plan is None
    last = after[-1]
    assert isinstance(last, steps.Raise) and type(last.error) is errors.ServiceUnavailableError
    assert len(last.error.attempt_chain) == 1


def test_the_failover_is_the_filters_own_dispatch_for_medium() -> None:
    first, _ = _failed_over(_three(), "SMALL 0.9")
    assert first.plan is not None and first.plan.later == ()
    medium = first.plan.failover
    assert isinstance(medium, steps.Dispatch) and medium.request.model == "gpt-4o"


@pytest.mark.parametrize(
    ("bindings", "enabled", "reason"),
    [
        (THREE, ["SMALL", "POWERFUL"], "MEDIUM is not in enabled_tiers"),
        ({**THREE, "MEDIUM": {"model": "gpt-4o-mini"}}, None, "same model as the resolved Tier"),
    ],
)
def test_a_medium_that_cannot_run_is_a_skipped_row(
    bindings: dict[str, Any], enabled: list[str] | None, reason: str
) -> None:
    _, after = _failed_over(_three(bindings, enabled), "SMALL 0.9")
    last = after[-1]
    assert isinstance(last, steps.Raise)
    assert isinstance(last.error, errors.AttemptsExhaustedError)
    tried, passed = last.error.attempt_chain
    assert tried.skipped is None and passed.skipped == reason and passed.dispatches == []
    skips = [
        e
        for a in after
        if isinstance(a, steps.Emit)
        for e in a.events
        if isinstance(e, events.AttemptSkipped)
    ]
    assert [(e.model, e.reason) for e in skips] == [(passed.model, reason)]


def test_a_raise_class_error_on_the_resolved_tier_ends_the_call() -> None:
    _, after = _failed_over(_three(), "SMALL 0.9", errors.BadRequestError("bad"))
    last = after[-1]
    assert isinstance(last, steps.Raise) and type(last.error) is errors.BadRequestError
    assert len(last.error.attempt_chain) == 1


# --- the Fallback List after Tier Failover ---


def _later(
    verdict: str,
    fallbacks: Fallbacks,
    tiers: TierRouting | None = None,
) -> steps.Dispatch:
    first, _ = _failed_over(tiers or _three(), verdict, fallbacks=fallbacks)
    return first


def test_the_resolved_models_own_list_replaces_the_default_on_a_routed_call() -> None:
    fallbacks = parse_fallbacks({"gpt-4o-mini": ["gpt-4.1-mini"]}, ["o3"])
    first = _later("SMALL 0.9", fallbacks)
    assert first.plan is not None and first.plan.later == (("openai", "gpt-4.1-mini"),)
    other = _later("POWERFUL 0.9", fallbacks)
    assert other.plan is not None and other.plan.later == (("openai", "o3"),)


def test_a_medium_choice_carries_the_fallback_list_and_no_failover() -> None:
    first = _later("MEDIUM 0.9", parse_fallbacks(None, ["o3"]))
    assert first.plan is not None and first.plan.failover is None
    assert first.plan.later == (("openai", "o3"),)


def test_the_resolved_model_and_the_medium_failover_are_dropped_from_the_list() -> None:
    fallbacks = parse_fallbacks(None, ["gpt-4o-mini", "gpt-4o", "o3", "gpt-4o"])
    first = _later("SMALL 0.9", fallbacks)
    assert first.plan is not None and isinstance(first.plan.failover, steps.Dispatch)
    assert first.plan.later == (("openai", "o3"),)


def test_a_medium_that_was_only_skipped_stays_in_the_list() -> None:
    fallbacks = parse_fallbacks(None, ["gpt-4o", "o3"])
    first = _later("SMALL 0.9", fallbacks, _three(enabled=["SMALL", "POWERFUL"]))
    assert first.plan is not None and isinstance(first.plan.failover, Attempt)
    assert first.plan.later == (("openai", "gpt-4o"), ("openai", "o3"))


def test_the_fallback_list_comes_after_the_failover_and_carries_the_call() -> None:
    fallbacks = parse_fallbacks(None, ["o3"])
    first, after = _failed_over(_three(), "SMALL 0.9", fallbacks=fallbacks)
    medium = after[-1]
    assert isinstance(medium, steps.Dispatch) and medium.tier is M
    assert medium.plan is not None and medium.plan.failover is None
    following = advance(
        medium, steps.DispatchOutcome(error=errors.ServiceUnavailableError("x", status_code=503))
    )
    while isinstance(following, steps.Emit):
        following = advance(following, steps.EmitOutcome())
    assert isinstance(following, steps.Dispatch) and following.request.model == "o3"
    assert following.tier is None and following.plan is None
    for name in (
        "request_id",
        "payload_retention",
        "prices",
        "cache_key",
        "no_cache",
        "analyzer",
        "spending_ceiling",
    ):
        assert getattr(following, name) == getattr(first, name), name
    assert [a.model for a in following.past] == ["openai/gpt-4o-mini", "openai/gpt-4o"]


ANALYZER_USAGE = Usage(100, 4, 104)


def _routed_fallback() -> steps.Dispatch:
    """A routed call whose Analyzer cost something, failed over to its one fallback."""
    fallbacks = parse_fallbacks(None, ["gpt-4.1-mini"])
    _, after = _failed_over(
        _three(enabled=["SMALL", "POWERFUL"]),
        "SMALL 0.9",
        fallbacks=fallbacks,
        analyzer_usage=ANALYZER_USAGE,
    )
    fallback = after[-1]
    assert isinstance(fallback, steps.Dispatch) and fallback.tier is None
    assert fallback.analyzer is not None and fallback.analyzer.cost is not None
    return fallback


def test_a_fallback_answer_on_a_routed_call_is_counted_against_the_ceiling() -> None:
    fallback = _routed_fallback()
    usage = Usage(10, 5, 15)
    answered = ChatCompletion(
        id="w",
        created=1,
        model="gpt-4.1-mini",
        choices=[Choice(index=0, message=Message(role="assistant", content="ok"))],
        usage=usage,
    )
    following = advance(fallback, steps.DispatchOutcome(answered))
    while isinstance(following, steps.Emit | steps.CacheWrite):
        following = advance(
            following,
            steps.EmitOutcome() if isinstance(following, steps.Emit) else steps.CacheWriteOutcome(),
        )
    assert isinstance(following, steps.RecordSpend)
    own = estimate("openai", "gpt-4.1-mini", usage, fallback.prices).cost
    assert own is not None and fallback.analyzer is not None and fallback.analyzer.cost
    assert following.ceiling == 2.0
    assert following.amount == pytest.approx(own.total + fallback.analyzer.cost.total)


def test_a_routed_call_cancelled_on_its_fallback_still_counts_the_analyzer_cost() -> None:
    fallback = _routed_fallback()
    following = advance(fallback, steps.Cancelled())
    assert isinstance(following, steps.RecordSpend)
    assert fallback.analyzer is not None and fallback.analyzer.cost is not None
    assert following.amount == pytest.approx(fallback.analyzer.cost.total)
    assert following.ceiling == 2.0


def test_only_a_routed_workers_dispatch_carries_a_spending_ceiling() -> None:
    request = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="hi")])
    plain: Action = start(request, CAPS, fallbacks=parse_fallbacks(None, ["o3"]), cache_salt=b"s")
    while not isinstance(plain, steps.Dispatch):
        assert isinstance(plain, steps.NewId | steps.CacheRead | steps.Emit)
        plain = advance(
            plain,
            steps.IdOutcome("req-1")
            if isinstance(plain, steps.NewId)
            else steps.CacheReadOutcome(None)
            if isinstance(plain, steps.CacheRead)
            else steps.EmitOutcome(),
        )
    assert plain.spending_ceiling is None
    action: Action = start(
        ChatCompletionRequest(
            model=TIER_ROUTING_ALIAS, messages=[Message(role="user", content="hi")]
        ),
        CAPS,
        tier_routing=_three(),
        cache_salt=b"s",
    )
    while not isinstance(action, steps.Analyze):
        assert isinstance(action, steps.NewId | steps.CacheRead | steps.Emit)
        action = advance(
            action,
            steps.IdOutcome("req-1")
            if isinstance(action, steps.NewId)
            else steps.CacheReadOutcome(None)
            if isinstance(action, steps.CacheRead)
            else steps.EmitOutcome(),
        )
    assert action.dispatch.spending_ceiling is None
