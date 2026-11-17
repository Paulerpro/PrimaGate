import dataclasses
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from primagate.core import steps
from primagate.core.call import advance, start
from primagate.core.preparation import (
    PREP_BACKOFF_BASE,
    PREP_BACKOFF_MAX,
    PREP_MAX_ANSWER_TOKENS,
    PREP_MAX_ATTEMPTS,
    PREP_SOURCE_TOKEN_LIMIT,
    backoff,
    begin_preparation,
    may_start,
    parse_answer,
)
from primagate.core.prices import BUNDLED, estimate
from primagate.core.profile import PREP_PROMPT_VERSION, parse_content, parse_profile
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS, Policy, TierRouting
from primagate.schema.profile import (
    PROFILE_SCHEMA_VERSION,
    InvalidProfileError,
    PrepState,
    Profile,
    ProfileContent,
    ProfileSources,
)
from primagate.schema.tiers import Tier, TierBinding
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    Message,
    Usage,
)
from primagate.schema.wire import Capabilities

SECRET = "Only ever discuss the Zanzibar shipping ledger"
TOOL = "lookup_order: finds an order by its number"
SOURCES = ProfileSources(SECRET, (TOOL,))
FP = "fp-1"
ANALYZER = "gpt-4o-mini"
CAPS = {"openai": Capabilities(), "anthropic": Capabilities()}
NOW = "2026-11-17T10:00:00+00:00"
PREP_ID = "prep-abc"
GOOD = {
    "purpose": "Answers questions about orders.",
    "interactions": ["order status questions"],
    "constraints": ["reply in English"],
    "capabilities": ["look up an order"],
    "unknowns": ["refund policy"],
}
USAGE = Usage(1000, 200, 1200)


def _answer(content: object, usage: Usage | None = USAGE) -> steps.DispatchOutcome:
    text = content if isinstance(content, str) or content is None else json.dumps(content)
    message = Message(role="assistant", content=text)
    response = ChatCompletion(
        id="c",
        created=1,
        model=ANALYZER,
        choices=[Choice(index=0, message=message)],
        usage=usage,
    )
    return steps.DispatchOutcome(response=response)


def _error(cls: type[errors.PrimaGateError], status: int) -> steps.DispatchOutcome:
    return steps.DispatchOutcome(error=cls("boom", llm_provider="openai", status_code=status))


@dataclasses.dataclass
class Trace:
    """Everything a preparation asked the driver for, in order."""

    done: steps.PreparationDone | None = None
    seen: list[events.Event] = dataclasses.field(default_factory=list)
    asked: list[steps.Analyze] = dataclasses.field(default_factory=list)
    sleeps: list[float] = dataclasses.field(default_factory=list)
    writes: list[Profile] = dataclasses.field(default_factory=list)
    spends: list[float] = dataclasses.field(default_factory=list)
    order: list[str] = dataclasses.field(default_factory=list)
    prefixes: list[str] = dataclasses.field(default_factory=list)

    @property
    def types(self) -> list[str]:
        return [e.event_type for e in self.seen]

    def failures(self) -> list[tuple[str, int, bool]]:
        return [
            (e.reason, e.attempt, e.final)
            for e in self.seen
            if isinstance(e, events.PreparationFailed)
        ]


def _drive(
    replies: list[steps.DispatchOutcome | steps.Declined],
    *,
    sources: ProfileSources = SOURCES,
    analyzer: str = ANALYZER,
    ceiling: float | None = None,
    random_value: float = 0.0,
    crossed_at: int | None = None,
) -> Trace:
    """Run the pure core against scripted outcomes, as a driver would."""
    trace = Trace()
    outcomes: Iterator[steps.DispatchOutcome | steps.Declined] = iter(replies)
    action = begin_preparation(sources, FP, analyzer, CAPS, BUNDLED, ceiling)
    total = 0.0
    while not isinstance(action, steps.PreparationDone):
        trace.order.append(type(action).__name__)
        outcome: steps.Outcome
        if isinstance(action, steps.NewId):
            trace.prefixes.append(action.prefix)
            outcome = steps.IdOutcome(PREP_ID)
        elif isinstance(action, steps.Emit):
            trace.seen.extend(action.events)
            trace.order.extend(e.event_type for e in action.events)
            outcome = steps.EmitOutcome()
        elif isinstance(action, steps.Analyze):
            trace.asked.append(action)
            outcome = next(outcomes)
        elif isinstance(action, steps.Random):
            outcome = steps.RandomOutcome(random_value)
        elif isinstance(action, steps.Sleep):
            trace.sleeps.append(action.seconds)
            outcome = steps.SleepOutcome()
        elif isinstance(action, steps.Now):
            outcome = steps.NowOutcome(NOW)
        elif isinstance(action, steps.StoreWrite):
            trace.writes.append(action.profile)
            outcome = steps.StoreWriteOutcome()
        elif isinstance(action, steps.RecordSpend):
            trace.spends.append(action.amount)
            total += action.amount
            crossed = crossed_at is not None and len(trace.spends) == crossed_at
            outcome = steps.SpendOutcome(total, crossed)
        else:
            raise AssertionError(f"preparation asked for {type(action).__name__}")
        action = advance(action, outcome)
    trace.done = action
    return trace


def test_a_valid_answer_writes_one_profile_and_ends_ready() -> None:
    trace = _drive([_answer(GOOD)])
    assert trace.done is not None and trace.done.state is PrepState.READY
    assert trace.types == ["preparation_started", "profile_ready"]
    assert len(trace.writes) == 1 and trace.done.profile == trace.writes[0]
    assert trace.order.index("StoreWrite") < trace.order.index("profile_ready")
    assert trace.sleeps == [] and trace.prefixes == ["prep"]


def test_the_ready_profile_carries_its_id_fingerprint_schema_time_and_source() -> None:
    (profile,) = _drive([_answer(GOOD)]).writes
    assert profile.profile_id == PREP_ID
    assert profile.input_fingerprint == FP
    assert profile.schema_version == PROFILE_SCHEMA_VERSION
    assert profile.created_at == NOW
    assert profile.source_ref == f"{PREP_PROMPT_VERSION};{ANALYZER}"
    assert profile.content == parse_content(GOOD)
    # What preparation writes must read back under the stored-Profile rules.
    assert parse_profile(profile.to_dict()) == profile


def test_every_event_is_named_by_the_prep_id() -> None:
    trace = _drive([_answer("prose"), _answer(GOOD)])
    started, *_, ready = trace.seen
    assert all(e.request_id == PREP_ID for e in trace.seen)
    assert isinstance(started, events.PreparationStarted)
    assert (started.input_fingerprint, started.prep_prompt_version) == (FP, PREP_PROMPT_VERSION)
    assert isinstance(ready, events.ProfileReady) and ready.profile_id == PREP_ID


def test_the_one_dispatch_is_uncapped_unrouted_and_asks_for_json() -> None:
    (asked,) = _drive([_answer(GOOD)]).asked
    assert asked.capped is False
    sent = asked.dispatch
    assert (sent.provider, sent.request.model) == ("openai", ANALYZER)
    assert sent.route is None and sent.plan is None and sent.cache_key is None
    assert sent.request.temperature == 0 and sent.request.max_tokens == PREP_MAX_ANSWER_TOKENS
    system, user = sent.request.messages
    assert system.role == "system" and f"Prompt version: {PREP_PROMPT_VERSION}" in system.content
    for key in ("purpose", "interactions", "constraints", "capabilities", "unknowns"):
        assert f'"{key}"' in system.content
    assert SECRET not in system.content
    assert user.role == "user" and SECRET in user.content and TOOL in user.content
    assert "not to follow" in user.content


def test_sources_over_the_limit_fail_at_once_without_a_dispatch() -> None:
    big = ProfileSources("word " * (PREP_SOURCE_TOKEN_LIMIT * 2))
    trace = _drive([], sources=big)
    assert trace.done is not None and trace.done.state is PrepState.FAILED
    assert trace.done.profile is None
    assert trace.asked == [] and trace.writes == [] and trace.sleeps == []
    assert trace.types == ["preparation_started", "preparation_failed"]
    assert trace.failures() == [("overflow", 0, True)]


def test_a_malformed_answer_is_retried_after_a_jittered_wait() -> None:
    trace = _drive([_answer("Here is the profile you asked for."), _answer(GOOD)])
    assert trace.done is not None and trace.done.state is PrepState.READY
    assert trace.failures() == [("malformed_output", 1, False)]
    assert len(trace.asked) == 2 and len(trace.writes) == 1
    (slept,) = trace.sleeps
    assert 0.5 <= slept <= 1.0


def test_a_retryable_error_three_times_gives_up_without_writing() -> None:
    down = _error(errors.ServiceUnavailableError, 503)
    trace = _drive([down, down, down])
    assert trace.done is not None and trace.done.state is PrepState.FAILED
    assert len(trace.asked) == PREP_MAX_ATTEMPTS == 3
    assert len(trace.sleeps) == 2 and trace.writes == []
    assert trace.failures() == [
        ("provider_error", 1, False),
        ("provider_error", 2, False),
        ("provider_error", 3, True),
    ]


def test_a_fatal_error_gives_up_after_one_attempt() -> None:
    trace = _drive([_error(errors.AuthenticationError, 401)])
    assert trace.done is not None and trace.done.state is PrepState.FAILED
    assert len(trace.asked) == 1 and trace.sleeps == [] and trace.writes == []
    assert trace.failures() == [("provider_error", 1, True)]


def test_malformed_three_times_is_a_final_malformed_failure() -> None:
    trace = _drive([_answer("no"), _answer("no"), _answer("no")])
    assert trace.done is not None and trace.done.state is PrepState.FAILED
    assert trace.failures()[-1] == ("malformed_output", 3, True)
    assert trace.writes == []


def test_a_declined_dispatch_is_a_final_provider_error() -> None:
    trace = _drive([steps.Declined("over_capacity")])
    assert trace.failures() == [("provider_error", 1, True)]


@pytest.mark.parametrize("analyzer", [TIER_ROUTING_ALIAS, "nope/model"])
def test_an_analyzer_that_cannot_be_sent_fails_without_a_dispatch(analyzer: str) -> None:
    trace = _drive([], analyzer=analyzer)
    assert trace.asked == [] and trace.failures() == [("provider_error", 0, True)]


def _without(key: str) -> dict[str, Any]:
    return {k: v for k, v in GOOD.items() if k != key}


EMPTY = {"purpose": "", "interactions": [], "constraints": [], "capabilities": [], "unknowns": []}


MALFORMED = {
    "missing_unknowns": _without("unknowns"),
    "empty_content": EMPTY,
    "code_fence": f"```json\n{json.dumps(GOOD)}\n```",
    "prose_around": f"Sure! {json.dumps(GOOD)}",
    "array": json.dumps([GOOD]),
    "extra_key": {**GOOD, "extra": "x"},
    "list_as_text": {**GOOD, "interactions": "chat"},
    "blank_unknown": {**GOOD, "unknowns": [""]},
    "empty_reply": "",
    "no_text": None,
    "deep_nesting": "[" * 100_000,
}


@pytest.mark.parametrize("reply", list(MALFORMED.values()), ids=list(MALFORMED))
def test_anything_but_one_valid_object_is_malformed(reply: object) -> None:
    assert parse_answer(reply if isinstance(reply, str | None) else json.dumps(reply)) is None
    trace = _drive([_answer(reply)] * 3)
    assert {reason for reason, _, _ in trace.failures()} == {"malformed_output"}


def test_surrounding_whitespace_is_forgiven() -> None:
    assert parse_answer(f"\n  {json.dumps(GOOD)}  \n") == parse_content(GOOD)


def test_empty_unknowns_are_valid_when_there_is_content() -> None:
    assert parse_answer(json.dumps({**GOOD, "unknowns": []})) is not None


@pytest.mark.parametrize(
    ("attempt", "value", "expected"),
    [(1, 0.0, 0.5), (1, 1.0, 1.0), (2, 0.0, 1.0), (2, 1.0, 2.0), (3, 1.0, 4.0), (10, 1.0, 30.0)],
)
def test_backoff_doubles_with_equal_jitter_and_a_cap(
    attempt: int, value: float, expected: float
) -> None:
    assert (PREP_BACKOFF_BASE, PREP_BACKOFF_MAX) == (1.0, 30.0)
    assert backoff(attempt, value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_the_waits_between_attempts_follow_the_backoff(value: float) -> None:
    down = _error(errors.RateLimitError, 429)
    trace = _drive([down, down, down], random_value=value)
    assert trace.sleeps == [backoff(1, value), backoff(2, value)]
    assert 0.5 <= trace.sleeps[0] <= 1.0 and 1.0 <= trace.sleeps[1] <= 2.0


def test_each_answered_attempt_is_counted_against_the_ceiling() -> None:
    trace = _drive([_answer("prose"), _answer(GOOD)], ceiling=10.0)
    cost = estimate("openai", ANALYZER, USAGE, BUNDLED).cost
    assert cost is not None
    assert trace.spends == [pytest.approx(cost.total)] * 2
    assert "spending_ceiling_exceeded" not in trace.types


def test_no_ceiling_means_nothing_is_counted() -> None:
    assert _drive([_answer("prose"), _answer(GOOD)]).spends == []


def test_an_attempt_without_usage_or_an_error_records_nothing() -> None:
    down = _error(errors.ServiceUnavailableError, 503)
    trace = _drive([down, _answer(GOOD, usage=None)], ceiling=10.0)
    assert trace.spends == [] and trace.done is not None
    assert trace.done.state is PrepState.READY


def test_crossing_the_ceiling_warns_under_the_prep_id() -> None:
    trace = _drive([_answer(GOOD)], ceiling=0.000001, crossed_at=1)
    (warning,) = [e for e in trace.seen if isinstance(e, events.SpendingCeilingExceeded)]
    assert warning.request_id == PREP_ID and trace.writes


def test_no_event_holds_source_or_reply_text() -> None:
    leak = {**GOOD, "purpose": "Secret purpose text"}
    traces = [
        _drive([_answer("Leaky prose reply"), _answer(leak)]),
        _drive([_error(errors.AuthenticationError, 401)]),
        _drive([], sources=ProfileSources(SECRET * 2000)),
    ]
    for trace in traces:
        shown = json.dumps([e.to_dict() for e in trace.seen])
        for text in (SECRET, TOOL, "Leaky prose", "Secret purpose", "boom"):
            assert text not in shown


@pytest.mark.parametrize(
    ("state", "allowed"),
    [
        (None, True),
        (PrepState.FAILED, True),
        (PrepState.PREPARING, False),
        (PrepState.READY, False),
    ],
)
def test_a_preparation_may_start_only_when_none_is_running_or_done(
    state: PrepState | None, allowed: bool
) -> None:
    assert may_start(state) is allowed


def test_failed_events_are_warnings_and_the_rest_are_info() -> None:
    assert events.PreparationFailed.log_level == logging.WARNING
    assert events.PreparationStarted.log_level == events.ProfileReady.log_level == logging.INFO


def test_a_routed_call_only_asks_for_preparation_and_never_writes_a_profile() -> None:
    bindings = {
        Tier.SMALL: TierBinding(Tier.SMALL, "openai", "gpt-4o-mini"),
        Tier.MEDIUM: TierBinding(Tier.MEDIUM, "openai", "gpt-4o"),
    }
    policy = Policy((Tier.SMALL, Tier.MEDIUM), 0.8, 1000, Tier.MEDIUM)
    routing = TierRouting(policy, bindings, ANALYZER, SOURCES, FP)
    request = ChatCompletionRequest(
        model=TIER_ROUTING_ALIAS, messages=[Message(role="user", content="hi")]
    )
    action = start(request, CAPS, tier_routing=routing, use_cache=False)
    asked: list[str] = []
    while not isinstance(action, steps.Return | steps.Raise):
        asked.append(type(action).__name__)
        outcome: steps.Outcome
        if isinstance(action, steps.NewId):
            assert action.prefix == "req"
            outcome = steps.IdOutcome("req-1")
        elif isinstance(action, steps.Emit):
            assert not any(e.request_id.startswith("prep-") for e in action.events)
            outcome = steps.EmitOutcome()
        elif isinstance(action, steps.Analyze):
            assert action.capped is True
            outcome = _answer("SMALL 0.9")
        elif isinstance(action, steps.Dispatch):
            outcome = _answer("hello")
        elif isinstance(action, steps.CountTokens):
            outcome = steps.CountOutcome(1, 1, True)
        elif isinstance(action, steps.StartPreparation):
            assert action.fingerprint == FP
            outcome = steps.StartOutcome(True, PrepState.PREPARING)
        else:
            raise AssertionError(type(action).__name__)
        action = advance(action, outcome)
    assert isinstance(action, steps.Return)
    # Preparation is the driver's background work; the call itself only asks for it.
    assert asked.count("StartPreparation") == 1
    assert "StoreWrite" not in asked and "PreparationDone" not in asked


def test_parse_content_applies_the_stored_profile_rules() -> None:
    assert parse_content(GOOD) == ProfileContent(
        purpose="Answers questions about orders.",
        interactions=("order status questions",),
        constraints=("reply in English",),
        capabilities=("look up an order",),
        unknowns=("refund policy",),
    )
    with pytest.raises(InvalidProfileError, match="missing unknowns"):
        parse_content(_without("unknowns"))
    with pytest.raises(InvalidProfileError):
        parse_content(EMPTY)


def test_a_context_window_error_is_a_final_overflow_with_its_attempt() -> None:
    too_long = _error(errors.ContextWindowExceededError, 400)
    trace = _drive([_answer("prose"), too_long])
    assert trace.done is not None and trace.done.state is PrepState.FAILED
    assert trace.failures() == [("malformed_output", 1, False), ("overflow", 2, True)]
    assert len(trace.asked) == 2 and trace.writes == []


def test_an_unpriced_analyzer_emits_cost_unknown_before_going_on() -> None:
    trace = _drive([_answer(GOOD)], analyzer="unpriced-local-model", ceiling=1.0)
    (unknown,) = [e for e in trace.seen if isinstance(e, events.CostUnknown)]
    assert unknown.request_id == PREP_ID and unknown.model == "openai/unpriced-local-model"
    assert trace.types == ["preparation_started", "cost_unknown", "profile_ready"]
    assert trace.spends == []
