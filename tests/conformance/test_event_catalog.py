import dataclasses
import re
from typing import Any

import pytest
from conftest import Caller
from test_cache import GPT, MSGS, OPENAI, _answer
from test_events import Recorder
from test_retry_routing import Run, _status

import primagate
from primagate.schema import errors, events

DECISION_FIELDS = (
    "profile_id",
    "input_fingerprint",
    "prep_prompt_version",
    "analyzer_prompt_version",
    "binding_id",
)
DECISION = dict.fromkeys(DECISION_FIELDS, "unknown")

# One valid instance of every catalog class. A class missing here fails the test below, so a new
# event cannot join the catalog without its shape being checked.
EXAMPLES: dict[str, events.Event] = {
    "DispatchStarted": events.DispatchStarted("req-1", "gpt-4o", "openai", 0),
    "DispatchFailed": events.DispatchFailed(
        "req-1", "gpt-4o", "openai", "ServiceUnavailableError", 503, "retry"
    ),
    "RetryScheduled": events.RetryScheduled("req-1", "gpt-4o", "openai", 1, 0.5),
    "AttemptSkipped": events.AttemptSkipped("req-1", "claude-sonnet-5-5", "no tools"),
    "CacheHit": events.CacheHit("req-1", "gpt-4o", "abc"),
    "CostUnknown": events.CostUnknown("req-1", "mystery"),
    "CallSucceeded": events.CallSucceeded("req-1", "gpt-4o", 1, ()),
    "CallFailed": events.CallFailed("req-1", "ServiceUnavailableError", 503, 1),
    "RoutingDecision": events.RoutingDecision("req-1", **DECISION),
    "LowConfidenceDowngrade": events.LowConfidenceDowngrade("req-1", **DECISION),
    "PremiumRoutingSelected": events.PremiumRoutingSelected("req-1", **DECISION),
    "RoutingDegraded": events.RoutingDegraded("req-1", **DECISION, reason="over_budget"),
    "SpendingCeilingExceeded": events.SpendingCeilingExceeded("req-1", 1.0, 1.5),
    "PreparationStarted": events.PreparationStarted("prep-a", "fp-1", "prompt-v1"),
    "PreparationFailed": events.PreparationFailed(
        "prep-a", "fp-1", "prompt-v1", "malformed_output", 2, False
    ),
    "ProfileReady": events.ProfileReady("prep-a", "fp-1", "prompt-v1", "profile-b"),
}

DECISION_CLASSES = (
    events.RoutingDecision,
    events.LowConfidenceDowngrade,
    events.PremiumRoutingSelected,
    events.RoutingDegraded,
)


def _check_shape(event: events.Event) -> None:
    cls = type(event)
    assert dataclasses.is_dataclass(cls)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.request_id = "changed"  # type: ignore[misc]
    assert re.fullmatch(r"[a-z]+(_[a-z]+)*", event.event_type)
    assert dataclasses.fields(cls)[0].name == "request_id"
    as_dict = event.to_dict()
    assert as_dict["event_type"] == event.event_type
    for field in dataclasses.fields(cls):
        required = (
            field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        )
        if required:
            assert field.name in as_dict, f"{cls.__name__}.{field.name} is missing from to_dict"
            assert as_dict[field.name] is not None


def test_every_catalog_class_has_an_example_and_a_unique_snake_case_type() -> None:
    assert {cls.__name__ for cls in events.EVENT_CLASSES} == set(EXAMPLES)
    assert len(set(events.EVENT_TYPES)) == len(events.EVENT_TYPES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_each_catalog_event_has_the_expected_shape(name: str) -> None:
    event = EXAMPLES[name]
    assert type(event).__name__ == name
    _check_shape(event)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_catalog_event_is_exported_from_the_package_root(name: str) -> None:
    assert name in events.EXPORTED
    assert getattr(primagate, name) is type(EXAMPLES[name])


def test_the_decision_types_are_distinct_catalog_entries() -> None:
    types = [cls.event_type for cls in DECISION_CLASSES]
    assert types == [
        "routing_decision",
        "low_confidence_downgrade",
        "premium_routing_selected",
        "routing_degraded",
    ]
    assert all(cls in events.EVENT_CLASSES for cls in DECISION_CLASSES)


@pytest.mark.parametrize("cls", DECISION_CLASSES)
def test_a_decision_event_carries_the_traceability_fields(cls: type[events.DecisionEvent]) -> None:
    assert issubclass(cls, events.DecisionEvent)
    names = [f.name for f in dataclasses.fields(cls)]
    assert names[0] == "request_id" and set(DECISION_FIELDS) <= set(names)
    extra = {"reason": "over_budget"} if cls is events.RoutingDegraded else {}
    shown = cls("req-1", **DECISION, **extra).to_dict()
    assert all(shown[name] == "unknown" for name in DECISION_FIELDS)


@pytest.mark.parametrize("missing", DECISION_FIELDS)
def test_a_decision_event_without_one_of_the_fields_cannot_be_built(missing: str) -> None:
    fields: dict[str, Any] = {k: v for k, v in DECISION.items() if k != missing}
    with pytest.raises(TypeError):
        events.RoutingDecision("req-1", **fields)


def test_the_decision_base_cannot_be_built_and_is_exported() -> None:
    with pytest.raises(TypeError):
        events.DecisionEvent("req-1", **DECISION)
    assert "DecisionEvent" in events.EXPORTED
    assert primagate.DecisionEvent is events.DecisionEvent


def test_only_the_decision_types_are_decision_events() -> None:
    others = [cls for cls in events.EVENT_CLASSES if cls not in DECISION_CLASSES]
    assert not any(issubclass(cls, events.DecisionEvent) for cls in others)


def test_every_event_a_real_call_delivers_has_the_catalog_shape(
    call: Caller, delays: list[float]
) -> None:
    handler = Recorder()
    retried = Run(
        {OPENAI: [_status(503), _answer]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
        event_handler=handler,
    )
    call(retried.gate, model=GPT, messages=MSGS)
    call(retried.gate, model=GPT, messages=MSGS)
    failing = Run({OPENAI: [_status(400)]}, event_handler=handler)
    with pytest.raises(errors.BadRequestError):
        call(failing.gate, model=GPT, messages=MSGS)
    assert {"dispatch_started", "dispatch_failed", "retry_scheduled"} <= set(handler.types)
    assert {"call_succeeded", "call_failed", "cache_hit"} <= set(handler.types)
    for event in handler.seen:
        assert event.event_type in events.EVENT_TYPES
        assert event.request_id.startswith("req-")
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT[\d:.]+(Z|\+00:00)", event.timestamp), event.timestamp
        _check_shape(event)


def test_a_cache_hit_emits_the_cache_event_and_no_decision_event(call: Caller) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler)
    call(run.gate, model=GPT, messages=MSGS)
    first = len(handler.seen)
    call(run.gate, model=GPT, messages=MSGS)
    assert [e.event_type for e in handler.seen[first:]] == ["cache_hit"]
    assert not any(isinstance(e, events.DecisionEvent) for e in handler.seen)
