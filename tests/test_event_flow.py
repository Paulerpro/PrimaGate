import dataclasses
import re

import pytest

from primagate.core import steps
from primagate.core.call import advance, start
from primagate.schema import errors, events
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    Message,
)
from primagate.schema.wire import Capabilities, Endpoint, Fallbacks, RetryPolicy

POLICY = RetryPolicy(max_retries=1, backoff_base=0.5, backoff_max=1.0)
CAPS = {"openai": Capabilities(), "anthropic": Capabilities()}
B = Endpoint("https://b.test/v1")
FALLBACKS = Fallbacks({}, default=(("anthropic", "claude-sonnet-5-5"),))


def _request(**changes: object) -> ChatCompletionRequest:
    base = ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="hi")])
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def _ok() -> ChatCompletion:
    message = Message(role="assistant", content="yo")
    return ChatCompletion(
        id="c", created=1, model="gpt-4o", choices=[Choice(index=0, message=message)]
    )


def _drive(
    first: steps.Action, script: list[steps.DispatchOutcome], request_id: str = "req-1"
) -> tuple[list[events.Event], steps.Action]:
    """Run the pure core against scripted dispatch outcomes, as a driver would."""
    seen: list[events.Event] = []
    outcomes = iter(script)
    action = first
    while not isinstance(action, steps.Return | steps.Raise):
        if isinstance(action, steps.NewId):
            outcome: steps.Outcome = steps.IdOutcome(request_id)
        elif isinstance(action, steps.Emit):
            seen.extend(action.events)
            outcome = steps.EmitOutcome()
        elif isinstance(action, steps.Random):
            outcome = steps.RandomOutcome(1.0)
        elif isinstance(action, steps.Sleep):
            outcome = steps.SleepOutcome()
        elif isinstance(action, steps.CountTokens):
            outcome = steps.CountOutcome(1, 1, True)
        else:
            outcome = next(outcomes)
        action = advance(action, outcome)
    return seen, action


def _fail(status: int, cls: type[errors.PrimaGateError]) -> steps.DispatchOutcome:
    return steps.DispatchOutcome(error=cls("boom", llm_provider="openai", status_code=status))


def _types(seen: list[events.Event]) -> list[str]:
    return [e.event_type for e in seen]


def test_every_catalog_event_is_frozen_with_a_snake_case_type() -> None:
    assert len(events.EVENT_TYPES) == len(set(events.EVENT_TYPES))
    for cls in events.EVENT_CLASSES:
        assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
        assert re.fullmatch(r"[a-z]+(_[a-z]+)*", cls.event_type), cls
    assert {c.event_type for c in events.EVENT_CLASSES} == set(events.EVENT_TYPES)


def test_events_serialize_with_their_type_and_without_empty_fields() -> None:
    event = events.CallFailed(request_id="r", error_type="Timeout", status_code=None, attempts=1)
    assert event.to_dict() == {
        "event_type": "call_failed",
        "request_id": "r",
        "timestamp": "",
        "error_type": "Timeout",
        "attempts": 1,
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.request_id = "other"  # type: ignore[misc]


def test_a_clean_call_emits_started_then_succeeded_with_one_id() -> None:
    first = start(_request(), CAPS)
    assert isinstance(first, steps.NewId)
    seen, last = _drive(first, [steps.DispatchOutcome(response=_ok())])
    assert _types(seen) == ["dispatch_started", "call_succeeded"]
    assert {e.request_id for e in seen} == {"req-1"}
    done = seen[1]
    assert isinstance(done, events.CallSucceeded)
    assert (done.model, done.attempts, done.dropped_params) == ("openai/gpt-4o", 1, ())
    assert isinstance(last, steps.Return)


def test_core_never_stamps_a_timestamp() -> None:
    seen, _ = _drive(start(_request(), CAPS), [steps.DispatchOutcome(response=_ok())])
    assert all(e.timestamp == "" for e in seen)


def test_a_retry_emits_failure_delay_and_a_second_start() -> None:
    routing = {("openai", "gpt-4o"): (B,)}
    first = start(_request(), CAPS, routing=routing, retry=POLICY)
    script = [_fail(503, errors.ServiceUnavailableError), steps.DispatchOutcome(response=_ok())]
    seen, _ = _drive(first, script)
    assert _types(seen) == [
        "dispatch_started",
        "dispatch_failed",
        "retry_scheduled",
        "dispatch_started",
        "call_succeeded",
    ]
    failed, scheduled = seen[1], seen[2]
    assert isinstance(failed, events.DispatchFailed) and isinstance(
        scheduled, events.RetryScheduled
    )
    assert (failed.error_type, failed.status_code, failed.next) == (
        "ServiceUnavailableError",
        503,
        "retry",
    )
    assert scheduled.delay == 0.5 and scheduled.retry == 1
    assert isinstance(seen[3], events.DispatchStarted) and seen[3].retries == 1


def test_moving_to_the_redundant_endpoint_says_next_endpoint() -> None:
    routing = {("openai", "gpt-4o"): (B,)}
    first = start(_request(), CAPS, routing=routing, retry=RetryPolicy(0, 0.5, 1.0))
    script = [_fail(503, errors.ServiceUnavailableError), steps.DispatchOutcome(response=_ok())]
    seen, _ = _drive(first, script)
    failed = next(e for e in seen if isinstance(e, events.DispatchFailed))
    assert failed.next == "next_endpoint"
    started = [e for e in seen if isinstance(e, events.DispatchStarted)]
    assert [s.endpoint for s in started] == ["openai", B.base_url]


def test_total_failure_ends_with_one_call_failed_counting_attempts() -> None:
    first = start(_request(), CAPS, fallbacks=FALLBACKS, retry=POLICY)
    script = [_fail(503, errors.ServiceUnavailableError), _fail(429, errors.RateLimitError)]
    seen, last = _drive(first, script)
    assert _types(seen) == [
        "dispatch_started",
        "dispatch_failed",
        "dispatch_started",
        "dispatch_failed",
        "call_failed",
    ]
    assert [e.next for e in seen if isinstance(e, events.DispatchFailed)] == [
        "next_attempt",
        "raise",
    ]
    final = seen[-1]
    assert isinstance(final, events.CallFailed)
    assert (final.error_type, final.status_code, final.attempts) == (
        "AttemptsExhaustedError",
        429,
        2,
    )
    assert isinstance(last, steps.Raise)


def test_a_skipped_fallback_is_reported_with_its_reason() -> None:
    caps = {"openai": Capabilities(tools=True), "anthropic": Capabilities()}
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    first = start(_request(tools=[tool]), caps, fallbacks=FALLBACKS, retry=POLICY)
    seen, _ = _drive(first, [_fail(503, errors.ServiceUnavailableError)])
    skipped = next(e for e in seen if isinstance(e, events.AttemptSkipped))
    assert skipped.model == "anthropic/claude-sonnet-5-5"
    assert "tools" in skipped.reason
    final = seen[-1]
    assert isinstance(final, events.CallFailed) and final.attempts == 1


def test_failing_before_any_dispatch_emits_only_call_failed() -> None:
    seen, last = _drive(start(_request(model="nope/x"), CAPS), [])
    assert _types(seen) == ["call_failed"]
    assert seen[0].request_id == "req-1"
    assert isinstance(last, steps.Raise)


def test_events_hold_no_message_text_or_error_text() -> None:
    secret = _request(messages=[Message(role="user", content="TOP-SECRET-CONTENT")])
    bad = steps.DispatchOutcome(
        error=errors.BadRequestError("echoed TOP-SECRET-CONTENT", status_code=400)
    )
    seen, _ = _drive(start(secret, CAPS), [bad])
    assert "TOP-SECRET-CONTENT" not in repr(seen)


def test_emit_and_new_id_are_part_of_the_closed_action_set() -> None:
    assert {steps.NewId, steps.Emit} <= set(steps.ACTION_TYPES)


def test_an_outcome_that_does_not_answer_the_action_is_refused() -> None:
    action = advance(start(_request(), CAPS), steps.IdOutcome("r"))
    assert isinstance(action, steps.Emit)
    bad = advance(action, steps.SleepOutcome())
    assert isinstance(bad, steps.Raise)


def test_payload_retention_is_off_unless_asked_and_core_then_builds_no_content() -> None:
    seen, _ = _drive(start(_request(), CAPS), [steps.DispatchOutcome(response=_ok())])
    started, done = seen
    assert isinstance(started, events.DispatchStarted) and started.messages is None
    assert isinstance(done, events.CallSucceeded) and done.reply_messages is None


def test_retention_carries_the_request_and_the_reply_and_survives_a_fallback() -> None:
    first = start(_request(), CAPS, fallbacks=FALLBACKS, retry=POLICY, payload_retention=True)
    script = [_fail(503, errors.ServiceUnavailableError), steps.DispatchOutcome(response=_ok())]
    seen, _ = _drive(first, script)
    started = [e for e in seen if isinstance(e, events.DispatchStarted)]
    assert len(started) == 2
    assert all(s.messages == ({"role": "user", "content": "hi"},) for s in started)
    done = seen[-1]
    assert isinstance(done, events.CallSucceeded)
    assert done.reply_messages == ({"role": "assistant", "content": "yo"},)


def test_error_text_stays_out_of_events_even_with_retention() -> None:
    bad = steps.DispatchOutcome(error=errors.BadRequestError("echoed hi", status_code=400))
    seen, _ = _drive(start(_request(), CAPS, payload_retention=True), [bad])
    failures = [e for e in seen if not isinstance(e, events.DispatchStarted)]
    assert "echoed" not in repr(failures)


@pytest.mark.parametrize(
    ("base_url", "shown"),
    [
        ("https://u:pw@b.test/v1?api-key=K#f", "https://b.test/v1"),
        ("https://b.test:8443/v1", "https://b.test:8443/v1"),
        ("http://b.test", "http://b.test"),
        ("user:pw@b.test/v1", "openai"),
        ("b.test/v1?k=K", "openai"),
        ("http://[bad", "openai"),
        ("ftp://u:pw@b.test/x", "openai"),
        ("https://", "openai"),
    ],
)
def test_the_endpoint_in_an_event_never_carries_credentials(base_url: str, shown: str) -> None:
    routing = {("openai", "gpt-4o"): (Endpoint(base_url),)}
    first = start(_request(), CAPS, routing=routing, retry=RetryPolicy(0, 0.5, 1.0))
    script = [_fail(503, errors.ServiceUnavailableError), steps.DispatchOutcome(response=_ok())]
    seen, _ = _drive(first, script)
    started = [e for e in seen if isinstance(e, events.DispatchStarted)]
    assert [s.endpoint for s in started] == ["openai", shown]
