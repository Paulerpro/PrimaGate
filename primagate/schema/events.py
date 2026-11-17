from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal

# What the SDK does after a failed dispatch.
NextStep = Literal["retry", "next_endpoint", "next_attempt", "raise"]


@dataclass(frozen=True)
class Event:
    """One thing the SDK did, as an immutable record.

    Events never hold message content, error text or keys: they say what happened, not what was
    said, so they are safe to log whatever the caller's privacy setting.
    """

    event_type: ClassVar[str]
    # The stdlib logging level the event is written at.
    log_level: ClassVar[int] = logging.INFO

    request_id: str
    # Set by the driver when the event is delivered; core never reads a clock.
    timestamp: str = field(default="", kw_only=True)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"event_type": self.event_type}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


@dataclass(frozen=True)
class DispatchStarted(Event):
    event_type: ClassVar[str] = "dispatch_started"

    model: str
    endpoint: str
    retries: int
    # The request's messages; set only when the caller opted in to payload retention.
    messages: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class DispatchFailed(Event):
    event_type: ClassVar[str] = "dispatch_failed"

    model: str
    endpoint: str
    error_type: str
    status_code: int | None
    next: NextStep


@dataclass(frozen=True)
class RetryScheduled(Event):
    event_type: ClassVar[str] = "retry_scheduled"

    model: str
    endpoint: str
    retry: int
    delay: float


@dataclass(frozen=True)
class AttemptSkipped(Event):
    event_type: ClassVar[str] = "attempt_skipped"

    model: str
    reason: str


@dataclass(frozen=True)
class CacheHit(Event):
    """A call was answered from the Cache. Only the model and a salted hash, never content."""

    event_type: ClassVar[str] = "cache_hit"

    model: str
    key: str


@dataclass(frozen=True)
class CostUnknown(Event):
    """The answering model has no price, so the response carries no cost Estimate."""

    event_type: ClassVar[str] = "cost_unknown"
    log_level: ClassVar[int] = logging.WARNING

    model: str


@dataclass(frozen=True)
class CallSucceeded(Event):
    event_type: ClassVar[str] = "call_succeeded"

    model: str
    attempts: int
    dropped_params: tuple[str, ...]
    # The reply's messages; set only when the caller opted in to payload retention.
    reply_messages: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class CallFailed(Event):
    event_type: ClassVar[str] = "call_failed"

    error_type: str
    status_code: int | None
    attempts: int


@dataclass(frozen=True)
class DecisionEvent(Event):
    """What an operator needs to explain one routing decision, without any conversation content.

    Every field is always present. With no READY Profile, as for an Interim Binding request, the
    value is "unknown", the word the Tier Binding contract uses for an undisclosed field, so a
    lookup never meets a missing key.
    """

    profile_id: str
    input_fingerprint: str
    prep_prompt_version: str
    analyzer_prompt_version: str
    binding_id: str

    def __post_init__(self) -> None:
        # The base has no event_type, so an instance of it could not be logged or serialised.
        if type(self) is DecisionEvent:
            raise TypeError("DecisionEvent is a base class; use one of its subclasses")


@dataclass(frozen=True)
class RoutingDecision(DecisionEvent):
    """A request was routed to a Tier."""

    event_type: ClassVar[str] = "routing_decision"


@dataclass(frozen=True)
class LowConfidenceDowngrade(DecisionEvent):
    """The Analyzer answered below the confidence threshold, so the default Tier was used."""

    event_type: ClassVar[str] = "low_confidence_downgrade"


@dataclass(frozen=True)
class PremiumRoutingSelected(DecisionEvent):
    """The resolved Tier costs more than the cheapest eligible candidate."""

    event_type: ClassVar[str] = "premium_routing_selected"


@dataclass(frozen=True)
class RoutingDegraded(DecisionEvent):
    """Routing could not decide, so the Interim Binding was used; `reason` says why."""

    event_type: ClassVar[str] = "routing_degraded"
    log_level: ClassVar[int] = logging.WARNING

    # One of a fixed set of names, never provider text, so the event is safe to log.
    reason: str


EVENT_CLASSES: tuple[type[Event], ...] = (
    DispatchStarted,
    DispatchFailed,
    RetryScheduled,
    AttemptSkipped,
    CacheHit,
    CostUnknown,
    CallSucceeded,
    CallFailed,
    RoutingDecision,
    LowConfidenceDowngrade,
    PremiumRoutingSelected,
    RoutingDegraded,
)
EVENT_TYPES: tuple[str, ...] = tuple(cls.event_type for cls in EVENT_CLASSES)
# Names the package root re-exports.
EXPORTED: tuple[str, ...] = ("Event", "DecisionEvent", *(cls.__name__ for cls in EVENT_CLASSES))
