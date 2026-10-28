from __future__ import annotations

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


EVENT_CLASSES: tuple[type[Event], ...] = (
    DispatchStarted,
    DispatchFailed,
    RetryScheduled,
    AttemptSkipped,
    CallSucceeded,
    CallFailed,
)
EVENT_TYPES: tuple[str, ...] = tuple(cls.event_type for cls in EVENT_CLASSES)
# Names the package root re-exports.
EXPORTED: tuple[str, ...] = ("Event", *(cls.__name__ for cls in EVENT_CLASSES))
