from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from primagate.core.prices import BUNDLED, PriceTable
from primagate.schema.errors import PrimaGateError
from primagate.schema.events import Event
from primagate.schema.profile import PrepState, Profile
from primagate.schema.tiers import Tier
from primagate.schema.types import (
    AnalyzerUsage,
    Attempt,
    ChatCompletion,
    ChatCompletionRequest,
    DispatchRecord,
)
from primagate.schema.wire import Capabilities, Endpoint, RetryPolicy


@dataclass(frozen=True)
class DropPolicy:
    """What the caller allowed to be dropped: hint parameters, and/or the named features."""

    hints: bool = False
    features: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Plan:
    """What the next candidate needs, kept with the call.

    Each fallback is judged by its own provider's capabilities and routing when its turn comes.
    The failover slot is not: the Compatibility Filter builds it ahead, when routing runs, so
    MEDIUM is judged with its binding's narrowed limits exactly as routing judged it.
    """

    # The Caller's request before any drops or model-name stripping.
    request: ChatCompletionRequest
    capabilities: Mapping[str, Capabilities]
    drops: DropPolicy
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]]
    retry: RetryPolicy
    later: tuple[tuple[str, str], ...]
    # Tried before `later`: MEDIUM's dispatch as the filter built it for this request, or the
    # skipped row saying why MEDIUM cannot run. None when the call has no Tier Failover.
    failover: Dispatch | Attempt | None = None


@dataclass(frozen=True)
class Route:
    """Provider Routing for one model: the redundant endpoints that follow the primary."""

    redundant: tuple[Endpoint, ...]
    policy: RetryPolicy


@dataclass(frozen=True)
class Dispatch:
    provider: str
    request: ChatCompletionRequest
    dropped_params: tuple[str, ...] = ()
    # None is the provider's configured endpoint; otherwise one of the route's redundant ones.
    endpoint: Endpoint | None = None
    # 0 is the primary endpoint, n the n-th redundant one.
    position: int = 0
    retries: int = 0
    # None means Provider Routing is off: one direct dispatch, no retries.
    route: Route | None = None
    # Rows of the endpoints already given up on, so any later error carries the whole chain.
    history: tuple[DispatchRecord, ...] = ()
    # Attempts already over, for models tried before this one.
    past: tuple[Attempt, ...] = ()
    # None when no fallback remains.
    plan: Plan | None = None
    # Empty only in core-level tests that build a Dispatch by hand.
    request_id: str = ""
    # Carried with the call so every event it builds follows the caller's privacy choice.
    payload_retention: bool = False
    # None for a streamed call, which never uses the Cache.
    cache_key: str | None = None
    # Set for an answer that must never be replayed, such as one from a degraded route.
    no_cache: bool = False
    # Bundled prices with the caller's overrides on top; carried so the answer can be priced.
    prices: PriceTable = field(default_factory=lambda: BUNDLED)
    # The Tier this dispatch serves under Tier Routing, and what choosing it cost; None otherwise.
    tier: Tier | None = None
    analyzer: AnalyzerUsage | None = None
    # The Policy's advisory spending ceiling, carried so the finished call can be counted.
    spending_ceiling: float | None = None


@dataclass(frozen=True)
class Return:
    response: ChatCompletion


@dataclass(frozen=True)
class Raise:
    error: PrimaGateError


@dataclass(frozen=True)
class Random:
    """Ask the driver for a random value in [0, 1) to jitter the next delay.

    A call's retry resumes its Dispatch, whose retry policy sets the delay. Any other wait
    resumes a function that turns the value into its next action.
    """

    resume: Dispatch | Callable[[float], Action]


@dataclass(frozen=True)
class Sleep:
    seconds: float
    resume: Action


@dataclass(frozen=True)
class NewId:
    """Ask the driver for a new id; `resume` turns it into the next action.

    The prefix tells a call's events from a preparation's, so the driver must keep it.
    """

    resume: Callable[[str], Action]
    prefix: str = "req"


@dataclass(frozen=True)
class Now:
    """Ask the driver for the current time as ISO 8601 UTC, since core reads no clock."""

    resume: Callable[[str], Action]


@dataclass(frozen=True)
class Emit:
    """Ask the driver to deliver events to the handler and the logger, then go on."""

    events: tuple[Event, ...]
    resume: Action


@dataclass(frozen=True)
class CountTokens:
    """Ask the driver to count the text of a response whose provider reported no usage."""

    model: str
    prompt: str
    completion: str
    resume: Callable[[CountOutcome], Action]
    # The call's own, so a cancellation here is reported against it.
    request_id: str = ""
    # The dispatch whose answer is counted, so a cancellation here still counts what an earlier
    # Analyzer request was billed.
    dispatch: Dispatch | None = None


@dataclass(frozen=True)
class CacheRead:
    """Ask the driver for the answer stored under `key`; `resume` handles a hit or a miss."""

    key: str
    resume: Callable[[CacheReadOutcome], Action]
    # The call's own, so a cancellation here is reported against it.
    request_id: str = ""


@dataclass(frozen=True)
class CacheWrite:
    """Ask the driver to store `response` under `key`, then go on with `resume`."""

    key: str
    response: ChatCompletion
    resume: Action
    # The call's own, so a cancellation here is reported against it.
    request_id: str = ""


@dataclass(frozen=True)
class StoreRead:
    """Ask the driver for the stored Profile; `resume` reads it from `outcome.profile`, None when
    the store is empty."""

    resume: Callable[[StoreReadOutcome], Action]


@dataclass(frozen=True)
class StoreWrite:
    """Ask the driver to put `profile` in the store, then go on with `resume`."""

    profile: Profile
    resume: Action


@dataclass(frozen=True)
class Analyze:
    """Ask the driver to send the Analyzer's one dispatch; `resume` reads its outcome.

    A separate action from `Dispatch` so the Analyzer's call can never be retried, fallen back
    from or routed again.
    """

    dispatch: Dispatch
    resume: Callable[[DispatchOutcome | Declined], Action]
    # False only for preparation, which runs apart from requests, so it must not take a permit
    # a request is waiting on.
    capped: bool = True


@dataclass(frozen=True)
class RecordSpend:
    """Ask the driver to add `amount` (USD) to the running total against `ceiling`."""

    amount: float
    ceiling: float
    resume: Callable[[SpendOutcome], Action]
    # The call's own, so a cancellation here is reported against it.
    request_id: str = ""


@dataclass(frozen=True)
class StartPreparation:
    """Ask the driver to start preparing the Profile for `fingerprint` in the background.

    Fire-and-forget: the driver answers at once with whether one started, the state after that,
    and the Profile when READY. The call goes on with `resume` without waiting for any of it.
    """

    fingerprint: str
    resume: Callable[[StartOutcome], Action]
    # The call's own, so a cancellation here is reported against it.
    request_id: str = ""


@dataclass(frozen=True)
class PreparationDone:
    """The end of a preparation: its state, and the Profile it wrote when READY."""

    state: PrepState
    profile: Profile | None


@dataclass(frozen=True)
class Stopped:
    """The call was cancelled and has stopped; the driver turns this into the caller's
    cancellation."""


Action = (
    Dispatch
    | Return
    | Raise
    | Random
    | Sleep
    | NewId
    | Now
    | Emit
    | CountTokens
    | CacheRead
    | CacheWrite
    | StoreRead
    | StoreWrite
    | Analyze
    | RecordSpend
    | StartPreparation
    | PreparationDone
    | Stopped
)

# Drivers must handle exactly these; a test fails when one lacks a handler.
ACTION_TYPES: tuple[type[Action], ...] = (
    Dispatch,
    Return,
    Raise,
    Random,
    Sleep,
    NewId,
    Now,
    Emit,
    CountTokens,
    CacheRead,
    CacheWrite,
    StoreRead,
    StoreWrite,
    Analyze,
    RecordSpend,
    StartPreparation,
    PreparationDone,
    Stopped,
)


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of running a Dispatch: a response or the mapped error, never both."""

    response: ChatCompletion | None = None
    error: PrimaGateError | None = None
    dropped_params: tuple[str, ...] = ()


@dataclass(frozen=True)
class Declined:
    """The driver did not send what it was asked to, and says why; core decides what follows."""

    reason: str


@dataclass(frozen=True)
class SpendOutcome:
    total: float
    # True only for the first call to take the total past the ceiling.
    crossed: bool


@dataclass(frozen=True)
class StartOutcome:
    """Where preparation stands as the call starts, which the call keeps to its end."""

    # False when one is already running or done for the fingerprint, or may not start again yet.
    started: bool
    state: PrepState
    # Handed over with the state, so serving it never reads the store on the request path.
    profile: Profile | None = None


@dataclass(frozen=True)
class RandomOutcome:
    value: float


@dataclass(frozen=True)
class SleepOutcome:
    pass


@dataclass(frozen=True)
class IdOutcome:
    value: str


@dataclass(frozen=True)
class NowOutcome:
    value: str


@dataclass(frozen=True)
class EmitOutcome:
    pass


@dataclass(frozen=True)
class CountOutcome:
    prompt_tokens: int
    completion_tokens: int
    # True when a built-in rough count was used instead of a real tokenizer or the caller's.
    heuristic: bool


@dataclass(frozen=True)
class CacheReadOutcome:
    # None is a miss.
    response: ChatCompletion | None


@dataclass(frozen=True)
class CacheWriteOutcome:
    pass


@dataclass(frozen=True)
class StoreReadOutcome:
    # Whatever the store holds, unchecked; core decides whether it still fits.
    profile: Profile | None


@dataclass(frozen=True)
class StoreWriteOutcome:
    pass


@dataclass(frozen=True)
class Cancelled:
    """The caller stopped the call before the action was performed, or while it ran.

    An outcome, never an error, so core alone decides that nothing more is done.
    """


Outcome = (
    DispatchOutcome
    | Declined
    | SpendOutcome
    | StartOutcome
    | RandomOutcome
    | SleepOutcome
    | IdOutcome
    | NowOutcome
    | EmitOutcome
    | CountOutcome
    | CacheReadOutcome
    | CacheWriteOutcome
    | StoreReadOutcome
    | StoreWriteOutcome
    | Cancelled
)
