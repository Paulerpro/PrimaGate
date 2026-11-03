from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from primagate.core.prices import BUNDLED, PriceTable
from primagate.schema.errors import PrimaGateError
from primagate.schema.events import Event
from primagate.schema.types import (
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
    """What the next fallback needs, kept with the call because each candidate is judged by
    its own provider's capabilities and routing when its turn comes."""

    # The Caller's request before any drops or model-name stripping.
    request: ChatCompletionRequest
    capabilities: Mapping[str, Capabilities]
    drops: DropPolicy
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]]
    retry: RetryPolicy
    later: tuple[tuple[str, str], ...]


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
    # Bundled prices with the caller's overrides on top; carried so the answer can be priced.
    prices: PriceTable = field(default_factory=lambda: BUNDLED)


@dataclass(frozen=True)
class Return:
    response: ChatCompletion


@dataclass(frozen=True)
class Raise:
    error: PrimaGateError


@dataclass(frozen=True)
class Random:
    """Ask the driver for a random value in [0, 1) to jitter the next delay."""

    resume: Dispatch


@dataclass(frozen=True)
class Sleep:
    seconds: float
    resume: Dispatch


@dataclass(frozen=True)
class NewId:
    """Ask the driver for this call's request id; `resume` turns it into the next action."""

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


@dataclass(frozen=True)
class CacheRead:
    """Ask the driver for the answer stored under `key`; `resume` handles a hit or a miss."""

    key: str
    resume: Callable[[CacheReadOutcome], Action]


@dataclass(frozen=True)
class CacheWrite:
    """Ask the driver to store `response` under `key`, then go on with `resume`."""

    key: str
    response: ChatCompletion
    resume: Action


Action = (
    Dispatch | Return | Raise | Random | Sleep | NewId | Emit | CountTokens | CacheRead | CacheWrite
)

# Drivers must handle exactly these; a test fails when one lacks a handler.
ACTION_TYPES: tuple[type[Action], ...] = (
    Dispatch,
    Return,
    Raise,
    Random,
    Sleep,
    NewId,
    Emit,
    CountTokens,
    CacheRead,
    CacheWrite,
)


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of running a Dispatch: a response or the mapped error, never both."""

    response: ChatCompletion | None = None
    error: PrimaGateError | None = None
    dropped_params: tuple[str, ...] = ()


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


Outcome = (
    DispatchOutcome
    | RandomOutcome
    | SleepOutcome
    | IdOutcome
    | EmitOutcome
    | CountOutcome
    | CacheReadOutcome
    | CacheWriteOutcome
)
