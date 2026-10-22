from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from primagate.schema.errors import PrimaGateError
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


Action = Dispatch | Return | Raise | Random | Sleep

# Drivers must handle exactly these; a test fails when one lacks a handler.
ACTION_TYPES: tuple[type[Action], ...] = (Dispatch, Return, Raise, Random, Sleep)


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


Outcome = DispatchOutcome | RandomOutcome | SleepOutcome
