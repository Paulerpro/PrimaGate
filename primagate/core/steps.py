from __future__ import annotations

from dataclasses import dataclass

from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, DispatchRecord
from primagate.schema.wire import Endpoint, RetryPolicy


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
