from __future__ import annotations

from dataclasses import dataclass

from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest


@dataclass(frozen=True)
class Dispatch:
    provider: str
    request: ChatCompletionRequest
    dropped_params: tuple[str, ...] = ()


@dataclass(frozen=True)
class Return:
    response: ChatCompletion


@dataclass(frozen=True)
class Raise:
    error: PrimaGateError


Action = Dispatch | Return | Raise

# Drivers must handle exactly these; a test fails when one lacks a handler.
ACTION_TYPES: tuple[type[Action], ...] = (Dispatch, Return, Raise)


@dataclass(frozen=True)
class DispatchOutcome:
    """Result of running a Dispatch: a response or the mapped error, never both."""

    response: ChatCompletion | None = None
    error: PrimaGateError | None = None
    dropped_params: tuple[str, ...] = ()
