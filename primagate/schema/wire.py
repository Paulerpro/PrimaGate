from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    """A provider HTTP request, described as data so the adapter stays free of I/O."""

    method: str
    url: str
    # Headers carry the provider key, so they stay out of repr.
    headers: dict[str, str] = field(repr=False)
    json_body: dict[str, Any]


@dataclass(frozen=True)
class RawOutcome:
    """What transport saw. Either an HTTP exchange or a failure kind, never an error class."""

    status: int | None
    headers: dict[str, str]
    body: bytes
    failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None and self.status is not None and 200 <= self.status < 300


@dataclass(frozen=True)
class RawEvent:
    """One server-sent event, undecoded beyond its framing."""

    event: str
    data: str


class StreamFailure(Exception):
    """Transport lost a stream after it opened; `kind` is a failure kind, never a mapped error."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _no_model_hints(model: str) -> frozenset[str]:
    return frozenset()


def _never(model: str) -> bool:
    return False


@dataclass(frozen=True)
class Capabilities:
    """Features the adapter implements today, not what the provider could do."""

    tools: bool = False
    structured_output: bool = False
    images: bool = False
    streaming: bool = False
    # Hint-type parameters (not behavior features) the provider cannot honor.
    unsupported_hints: frozenset[str] = frozenset()
    # Hints that only some of the provider's models reject, so the rule depends on the model name.
    model_hints: Callable[[str], frozenset[str]] = _no_model_hints
    # Whether a model reasons before answering unless told not to, which uses up a small
    # `max_tokens` before any text appears.
    thinks_by_default: Callable[[str], bool] = _never

    def hints_for(self, model: str) -> frozenset[str]:
        """Every hint-type parameter `model` cannot honor."""
        return self.unsupported_hints | self.model_hints(model)


@dataclass(frozen=True)
class ProviderSettings:
    api_key: str = field(repr=False)
    base_url: str


@dataclass(frozen=True)
class Endpoint:
    """A redundant place to serve a model from, on the same provider as that model."""

    base_url: str
    # None means the provider's own key; it stays out of repr like every other key.
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RetryPolicy:
    """Per-endpoint retry budget and backoff; starting values, tuned by testing."""

    max_retries: int = 2
    backoff_base: float = 0.5
    backoff_max: float = 8.0


DEFAULT_RETRY = RetryPolicy()


@dataclass(frozen=True)
class Fallbacks:
    """The Caller's declared fallback models, as resolved (provider, model) pairs."""

    # An entry here, even an empty one, replaces `default` for that model.
    per_model: Mapping[tuple[str, str], tuple[tuple[str, str], ...]]
    default: tuple[tuple[str, str], ...] = ()


NO_FALLBACKS = Fallbacks(MappingProxyType({}))
