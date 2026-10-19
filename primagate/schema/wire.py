from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Capabilities:
    """Features the adapter implements today, not what the provider could do."""

    tools: bool = False
    structured_output: bool = False
    images: bool = False
    streaming: bool = False
    # Hint-type parameters (not behavior features) the provider cannot honor.
    unsupported_hints: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ProviderSettings:
    api_key: str = field(repr=False)
    base_url: str
