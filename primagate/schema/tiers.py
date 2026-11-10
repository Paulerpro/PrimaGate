from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Final


class Tier(IntEnum):
    """Ordered so that a cheaper Tier compares lower than a stronger one."""

    SMALL = 1
    MEDIUM = 2
    POWERFUL = 3


class Unknown:
    """The one value for a fact a provider does not disclose.

    A distinct type, so it can never be mistaken for a real number, an empty list or `None`.
    """

    _instance: Unknown | None = None

    def __new__(cls) -> Unknown:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    # Copies and pickles must come back as the same object, or `is UNKNOWN` would stop working.
    def __reduce__(self) -> str:
        return "UNKNOWN"

    def __copy__(self) -> Unknown:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Unknown:
        return self


UNKNOWN: Final = Unknown()

# What a binding's limits may claim a model can do; the same names the adapters declare.
FEATURES: Final = ("tools", "structured_output", "images", "streaming")


@dataclass(frozen=True)
class BindingLimits:
    context_window: int | Unknown = UNKNOWN
    max_output_tokens: int | Unknown = UNKNOWN
    supported_features: frozenset[str] | Unknown = UNKNOWN


@dataclass(frozen=True)
class BindingPricing:
    """USD per million tokens."""

    input: float | Unknown = UNKNOWN
    output: float | Unknown = UNKNOWN


@dataclass(frozen=True)
class TierBinding:
    """What one Tier means in the Caller's app: model, settings, limits and price."""

    tier: Tier
    provider: str
    model_id: str
    # The Caller's own choices, so one not chosen is absent rather than unknown.
    settings: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    limits: BindingLimits = field(default_factory=BindingLimits)
    pricing: BindingPricing = field(default_factory=BindingPricing)

    def to_dict(self) -> dict[str, Any]:
        features = self.limits.supported_features
        return {
            "tier": self.tier.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "settings": {
                k: list(v) if isinstance(v, tuple) else v for k, v in self.settings.items()
            },
            "limits": {
                "context_window": _shown(self.limits.context_window),
                "max_output_tokens": _shown(self.limits.max_output_tokens),
                "supported_features": (
                    sorted(features) if isinstance(features, frozenset) else _shown(features)
                ),
            },
            "pricing": {
                "input": _shown(self.pricing.input),
                "output": _shown(self.pricing.output),
            },
        }


def _shown(value: object) -> Any:
    return "unknown" if isinstance(value, Unknown) else value
