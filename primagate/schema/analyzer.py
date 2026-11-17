from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from primagate.schema.tiers import Tier
from primagate.schema.types import AnalyzerUsage

# Fixed names, so a failure can be counted and logged without ever holding what the model said.
FailureReason = Literal["no_conversation", "provider_error", "malformed_output", "tier_not_enabled"]
VerdictReason = Literal["accepted", "ambiguous_context_default"]


@dataclass(frozen=True)
class Verdict:
    """The Tier the Analyzer chose, or the default when it was not confident enough."""

    tier: Tier
    confidence: float
    # "accepted", or "ambiguous_context_default" when the answer was below the threshold.
    reason: VerdictReason
    low_confidence_downgrade: bool


@dataclass(frozen=True)
class AnalyzerFailure:
    """An analysis that gave no usable Tier; what to do about it is the router's decision."""

    reason: FailureReason


@dataclass(frozen=True)
class Analysis:
    outcome: Verdict | AnalyzerFailure
    # None when nothing was answered, so nothing was spent.
    usage: AnalyzerUsage | None


__all__ = [
    "Analysis",
    "AnalyzerFailure",
    "AnalyzerUsage",
    "FailureReason",
    "Verdict",
    "VerdictReason",
]
