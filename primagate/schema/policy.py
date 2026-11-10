from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from primagate.schema.tiers import Tier

COST_PREFERENCES: Final = ("balanced", "cost_conscious", "capability_first")


@dataclass(frozen=True)
class Policy:
    """The app's Tier Routing rules, fixed at setup so a request can rely on them."""

    enabled_tiers: tuple[Tier, ...]
    confidence_threshold: float
    context_token_budget: int
    interim_binding: Tier
    cost_preference: str = "balanced"
    ambiguous_context_default: Tier = Tier.MEDIUM
    # Unset until the cap is built, which is where its default is chosen.
    max_concurrent_analyzer_calls: int | None = None
    spending_ceiling: float | None = None
    # A short hash of every field above, so a decision can name the exact rules it used.
    version: str = ""
