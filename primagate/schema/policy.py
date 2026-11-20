from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from primagate.schema.profile import ProfileSources
from primagate.schema.tiers import Tier, TierBinding

COST_PREFERENCES: Final = ("balanced", "cost_conscious", "capability_first")

# Passed as `model=` to ask for Tier Routing; any other name is a plain model and bypasses it. The
# prefix is not one of PrimaGate's own providers, so it cannot be mistaken for a model on one.
TIER_ROUTING_ALIAS: Final = "primagate/auto"


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
    # How many earlier exchanges the Analyzer may read, from 0 to 3.
    context_exchanges: int = 3
    # A short hash of every field above, so a decision can name the exact rules it used.
    version: str = ""


@dataclass(frozen=True)
class TierRouting:
    """The Policy and the Tier Bindings it enables, taken together when a call starts."""

    policy: Policy
    bindings: Mapping[Tier, TierBinding]
    # The model that judges which Tier a request needs; None when only one Tier is enabled.
    analyzer: str | None = None
    # Kept beside the Policy so a call pins the Profile inputs it started with. Both are None
    # when nothing was declared, and routing then reads the empty Profile.
    # Left out of the repr, so a traceback or log line never echoes the app's own text.
    sources: ProfileSources | None = field(default=None, repr=False)
    fingerprint: str | None = None
