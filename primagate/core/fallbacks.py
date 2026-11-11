from __future__ import annotations

from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.wire import Fallbacks

# Names that mean Tier Routing rather than a concrete model; a fallback may not name one, so a
# fallback can never re-enter Tier Routing.
RESERVED_MODEL_NAMES: frozenset[str] = frozenset({TIER_ROUTING_ALIAS})


def fallback_candidates(
    primary: tuple[str, str], fallbacks: Fallbacks
) -> tuple[tuple[str, str], ...]:
    """The models to try after `primary`, in the order the Caller declared them.

    A per-model list, even an empty one, replaces the default. The primary and repeats are
    dropped so a model appears at most once in a chain.
    """
    chosen = fallbacks.per_model.get(primary, fallbacks.default)
    ordered: dict[tuple[str, str], None] = {}
    for model in chosen:
        if model != primary:
            ordered.setdefault(model)
    return tuple(ordered)
