from __future__ import annotations

from collections.abc import Iterable

from primagate.core.prices import Price
from primagate.schema.errors import BadRequestError
from primagate.schema.tiers import FEATURES, TierBinding
from primagate.schema.wire import Capabilities


def effective_features(
    declared: Capabilities,
    binding: TierBinding | None = None,
    registry: frozenset[str] | None = None,
) -> frozenset[str]:
    """What a model can do: the adapter's declaration, narrowed and never widened.

    The Caller's binding limits are the Caller's own word about their model, so when they are
    known they replace the registry's answer as the narrowing set; the registry only fills a gap
    they leave. Either way the result stays inside what the adapter implements.
    """
    implemented = frozenset(name for name in FEATURES if getattr(declared, name))
    narrowing = registry
    if binding is not None and isinstance(binding.limits.supported_features, frozenset):
        narrowing = binding.limits.supported_features
    return implemented if narrowing is None else implemented & narrowing


def binding_prices(bindings: Iterable[TierBinding]) -> dict[tuple[str, str], Price]:
    """Prices the Caller's bindings give their models; a half-known price gives none.

    Raises when one model is priced two ways, since which Tier's price applies would be a guess.
    """
    prices: dict[tuple[str, str], Price] = {}
    for binding in bindings:
        low, high = binding.pricing.input, binding.pricing.output
        if isinstance(low, float | int) and isinstance(high, float | int):
            key, price = (binding.provider, binding.model_id), Price(float(low), float(high))
            if prices.setdefault(key, price) != price:
                raise BadRequestError(
                    f"tier_bindings give model '{binding.model_id}' two different prices"
                )
    return prices
