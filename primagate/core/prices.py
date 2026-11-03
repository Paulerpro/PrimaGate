from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from primagate.schema.types import Cost, TokenCounts, Usage

PRICES_AS_OF = "2026-11-03"

_PER_MILLION = 1_000_000
# A dated snapshot of a model, which the provider prices like the model itself.
_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float


PriceTable = Mapping[tuple[str, str], Price]

# Only prices read plainly from the providers' pages are listed; see docs/price-map.md. A model
# that is not here reports an unknown cost rather than a guess.
BUNDLED: PriceTable = MappingProxyType(
    {
        ("openai", "gpt-4o"): Price(2.50, 10.00),
        ("openai", "gpt-4o-mini"): Price(0.15, 0.60),
        ("openai", "gpt-4.1"): Price(2.00, 8.00),
        ("openai", "gpt-4.1-mini"): Price(0.40, 1.60),
        ("anthropic", "claude-fable-5-1"): Price(10.00, 50.00),
        ("anthropic", "claude-opus-5-5"): Price(4.00, 20.00),
        ("anthropic", "claude-sonnet-5-5"): Price(2.00, 10.00),
    }
)


@dataclass(frozen=True)
class Estimate:
    token_counts: TokenCounts | None
    cost: Cost | None
    unknown_model: bool


def lookup(table: PriceTable, provider: str, model: str) -> Price | None:
    """The price of `model`: its own entry first, then its base model if it is a dated snapshot.

    Nothing else is matched by prefix, so `gpt-4o-mini` can never be priced as `gpt-4o`.
    """
    exact = table.get((provider, model))
    if exact is not None:
        return exact
    base = _SNAPSHOT.sub("", model)
    return table.get((provider, base)) if base != model else None


def estimate(provider: str, model: str, usage: Usage | None, table: PriceTable) -> Estimate:
    """Token counts and cost for one answer. Information only: it never raises."""
    price = lookup(table, provider, model)
    unknown = price is None
    if usage is None:
        return Estimate(None, None, unknown)
    counts = TokenCounts(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    if price is None:
        return Estimate(counts, None, True)
    spent_in = usage.prompt_tokens * price.input / _PER_MILLION
    spent_out = usage.completion_tokens * price.output / _PER_MILLION
    return Estimate(counts, Cost(spent_in, spent_out, spent_in + spent_out), False)
