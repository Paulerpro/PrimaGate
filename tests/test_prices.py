import math
import re
from pathlib import Path

import pytest

from primagate.core import prices
from primagate.core.prices import BUNDLED, Price, estimate, lookup
from primagate.drivers.providers import parse_price_overrides
from primagate.schema.errors import BadRequestError
from primagate.schema.types import Usage

GPT4O, MINI = ("openai", "gpt-4o"), ("openai", "gpt-4o-mini")
USAGE = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)


def test_a_known_model_is_priced_per_million_tokens() -> None:
    result = estimate("openai", "gpt-4o", USAGE, BUNDLED)
    assert result.unknown_model is False
    assert result.token_counts is not None
    assert (result.token_counts.prompt_tokens, result.token_counts.total_tokens) == (1000, 1500)
    assert result.token_counts.heuristic is False
    cost = result.cost
    assert cost is not None and cost.currency == "USD"
    assert cost.input == pytest.approx(0.0025) and cost.output == pytest.approx(0.005)
    assert cost.total == pytest.approx(0.0075)


@pytest.mark.parametrize("name", ["gpt-4o-2024-08-06", "gpt-4o-20240806"])
def test_a_dated_snapshot_is_priced_as_its_base_model(name: str) -> None:
    assert lookup(BUNDLED, "openai", name) == BUNDLED[GPT4O]


def test_a_near_name_is_never_priced_as_a_shorter_one() -> None:
    assert lookup(BUNDLED, "openai", "gpt-4o-mini") == BUNDLED[MINI] != BUNDLED[GPT4O]
    assert lookup(BUNDLED, "openai", "gpt-4o-mini-2024-07-18") == BUNDLED[MINI]
    assert lookup(BUNDLED, "openai", "gpt-4o-turbo") is None
    assert lookup(BUNDLED, "openai", "gpt-4o-extra-2024-08-06") is None


def test_an_exact_snapshot_entry_beats_its_base_and_an_override_of_the_base_covers_it() -> None:
    exact = Price(1.0, 2.0)
    table = {**BUNDLED, ("openai", "gpt-4o-2024-08-06"): exact}
    assert lookup(table, "openai", "gpt-4o-2024-08-06") == exact
    override = {**BUNDLED, GPT4O: Price(9.0, 9.0)}
    assert lookup(override, "openai", "gpt-4o-2024-11-20") == Price(9.0, 9.0)


def test_an_unknown_model_has_no_cost_but_still_has_counts() -> None:
    result = estimate("openai", "mystery-1", USAGE, BUNDLED)
    assert result.unknown_model is True and result.cost is None
    assert result.token_counts is not None and result.token_counts.total_tokens == 1500


def test_a_response_without_usage_gets_neither_counts_nor_cost() -> None:
    known = estimate("openai", "gpt-4o", None, BUNDLED)
    assert (known.token_counts, known.cost, known.unknown_model) == (None, None, False)
    unknown = estimate("openai", "mystery-1", None, BUNDLED)
    assert unknown.unknown_model is True


def test_overrides_are_validated_and_resolved_like_model_names() -> None:
    parsed = parse_price_overrides(
        {"gpt-4o": {"input": 1, "output": 2.5}, "anthropic/x": {"input": 0, "output": 0}}
    )
    assert parsed[GPT4O] == Price(1.0, 2.5)
    assert parsed[("anthropic", "x")] == Price(0.0, 0.0)
    assert parse_price_overrides(None) == {}


@pytest.mark.parametrize(
    "bad",
    [
        "gpt-4o",
        {"gpt-4o": 3},
        {"gpt-4o": {"input": 1}},
        {"gpt-4o": {"input": 1, "output": 2, "extra": 3}},
        {"gpt-4o": {"input": -1, "output": 2}},
        {"gpt-4o": {"input": math.nan, "output": 2}},
        {"gpt-4o": {"input": math.inf, "output": 2}},
        {"gpt-4o": {"input": True, "output": 2}},
        {"gpt-4o": {"input": "1", "output": 2}},
        {"nope/gpt-4o": {"input": 1, "output": 2}},
        {3: {"input": 1, "output": 2}},
        {
            "gpt-4o": {"input": 1, "output": 2},
            "openai/gpt-4o": {"input": 3, "output": 4},
        },
    ],
)
def test_a_bad_override_is_refused_at_setup(bad: object) -> None:
    with pytest.raises(BadRequestError, match="price_overrides"):
        parse_price_overrides(bad)


def test_the_bundled_map_is_sane_and_documented() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", prices.PRICES_AS_OF)
    assert BUNDLED
    for (provider, model), price in BUNDLED.items():
        assert provider in {"openai", "anthropic"} and model
        assert price.input > 0 and price.output > 0
        assert not re.search(r"-(\d{4}-\d{2}-\d{2}|\d{8})$", model), model
    notes = (Path(__file__).parents[1] / "docs" / "price-map.md").read_text(encoding="utf-8")
    assert f"**{prices.PRICES_AS_OF}**" in notes
    documented = {
        name: (float(spent_in), float(spent_out))
        for name, spent_in, spent_out in re.findall(
            r"^\| `([^`]+)` \| ([\d.]+) \| ([\d.]+) \|$", notes, flags=re.MULTILINE
        )
    }
    assert documented == {m: (p.input, p.output) for (_, m), p in BUNDLED.items()}
