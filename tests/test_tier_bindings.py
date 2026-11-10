import copy
import pickle
from typing import Any

import pytest

import primagate
from primagate import PrimaGate
from primagate.core import tiers as core_tiers
from primagate.core.prices import Price
from primagate.schema import errors
from primagate.schema.tiers import UNKNOWN, Tier, TierBinding, Unknown
from primagate.schema.wire import Capabilities

FULL: dict[str, Any] = {
    "model": "openai/gpt-4o-mini",
    "settings": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 512, "seed": 7, "stop": ["END"]},
    "limits": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supported_features": ["tools", "streaming"],
    },
    "pricing": {"input": 0.15, "output": 0.6},
}


def _bound(**bindings: Any) -> dict[Tier, TierBinding]:
    return dict(PrimaGate(tier_bindings=bindings)._tier_bindings)


def test_a_full_binding_stores_every_part_with_typed_values() -> None:
    binding = _bound(SMALL=FULL)[Tier.SMALL]
    assert (binding.provider, binding.model_id) == ("openai", "gpt-4o-mini")
    assert binding.settings["temperature"] == 0.2 and binding.settings["stop"] == ("END",)
    assert binding.limits.context_window == 128000
    assert binding.limits.supported_features == frozenset({"tools", "streaming"})
    assert (binding.pricing.input, binding.pricing.output) == (0.15, 0.6)


def test_a_minimal_binding_holds_unknown_for_every_undisclosed_field() -> None:
    binding = _bound(MEDIUM={"model": "gpt-4o"})[Tier.MEDIUM]
    assert binding.limits.context_window is UNKNOWN
    assert binding.limits.max_output_tokens is UNKNOWN
    assert binding.limits.supported_features is UNKNOWN
    assert binding.pricing.input is UNKNOWN and binding.pricing.output is UNKNOWN
    assert dict(binding.settings) == {}


def test_the_word_unknown_is_the_same_as_leaving_the_field_out() -> None:
    said = {
        "model": "gpt-4o",
        "limits": {"context_window": "unknown"},
        "pricing": {"input": "unknown"},
    }
    binding = _bound(MEDIUM=said)[Tier.MEDIUM]
    assert binding.limits.context_window is UNKNOWN and binding.pricing.input is UNKNOWN


def test_unknown_is_one_value_that_survives_copying() -> None:
    assert Unknown() is UNKNOWN
    assert copy.deepcopy(UNKNOWN) is UNKNOWN and pickle.loads(pickle.dumps(UNKNOWN)) is UNKNOWN
    assert primagate.UNKNOWN is UNKNOWN and primagate.Unknown is Unknown
    assert not hasattr(primagate, "TierBinding")


def test_to_dict_shows_unknown_as_the_word_and_omits_nothing() -> None:
    shown = _bound(MEDIUM={"model": "claude-sonnet-5-5"})[Tier.MEDIUM].to_dict()
    assert shown["provider"] == "anthropic" and shown["model_id"] == "claude-sonnet-5-5"
    assert shown["limits"] == {
        "context_window": "unknown",
        "max_output_tokens": "unknown",
        "supported_features": "unknown",
    }
    assert shown["pricing"] == {"input": "unknown", "output": "unknown"}
    assert shown["tier"] == "MEDIUM"


def test_to_dict_shows_known_values_and_the_sorted_feature_list() -> None:
    shown = _bound(SMALL=FULL)[Tier.SMALL].to_dict()
    assert shown["settings"] == {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": 7,
        "stop": ["END"],
    }
    assert shown["limits"]["supported_features"] == ["streaming", "tools"]
    assert shown["limits"]["context_window"] == 128000
    assert shown["pricing"] == {"input": 0.15, "output": 0.6}


def test_unknown_is_not_falsy_so_it_is_never_taken_for_zero_or_absent() -> None:
    assert bool(UNKNOWN) is True


def test_tiers_are_ordered_and_only_those_three_exist() -> None:
    assert Tier.SMALL < Tier.MEDIUM < Tier.POWERFUL
    assert [t.name for t in Tier] == ["SMALL", "MEDIUM", "POWERFUL"]
    bound = _bound(POWERFUL={"model": "gpt-4o"}, SMALL={"model": "gpt-4o-mini"})
    assert sorted(bound) == [Tier.SMALL, Tier.POWERFUL]


@pytest.mark.parametrize("label", ["LARGE", "small", "Small", "", " SMALL", 1, None, Tier.SMALL])
def test_any_other_label_is_refused_at_setup(label: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        PrimaGate(tier_bindings={label: {"model": "gpt-4o"}})


@pytest.mark.parametrize(
    "binding",
    [
        None,
        "gpt-4o",
        [],
        {},
        {"model": ""},
        {"model": "  "},
        {"model": " gpt-4o"},
        {"model": "openai/gpt-4o "},
        {"model": "openai/"},
        {"model": 5},
        {"model": "nope/gpt-4o"},
        {"model": "gpt-4o", "extra": 1},
        {"model": "gpt-4o", "settings": []},
        {"model": "gpt-4o", "settings": None},
        {"model": "gpt-4o", "limits": None},
        {"model": "gpt-4o", "pricing": None},
        {"model": "gpt-4o", "settings": {"temperature": 10**400}},
        {"model": "gpt-4o", "pricing": {"input": 10**400}},
        {"model": "gpt-4o", "settings": {"colour": "red"}},
        {"model": "gpt-4o", "settings": {"temperature": True}},
        {"model": "gpt-4o", "settings": {"temperature": float("nan")}},
        {"model": "gpt-4o", "settings": {"temperature": -1}},
        {"model": "gpt-4o", "settings": {"max_tokens": 0}},
        {"model": "gpt-4o", "settings": {"max_tokens": 1.5}},
        {"model": "gpt-4o", "settings": {"seed": True}},
        {"model": "gpt-4o", "settings": {"stop": 5}},
        {"model": "gpt-4o", "settings": {"stop": [5]}},
        {"model": "gpt-4o", "settings": {"temperature": None}},
        {"model": "gpt-4o", "limits": []},
        {"model": "gpt-4o", "limits": {"window": 1}},
        {"model": "gpt-4o", "limits": {"context_window": None}},
        {"model": "gpt-4o", "limits": {"context_window": True}},
        {"model": "gpt-4o", "limits": {"context_window": 0}},
        {"model": "gpt-4o", "limits": {"context_window": -5}},
        {"model": "gpt-4o", "limits": {"max_output_tokens": 2.5}},
        {"model": "gpt-4o", "limits": {"supported_features": "tools"}},
        {"model": "gpt-4o", "limits": {"supported_features": ["telepathy"]}},
        {"model": "gpt-4o", "limits": {"supported_features": [5]}},
        {"model": "gpt-4o", "limits": {"supported_features": None}},
        {"model": "gpt-4o", "pricing": []},
        {"model": "gpt-4o", "pricing": {"cost": 1}},
        {"model": "gpt-4o", "pricing": {"input": None}},
        {"model": "gpt-4o", "pricing": {"input": True}},
        {"model": "gpt-4o", "pricing": {"input": -1}},
        {"model": "gpt-4o", "pricing": {"output": float("inf")}},
        {"model": "gpt-4o", "pricing": {"output": "free"}},
    ],
)
def test_a_bad_binding_is_refused_at_setup(binding: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        PrimaGate(tier_bindings={"SMALL": binding})


@pytest.mark.parametrize("value", [[], "SMALL", [("SMALL", {"model": "gpt-4o"})], 5])
def test_the_declaration_itself_must_be_a_mapping(value: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        PrimaGate(tier_bindings=value)


def test_a_refusal_names_the_tier_and_the_field_but_never_a_value() -> None:
    secret = "sk-very-secret-value"
    with pytest.raises(errors.BadRequestError) as caught:
        PrimaGate(tier_bindings={"MEDIUM": {"model": "gpt-4o", "settings": {"seed": secret}}})
    text = str(caught.value)
    assert "MEDIUM" in text and "seed" in text and secret not in text
    with pytest.raises(errors.BadRequestError) as caught:
        PrimaGate(tier_bindings={"MEDIUM": {"model": "gpt-4o", secret: 1}})
    assert secret not in str(caught.value)


def test_nothing_declared_leaves_the_gate_as_it_was() -> None:
    assert PrimaGate()._tier_bindings == {}
    assert PrimaGate(tier_bindings=None)._tier_bindings == {}
    assert PrimaGate(tier_bindings={})._tier_bindings == {}


def test_a_stored_binding_cannot_be_changed_and_ignores_later_edits_to_the_input() -> None:
    declared = {"SMALL": {"model": "gpt-4o", "settings": {"temperature": 0.1}}}
    binding = _bound(**declared)[Tier.SMALL]
    declared["SMALL"]["settings"]["temperature"] = 0.9  # type: ignore[index]
    assert binding.settings["temperature"] == 0.1
    with pytest.raises(AttributeError):
        binding.model_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        binding.settings["temperature"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        PrimaGate(tier_bindings=declared)._tier_bindings[Tier.MEDIUM] = binding  # type: ignore[index]


def test_the_same_model_may_serve_two_tiers() -> None:
    bound = _bound(SMALL={"model": "gpt-4o"}, MEDIUM={"model": "gpt-4o"})
    assert bound[Tier.SMALL].model_id == bound[Tier.MEDIUM].model_id


def test_one_model_with_two_different_prices_is_refused() -> None:
    with pytest.raises(errors.BadRequestError):
        PrimaGate(
            tier_bindings={
                "SMALL": {"model": "gpt-4o", "pricing": {"input": 1, "output": 2}},
                "MEDIUM": {"model": "openai/gpt-4o", "pricing": {"input": 1, "output": 3}},
            }
        )


def test_one_model_with_the_same_price_or_one_unpriced_is_fine() -> None:
    price = {"input": 1, "output": 2}
    _bound(
        SMALL={"model": "gpt-4o", "pricing": price}, MEDIUM={"model": "gpt-4o", "pricing": price}
    )
    _bound(SMALL={"model": "gpt-4o", "pricing": price}, MEDIUM={"model": "gpt-4o"})


def test_only_a_fully_known_price_is_offered() -> None:
    bound = _bound(
        SMALL={"model": "gpt-4o-mini", "pricing": {"input": 1}},
        MEDIUM={"model": "gpt-4o", "pricing": {"input": 2.0, "output": 3.0}},
        POWERFUL={"model": "claude-opus-5-5"},
    )
    assert core_tiers.binding_prices(bound.values()) == {("openai", "gpt-4o"): Price(2.0, 3.0)}


ADAPTER = Capabilities(tools=True, structured_output=True, images=True, streaming=True)
ALL = frozenset({"tools", "structured_output", "images", "streaming"})


def _binding(**limits: Any) -> TierBinding:
    return _bound(SMALL={"model": "gpt-4o", "limits": limits})[Tier.SMALL]


def test_known_limits_narrow_what_the_adapter_declares() -> None:
    narrowed = core_tiers.effective_features(ADAPTER, _binding(supported_features=["streaming"]))
    assert narrowed == frozenset({"streaming"})


def test_limits_cannot_grant_what_the_adapter_lacks() -> None:
    weak = Capabilities(streaming=True)
    got = core_tiers.effective_features(weak, _binding(supported_features=["tools", "streaming"]))
    assert got == frozenset({"streaming"})


def test_unknown_limits_leave_the_adapter_declaration_alone() -> None:
    assert core_tiers.effective_features(ADAPTER, _binding()) == ALL
    assert core_tiers.effective_features(ADAPTER, None) == ALL


def test_an_empty_feature_list_is_a_known_limit_of_nothing() -> None:
    assert core_tiers.effective_features(ADAPTER, _binding(supported_features=[])) == frozenset()


def test_binding_limits_win_over_the_registry_and_the_registry_fills_a_gap() -> None:
    registry = frozenset({"tools"})
    won = core_tiers.effective_features(
        ADAPTER, _binding(supported_features=["streaming", "images"]), registry
    )
    assert won == frozenset({"streaming", "images"})
    assert core_tiers.effective_features(ADAPTER, _binding(), registry) == registry
    assert core_tiers.effective_features(ADAPTER, None, registry) == registry
