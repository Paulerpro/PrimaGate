from typing import Any

import pytest

import primagate.core.fallbacks as fallbacks_module
from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.wire import NO_FALLBACKS

A = ("openai", "gpt-4o")
B = ("anthropic", "claude-sonnet-5-5")
C = ("openai", "gpt-4o-mini")


def _stored(**kwargs: Any):  # type: ignore[no-untyped-def]
    return PrimaGate(**kwargs)._fallbacks


def test_names_resolve_to_provider_and_model_and_keep_their_order() -> None:
    stored = _stored(
        fallbacks={"gpt-4o": ["claude-sonnet-5-5", "openai/gpt-4o-mini"]},
        default_fallbacks=["gpt-4o-mini"],
    )
    assert stored.per_model == {A: (B, C)}
    assert stored.default == (C,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fallbacks": None, "default_fallbacks": None},
        {"fallbacks": {}, "default_fallbacks": []},
    ],
)
def test_no_list_declared_is_the_same_empty_list(kwargs: dict[str, Any]) -> None:
    assert _stored(**kwargs) == NO_FALLBACKS


def test_an_explicit_empty_entry_is_kept_apart_from_the_default() -> None:
    stored = _stored(fallbacks={"gpt-4o": []}, default_fallbacks=["gpt-4o-mini"])
    assert stored.per_model == {A: ()} and stored.default == (C,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fallbacks": {"gpt-4o": "claude-sonnet-5-5"}},
        {"fallbacks": {"gpt-4o": None}},
        {"fallbacks": {"gpt-4o": [5]}},
        {"fallbacks": {"gpt-4o": [""]}},
        {"fallbacks": {5: ["gpt-4o-mini"]}},
        {"fallbacks": {"": ["gpt-4o-mini"]}},
        {"fallbacks": {"gpt-4o": ["  "]}},
        {"fallbacks": {"gpt-4o": ["openai/"]}},
        {"default_fallbacks": ["anthropic/"]},
        {"fallbacks": ["gpt-4o"]},
        {"fallbacks": "gpt-4o"},
        {"default_fallbacks": "gpt-4o-mini"},
        {"default_fallbacks": [None]},
        {"default_fallbacks": 3},
    ],
)
def test_a_malformed_declaration_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(errors.BadRequestError):
        _stored(**kwargs)


def test_an_unknown_provider_prefix_is_rejected_wherever_it_appears() -> None:
    with pytest.raises(errors.BadRequestError, match="nope"):
        _stored(fallbacks={"nope/x": ["gpt-4o"]})
    with pytest.raises(errors.BadRequestError, match="nope"):
        _stored(fallbacks={"gpt-4o": ["nope/x"]})
    with pytest.raises(errors.BadRequestError, match="nope"):
        _stored(default_fallbacks=["nope/x"])


def test_two_keys_naming_the_same_model_are_rejected() -> None:
    with pytest.raises(errors.BadRequestError, match="gpt-4o"):
        _stored(fallbacks={"gpt-4o": ["gpt-4o-mini"], "openai/gpt-4o": ["gpt-4.1"]})


def test_a_reserved_name_is_rejected_in_entries_and_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fallbacks_module, "RESERVED_MODEL_NAMES", frozenset({"tiers/auto"}))
    with pytest.raises(errors.BadRequestError, match="tiers/auto"):
        _stored(fallbacks={"gpt-4o": ["gpt-4o-mini", "tiers/auto"]})
    with pytest.raises(errors.BadRequestError, match="tiers/auto"):
        _stored(default_fallbacks=["tiers/auto"])


def test_the_stored_declaration_cannot_be_changed_afterwards() -> None:
    stored = _stored(fallbacks={"gpt-4o": ["gpt-4o-mini"]})
    with pytest.raises(TypeError):
        stored.per_model[B] = ()  # type: ignore[index]


def test_changing_the_callers_lists_afterwards_has_no_effect() -> None:
    entries = ["gpt-4o-mini"]
    declared = {"gpt-4o": entries}
    stored = _stored(fallbacks=declared, default_fallbacks=entries)
    entries.append("claude-sonnet-5-5")
    declared["claude-sonnet-5-5"] = []
    assert stored.per_model == {A: (C,)} and stored.default == (C,)
