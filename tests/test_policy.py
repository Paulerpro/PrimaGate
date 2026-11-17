import logging
from typing import Any

import pytest

import primagate
from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.policy import Policy
from primagate.schema.tiers import Tier

BINDINGS: dict[str, Any] = {
    "SMALL": {"model": "gpt-4o-mini"},
    "MEDIUM": {"model": "claude-sonnet-5-5"},
    "POWERFUL": {"model": "gpt-4o"},
}
REQUIRED: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"],
    "confidence_threshold": 0.8,
    "context_token_budget": 2000,
    "interim_binding": "MEDIUM",
}
FULL: dict[str, Any] = {
    **REQUIRED,
    "cost_preference": "balanced",
    "ambiguous_context_default": "SMALL",
    "max_concurrent_analyzer_calls": 4,
    "spending_ceiling": 25.5,
}


def _gate(policy: Any = REQUIRED, bindings: Any = BINDINGS) -> PrimaGate:
    analyzer = "gpt-4o-mini" if policy is not None else None
    return PrimaGate(tier_bindings=bindings, policy=policy, analyzer=analyzer)


def _policy(policy: Any = REQUIRED, bindings: Any = BINDINGS) -> Policy:
    stored = _gate(policy, bindings)._policy
    assert stored is not None
    return stored


def test_a_full_policy_stores_every_field_with_typed_values() -> None:
    policy = _policy(FULL)
    assert policy.enabled_tiers == (Tier.SMALL, Tier.MEDIUM, Tier.POWERFUL)
    assert policy.cost_preference == "balanced"
    assert policy.ambiguous_context_default is Tier.SMALL
    assert policy.confidence_threshold == 0.8 and policy.context_token_budget == 2000
    assert policy.max_concurrent_analyzer_calls == 4
    assert policy.interim_binding is Tier.MEDIUM
    assert policy.spending_ceiling == 25.5
    assert len(policy.version) >= 8


def test_the_minimal_policy_takes_the_documented_defaults() -> None:
    policy = _policy()
    assert policy.cost_preference == "balanced"
    assert policy.ambiguous_context_default is Tier.MEDIUM
    assert policy.max_concurrent_analyzer_calls is None
    assert policy.spending_ceiling is None


def test_enabled_tiers_are_stored_in_tier_order_whatever_order_they_were_given() -> None:
    policy = _policy({**REQUIRED, "enabled_tiers": ["POWERFUL", "MEDIUM"]})
    assert policy.enabled_tiers == (Tier.MEDIUM, Tier.POWERFUL)


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_a_policy_without_a_required_field_is_refused_naming_it(missing: str) -> None:
    policy = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(errors.BadRequestError, match=missing):
        _gate(policy)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled_tiers", []),
        ("enabled_tiers", "SMALL"),
        ("enabled_tiers", None),
        ("enabled_tiers", ["SMALL", "SMALL"]),
        ("enabled_tiers", ["LARGE"]),
        ("enabled_tiers", ["small"]),
        ("enabled_tiers", [1]),
        ("enabled_tiers", [Tier.SMALL]),
        ("confidence_threshold", 0),
        ("confidence_threshold", 1.01),
        ("confidence_threshold", -0.5),
        ("confidence_threshold", True),
        ("confidence_threshold", float("nan")),
        ("confidence_threshold", float("inf")),
        ("confidence_threshold", "0.8"),
        ("confidence_threshold", None),
        ("confidence_threshold", 10**400),
        ("context_token_budget", 0),
        ("context_token_budget", -1),
        ("context_token_budget", 2.5),
        ("context_token_budget", True),
        ("context_token_budget", None),
        ("interim_binding", "default"),
        ("interim_binding", "LARGE"),
        ("interim_binding", None),
        ("interim_binding", Tier.MEDIUM),
        ("cost_preference", "cheap"),
        ("cost_preference", None),
        ("cost_preference", True),
        ("ambiguous_context_default", "LARGE"),
        ("ambiguous_context_default", None),
        ("max_concurrent_analyzer_calls", 0),
        ("max_concurrent_analyzer_calls", 1.5),
        ("max_concurrent_analyzer_calls", True),
        ("max_concurrent_analyzer_calls", None),
        ("spending_ceiling", 0),
        ("spending_ceiling", -1),
        ("spending_ceiling", float("nan")),
        ("spending_ceiling", True),
        ("spending_ceiling", None),
        ("spending_ceiling", 10**400),
    ],
)
def test_a_bad_value_is_refused_naming_the_field(field: str, value: Any) -> None:
    with pytest.raises(errors.BadRequestError, match=field):
        _gate({**REQUIRED, field: value})


@pytest.mark.parametrize("policy", [[], "policy", [("enabled_tiers", ["SMALL"])], 5])
def test_the_declaration_must_be_a_mapping(policy: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        _gate(policy)


def test_an_unknown_key_is_refused_without_echoing_it() -> None:
    secret = "sk-very-secret-key"
    with pytest.raises(errors.BadRequestError) as caught:
        _gate({**REQUIRED, secret: 1})
    assert secret not in str(caught.value)


def test_a_refusal_never_echoes_a_value() -> None:
    secret = "sk-very-secret-value"
    for field in ("confidence_threshold", "cost_preference", "interim_binding"):
        with pytest.raises(errors.BadRequestError) as caught:
            _gate({**REQUIRED, field: secret})
        assert secret not in str(caught.value) and field in str(caught.value)


def test_an_enabled_tier_without_a_binding_is_refused_naming_the_tier() -> None:
    with pytest.raises(errors.BadRequestError, match="POWERFUL"):
        _gate(REQUIRED, {"SMALL": BINDINGS["SMALL"], "MEDIUM": BINDINGS["MEDIUM"]})


def test_a_policy_with_no_bindings_at_all_is_refused() -> None:
    with pytest.raises(errors.BadRequestError, match="SMALL"):
        PrimaGate(policy={**REQUIRED, "enabled_tiers": ["SMALL"], "interim_binding": "SMALL"})


def test_a_bound_tier_that_is_not_enabled_is_fine() -> None:
    policy = _policy({**REQUIRED, "enabled_tiers": ["MEDIUM"]})
    assert policy.enabled_tiers == (Tier.MEDIUM,)


@pytest.mark.parametrize("field", ["interim_binding", "ambiguous_context_default"])
def test_a_default_that_names_a_tier_that_is_not_enabled_is_refused(field: str) -> None:
    only_small = {**REQUIRED, "enabled_tiers": ["SMALL"], "interim_binding": "SMALL"}
    only_small["ambiguous_context_default"] = "SMALL"
    _policy(only_small)
    with pytest.raises(errors.BadRequestError, match=field):
        _gate({**only_small, field: "MEDIUM"})


def test_leaving_the_ambiguous_default_out_needs_medium_to_be_enabled() -> None:
    no_medium = {
        **REQUIRED,
        "enabled_tiers": ["SMALL", "POWERFUL"],
        "interim_binding": "SMALL",
    }
    with pytest.raises(errors.BadRequestError, match="ambiguous_context_default"):
        _gate(no_medium)
    assert _policy({**no_medium, "ambiguous_context_default": "POWERFUL"}).enabled_tiers


@pytest.mark.parametrize("preference", ["cost_conscious", "capability_first"])
def test_a_non_default_preference_logs_one_warning_and_setup_succeeds(
    preference: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="primagate"):
        policy = _policy({**REQUIRED, "cost_preference": preference})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert policy.cost_preference == preference
    assert len(warnings) == 1 and preference in warnings[0].getMessage()


def test_balanced_or_unset_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        _policy()
        _policy({**REQUIRED, "cost_preference": "balanced"})
    assert not caplog.records


def test_equal_policies_share_a_version_and_any_change_gives_another() -> None:
    first = _policy(FULL).version
    assert _policy({**FULL, "enabled_tiers": ["POWERFUL", "MEDIUM", "SMALL"]}).version == first
    changed = [
        {**FULL, "confidence_threshold": 0.7},
        {**FULL, "context_token_budget": 2001},
        {**FULL, "cost_preference": "capability_first"},
        {**FULL, "ambiguous_context_default": "MEDIUM"},
        {**FULL, "max_concurrent_analyzer_calls": 5},
        {**FULL, "spending_ceiling": 26},
        {**FULL, "interim_binding": "SMALL"},
        {**FULL, "enabled_tiers": ["SMALL", "MEDIUM"]},
    ]
    versions = {_policy(p).version for p in changed}
    assert first not in versions and len(versions) == len(changed)


def test_the_stored_policy_cannot_be_changed() -> None:
    gate = _gate()
    policy = gate._policy
    assert policy is not None
    with pytest.raises(AttributeError):
        policy.confidence_threshold = 0.1  # type: ignore[misc]


def test_a_stored_policy_ignores_later_edits_to_the_input() -> None:
    declared = {**REQUIRED, "enabled_tiers": ["SMALL", "MEDIUM"]}
    policy = _policy(declared)
    declared["enabled_tiers"].append("POWERFUL")
    declared["confidence_threshold"] = 0.1
    assert policy.enabled_tiers == (Tier.SMALL, Tier.MEDIUM) and policy.confidence_threshold == 0.8


def test_nothing_declared_leaves_the_gate_as_it_was() -> None:
    assert PrimaGate()._policy is None
    assert PrimaGate(policy=None)._policy is None
    assert PrimaGate(tier_bindings=BINDINGS)._policy is None


def test_any_mapping_is_accepted_as_a_policy() -> None:
    from types import MappingProxyType

    assert _policy(MappingProxyType(REQUIRED)).interim_binding is Tier.MEDIUM


def test_the_version_format_is_pinned() -> None:
    # A golden value: a change here means every recorded version silently stopped matching.
    assert _policy().version == "e3e4c6c554dd"


def test_an_unknown_key_refusal_lists_the_valid_fields() -> None:
    with pytest.raises(errors.BadRequestError, match="enabled_tiers"):
        _gate({**REQUIRED, "confidence_treshold": 0.5})


def test_the_policy_adds_no_public_name() -> None:
    assert not hasattr(primagate, "Policy")
