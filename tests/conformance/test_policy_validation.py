import logging
from typing import Any

import pytest
from conftest import Caller
from test_tier_routing import ALIAS, MSGS, TWO, Run

from primagate import PrimaGate
from primagate.schema import errors, events

pytestmark = pytest.mark.fr("FR21")

# Only the fields a Policy cannot do without; everything else takes its documented default.
REQUIRED: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "SMALL",
}


@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_a_policy_without_a_required_field_is_refused_at_setup(missing: str) -> None:
    policy = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(errors.BadRequestError, match=missing):
        PrimaGate(tier_bindings=TWO, policy=policy, analyzer="gpt-4o-mini")


def test_a_minimal_policy_sends_a_low_confidence_call_to_medium(call: Caller) -> None:
    run = Run(["SMALL 0.3"], policy=REQUIRED)
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert [w["model"] for w in run.workers] == ["gpt-4o"]
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert len(run.of(events.LowConfidenceDowngrade)) == 1


def _advice(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "cost_preference" in r.getMessage()
    ]


@pytest.mark.parametrize("preference", ["cost_conscious", "capability_first"])
def test_a_non_balanced_preference_only_advises_and_the_call_still_routes(
    call: Caller, preference: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="primagate"):
        run = Run(["SMALL 0.9"], policy={**REQUIRED, "cost_preference": preference})
    advice = _advice(caplog)
    assert len(advice) == 1 and preference in advice[0]
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert [w["model"] for w in run.workers] == ["gpt-4o-mini"]
    assert result.metadata is not None and result.metadata.tier == "SMALL"


@pytest.mark.parametrize("declared", [{}, {"cost_preference": "balanced"}], ids=["unset", "set"])
def test_balanced_is_the_default_and_gives_no_advisory(
    call: Caller, declared: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG, logger="primagate"):
        run = Run(["SMALL 0.9"], policy={**REQUIRED, **declared})
    assert _advice(caplog) == []
    assert run.gate._policy is not None and run.gate._policy.cost_preference == "balanced"
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert result.metadata is not None and result.metadata.tier == "SMALL"


@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
@pytest.mark.parametrize("preference", ["cheap", "", None, True])
def test_a_cost_preference_outside_the_three_is_refused_at_setup(preference: object) -> None:
    policy = {**REQUIRED, "cost_preference": preference}
    with pytest.raises(errors.BadRequestError, match="cost_preference"):
        PrimaGate(tier_bindings=TWO, policy=policy, analyzer="gpt-4o-mini")
