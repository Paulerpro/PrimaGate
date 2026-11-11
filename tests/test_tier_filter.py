from typing import Any

import pytest

import primagate
from primagate.core import fallbacks as core_fallbacks
from primagate.core import tiers as core_tiers
from primagate.core.call import first_action
from primagate.core.policy import build_policy
from primagate.core.steps import Dispatch, DropPolicy, Raise
from primagate.drivers.providers import parse_tier_bindings
from primagate.schema import errors
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.tiers import Tier
from primagate.schema.types import ChatCompletionRequest, Message
from primagate.schema.wire import Capabilities

OPENAI_CAPS = Capabilities(tools=True, structured_output=True, images=True, streaming=True)
CLAUDE_CAPS = Capabilities(tools=True, structured_output=True, streaming=True)
CAPS = {"openai": OPENAI_CAPS, "anthropic": CLAUDE_CAPS}
IMAGE = [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _routing(bindings: dict[str, Any], enabled: list[str] | None = None) -> TierRouting:
    parsed = parse_tier_bindings(bindings)
    names = enabled or [t.name for t in parsed]
    policy, _ = build_policy(
        {
            "enabled_tiers": names,
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            "interim_binding": names[0],
            "ambiguous_context_default": names[0],
        },
        parsed,
    )
    return TierRouting(policy, parsed)


def _request(model: str = TIER_ROUTING_ALIAS, **fields: Any) -> ChatCompletionRequest:
    messages = fields.pop("messages", [Message(role="user", content="hi")])
    return ChatCompletionRequest(model=model, messages=messages, **fields)


def _first(request: ChatCompletionRequest, routing: TierRouting | None, **kwargs: Any) -> Any:
    return first_action(request, CAPS, tier_routing=routing, **kwargs)


SMALL_CLAUDE = {"SMALL": {"model": "claude-sonnet-5-5"}}
BOTH = {"SMALL": {"model": "claude-sonnet-5-5"}, "MEDIUM": {"model": "gpt-4o"}}


def test_the_alias_is_one_string_exported_and_reserved_against_fallbacks() -> None:
    assert TIER_ROUTING_ALIAS == "primagate/auto"
    assert primagate.TIER_ROUTING_ALIAS == TIER_ROUTING_ALIAS
    assert TIER_ROUTING_ALIAS in core_fallbacks.RESERVED_MODEL_NAMES


def test_one_enabled_tier_is_dispatched_to_its_bound_model() -> None:
    action = _first(_request(), _routing({"MEDIUM": {"model": "gpt-4o"}}))
    assert isinstance(action, Dispatch)
    assert (action.provider, action.request.model) == ("openai", "gpt-4o")
    assert action.plan is None


def test_one_eligible_of_two_is_chosen_without_a_decision() -> None:
    request = _request(messages=[Message(role="user", content=IMAGE)])
    action = _first(request, _routing(BOTH))
    assert isinstance(action, Dispatch) and action.request.model == "gpt-4o"


def test_a_plain_model_never_reaches_the_filter() -> None:
    action = _first(_request("gpt-4o-mini"), _routing(BOTH))
    assert isinstance(action, Dispatch) and action.request.model == "gpt-4o-mini"
    assert isinstance(_first(_request("gpt-4o-mini"), None), Dispatch)


def test_zero_eligible_fails_before_any_dispatch_naming_each_tier_and_the_gap() -> None:
    request = _request(messages=[Message(role="user", content=IMAGE)])
    action = _first(request, _routing(SMALL_CLAUDE))
    assert isinstance(action, Raise)
    assert isinstance(action.error, errors.UnsupportedParamsError)
    assert "SMALL" in str(action.error) and "image" in str(action.error)
    assert action.error.attempt_chain == []


def test_binding_limits_remove_a_tier_its_adapter_would_allow() -> None:
    bound = {"SMALL": {"model": "gpt-4o-mini", "limits": {"supported_features": ["streaming"]}}}
    action = _first(_request(tools=[TOOL]), _routing(bound))
    assert isinstance(action, Raise)
    assert isinstance(action.error, errors.UnsupportedParamsError)
    assert "tools" in str(action.error)


def test_drop_params_is_applied_before_the_filter_and_recorded() -> None:
    bound = {"SMALL": {"model": "gpt-4o-mini", "limits": {"supported_features": ["streaming"]}}}
    action = _first(
        _request(tools=[TOOL]), _routing(bound), policy=DropPolicy(features=frozenset({"tools"}))
    )
    assert isinstance(action, Dispatch) and action.dropped_params == ("tools",)
    assert action.request.tools is None


def test_one_model_bound_to_two_tiers_is_one_candidate_and_the_lowest_keeps_it() -> None:
    twice = {"SMALL": {"model": "gpt-4o"}, "MEDIUM": {"model": "openai/gpt-4o"}}
    action = _first(_request(), _routing(twice))
    assert isinstance(action, Dispatch) and action.request.model == "gpt-4o"


def test_a_tier_that_is_not_enabled_is_not_a_candidate() -> None:
    action = _first(_request(), _routing(BOTH, enabled=["MEDIUM"]))
    assert isinstance(action, Dispatch) and action.request.model == "gpt-4o"


def test_a_repeated_model_is_dropped_after_the_filter_not_before() -> None:
    # SMALL's limits remove tools; MEDIUM names the same model with no such limit, so it stays.
    bound = {
        "SMALL": {"model": "gpt-4o", "limits": {"supported_features": ["streaming"]}},
        "MEDIUM": {"model": "gpt-4o"},
    }
    action = _first(_request(tools=[TOOL]), _routing(bound))
    assert isinstance(action, Dispatch) and action.request.tools


def test_two_eligible_tiers_need_the_analyzer_and_dispatch_nothing() -> None:
    both_able = {"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}}
    action = _first(_request(), _routing(both_able))
    assert isinstance(action, Raise)
    assert type(action.error) is errors.PrimaGateError
    assert "Analyzer" in str(action.error) and "not part of PrimaGate yet" in str(action.error)
    assert action.error.attempt_chain == []


def test_a_fault_that_is_not_a_missing_feature_is_not_reported_as_one() -> None:
    # No adapter for the Tier's provider: a setup fault, not a Tier lacking a feature.
    action = first_action(_request(), {"openai": OPENAI_CAPS}, tier_routing=_routing(SMALL_CLAUDE))
    assert isinstance(action, Raise)
    assert type(action.error) is errors.PrimaGateError


def test_the_alias_without_a_policy_is_refused_naming_what_is_missing() -> None:
    action = _first(_request(), None)
    assert isinstance(action, Raise)
    assert isinstance(action.error, errors.BadRequestError)
    assert "policy" in str(action.error) and "tier_bindings" in str(action.error)


def test_tier_routing_checks_each_candidate_with_the_same_function_as_every_other_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from primagate.core import call

    seen: list[str] = []
    real = call._build

    def spy(request: Any, provider: str, model: str, *rest: Any) -> Any:
        seen.append(model)
        return real(request, provider, model, *rest)

    monkeypatch.setattr(call, "_build", spy)
    _first(_request(messages=[Message(role="user", content=IMAGE)]), _routing(BOTH))
    assert seen == ["claude-sonnet-5-5", "gpt-4o"]
    seen.clear()
    _first(_request("gpt-4o"), _routing(BOTH))
    assert seen == ["gpt-4o"]


def test_narrowing_keeps_what_the_adapter_has_and_never_adds() -> None:
    bound = parse_tier_bindings(
        {
            "SMALL": {
                "model": "claude-sonnet-5-5",
                "limits": {"supported_features": ["tools", "images"]},
            }
        }
    )
    got = core_tiers.narrowed(CLAUDE_CAPS, bound[Tier.SMALL])
    assert got.tools and not got.images and not got.structured_output and not got.streaming
    assert got.unsupported_hints == CLAUDE_CAPS.unsupported_hints
    assert core_tiers.narrowed(CLAUDE_CAPS, None) == CLAUDE_CAPS


@pytest.mark.parametrize("name", ["tools", "structured_output", "images", "streaming"])
def test_unknown_limits_leave_each_feature_as_the_adapter_declares_it(name: str) -> None:
    bound = parse_tier_bindings({"SMALL": {"model": "gpt-4o"}})
    assert getattr(core_tiers.narrowed(OPENAI_CAPS, bound[Tier.SMALL]), name) is True
