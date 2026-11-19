import json
from typing import Any

import httpx2
import pytest
from conftest import Caller
from test_cache import MSGS
from test_retry_routing import PRIMARY, Run, _answer
from test_tier_alias import _setup
from test_tier_routing import _analyzer_reply

from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.tiers import UNKNOWN, Tier

pytestmark = pytest.mark.fr("FR16")

TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def test_a_routed_call_goes_to_the_bound_provider_and_model(call: Caller) -> None:
    run = Run(
        {PRIMARY["anthropic"]: [_answer]},
        **_setup({"SMALL": {"model": "anthropic/claude-sonnet-5-5"}}),
    )
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.hosts == [PRIMARY["anthropic"]]
    assert json.loads(run.sent[0].content)["model"] == "claude-sonnet-5-5"
    assert result.metadata is not None and result.metadata.tier == "SMALL"
    assert [row.model for row in result.metadata.attempt_chain] == ["anthropic/claude-sonnet-5-5"]


BOTH_BOUND = _setup(
    {"SMALL": {"model": "anthropic/claude-sonnet-5-5"}, "MEDIUM": {"model": "openai/gpt-4o"}}
)


def _verdict(text: str) -> Any:
    def reply(provider: str, stream: bool) -> httpx2.Response:
        return _analyzer_reply(text)

    return reply


@pytest.mark.parametrize(
    ("tier", "host", "model"),
    [("SMALL", "anthropic", "claude-sonnet-5-5"), ("MEDIUM", "openai", "gpt-4o")],
)
def test_each_tier_reaches_its_own_bound_provider_and_model(
    call: Caller, tier: str, host: str, model: str
) -> None:
    # The Analyzer (gpt-4o-mini) answers first on the OpenAI host, then the worker is sent.
    script = {
        PRIMARY["openai"]: [_verdict(f"{tier} 0.95"), _answer],
        PRIMARY["anthropic"]: [_answer],
    }
    run = Run(script, **BOTH_BOUND)
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.hosts == [PRIMARY["openai"], PRIMARY[host]]
    assert json.loads(run.sent[1].content)["model"] == model
    assert result.metadata is not None and result.metadata.tier == tier


EVERY_FIELD_UNKNOWN: dict[str, Any] = {
    "model": "gpt-4o",
    "limits": dict.fromkeys(
        ("context_window", "max_output_tokens", "supported_features"), "unknown"
    ),
    "pricing": {"input": "unknown", "output": "unknown"},
}


@pytest.mark.parametrize(
    "declared", [{"model": "gpt-4o"}, EVERY_FIELD_UNKNOWN], ids=["omitted", "said"]
)
def test_undisclosed_fields_are_unknown_never_omitted_and_still_route(
    call: Caller, declared: dict[str, Any]
) -> None:
    run = Run({PRIMARY["openai"]: [_answer]}, **_setup({"MEDIUM": declared}))
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS, tools=[TOOL])
    # Unknown limits narrow nothing, so a tool call still reaches the bound model.
    assert run.hosts == [PRIMARY["openai"]] and "tools" in json.loads(run.sent[0].content)
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"

    binding = run.gate._tier_bindings[Tier.MEDIUM]
    limits, pricing = binding.limits, binding.pricing
    assert limits.context_window is limits.max_output_tokens is UNKNOWN
    assert limits.supported_features is UNKNOWN
    assert pricing.input is pricing.output is UNKNOWN
    shown = binding.to_dict()
    assert shown["limits"] == dict.fromkeys(
        ("context_window", "max_output_tokens", "supported_features"), "unknown"
    )
    assert shown["pricing"] == {"input": "unknown", "output": "unknown"}


def test_saying_unknown_behaves_exactly_like_leaving_the_field_out(call: Caller) -> None:
    seen = []
    for declared in ({"model": "gpt-4o"}, EVERY_FIELD_UNKNOWN):
        run = Run({PRIMARY["openai"]: [_answer]}, **_setup({"MEDIUM": declared}))
        result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS, tools=[TOOL])
        assert result.metadata is not None
        meta = result.metadata.to_dict()
        binding = run.gate._tier_bindings[Tier.MEDIUM]
        seen.append((binding, binding.to_dict(), run.sent[0].content, meta))
    assert seen[0] == seen[1]


@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_a_policy_enabling_a_tier_with_no_binding_is_refused_at_setup() -> None:
    setup = _setup({"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}})
    setup["policy"]["enabled_tiers"] = ["SMALL", "MEDIUM", "POWERFUL"]
    with pytest.raises(errors.BadRequestError, match="POWERFUL"):
        PrimaGate(**setup)
