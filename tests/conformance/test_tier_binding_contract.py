import json
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_cache import MSGS
from test_retry_routing import PRIMARY, Run, _answer, _status
from test_tier_alias import _setup
from test_tier_failover import FOUR, GPT, MINI, POLICY
from test_tier_failover import Run as TierRun
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
    {
        "SMALL": {
            "model": "anthropic/claude-sonnet-5-5",
            "settings": {"temperature": 0.2, "max_tokens": 512, "stop": ["END"]},
        },
        "MEDIUM": {
            "model": "openai/gpt-4o",
            "settings": {"temperature": 0.7, "max_tokens": 2048, "stop": ["DONE"]},
        },
    }
)


def _verdict(text: str) -> Any:
    def reply(provider: str, stream: bool) -> httpx2.Response:
        return _analyzer_reply(text)

    return reply


@pytest.mark.parametrize(
    ("tier", "host", "model", "settings"),
    [
        (
            "SMALL",
            "anthropic",
            "claude-sonnet-5-5",
            {"temperature": 0.2, "max_tokens": 512, "stop_sequences": ["END"]},
        ),
        ("MEDIUM", "openai", "gpt-4o", {"temperature": 0.7, "max_tokens": 2048, "stop": ["DONE"]}),
    ],
)
def test_each_tier_reaches_its_own_bound_provider_and_model_with_its_settings(
    call: Caller, tier: str, host: str, model: str, settings: dict[str, Any]
) -> None:
    # The Analyzer (gpt-4o-mini) answers first on the OpenAI host, then the worker is sent.
    script = {
        PRIMARY["openai"]: [_verdict(f"{tier} 0.95"), _answer],
        PRIMARY["anthropic"]: [_answer],
    }
    run = Run(script, **BOTH_BOUND)
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.hosts == [PRIMARY["openai"], PRIMARY[host]]
    body = json.loads(run.sent[1].content)
    assert body["model"] == model
    assert {name: body.get(name) for name in settings} == settings
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


# --- a binding's settings on the wire ---

FIELDS = ("temperature", "top_p", "max_tokens", "seed", "stop")
SMALL_SENT: dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.5,
    "max_tokens": 512,
    "seed": 7,
    "stop": ["END"],
}
MEDIUM_SENT: dict[str, Any] = {"temperature": 0.6, "max_tokens": 1024}
SETTLED = {
    "SMALL": {"model": MINI, "settings": SMALL_SENT},
    "MEDIUM": {"model": GPT, "settings": MEDIUM_SENT},
    "POWERFUL": {"model": FOUR},
}
FALLBACK = "gpt-4.1-mini"


def _settings(body: dict[str, Any]) -> dict[str, Any]:
    return {name: body[name] for name in FIELDS if name in body}


def _settled(verdicts: list[Any], workers: dict[str, Any] | None = None, **kw: Any) -> TierRun:
    return TierRun(verdicts, workers or {}, bindings=SETTLED, **kw)


def test_a_tiers_settings_fill_what_the_request_leaves_unset(call: Caller) -> None:
    run = _settled(["SMALL 0.9"])
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.models == [MINI]
    # The tuple a binding stores for `stop` goes out as the list the request type expects.
    assert _settings(run.workers[0]) == SMALL_SENT
    assert result.metadata is not None and result.metadata.tier == "SMALL"


@pytest.mark.parametrize(
    "asked",
    [{"temperature": 0.9, "stop": "STOP"}, {"temperature": 0, "seed": 0}],
    ids=["values", "zeros"],
)
def test_the_callers_own_values_win_over_the_binding(call: Caller, asked: dict[str, Any]) -> None:
    # A zero is a value the Caller chose, never a field left unset.
    run = _settled(["SMALL 0.9"])
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS, **asked)
    assert _settings(run.workers[0]) == {**SMALL_SENT, **asked}


@pytest.mark.parametrize(
    ("tier", "sent"), [("SMALL", SMALL_SENT), ("MEDIUM", MEDIUM_SENT), ("POWERFUL", {})]
)
def test_each_tier_carries_only_its_own_settings(
    call: Caller, tier: str, sent: dict[str, Any]
) -> None:
    run = _settled([f"{tier} 0.9"])
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert _settings(run.workers[0]) == sent


def test_medium_failover_carries_mediums_settings_not_the_failed_tiers(call: Caller) -> None:
    run = _settled(["SMALL 0.9"], {MINI: [_status(503)]})
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.models == [MINI, GPT]
    assert [_settings(body) for body in run.workers] == [SMALL_SENT, MEDIUM_SENT]
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"


def test_the_interim_binding_carries_its_own_settings(call: Caller) -> None:
    # An unreadable verdict degrades to the Interim Binding, MEDIUM here.
    run = _settled(["nonsense"])
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert run.models == [GPT] and _settings(run.workers[0]) == MEDIUM_SENT
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"


def test_a_fallback_entry_gets_only_the_callers_values(call: Caller) -> None:
    script = {MINI: [_status(503)], GPT: [_status(503)]}
    run = _settled(["SMALL 0.9"], script, default_fallbacks=[FALLBACK])
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS, top_p=0.3)
    assert run.models == [MINI, GPT, FALLBACK]
    assert [_settings(body) for body in run.workers] == [
        {**SMALL_SENT, "top_p": 0.3},
        {**MEDIUM_SENT, "top_p": 0.3},
        {"top_p": 0.3},
    ]


def test_a_plain_model_call_ignores_the_bindings(call: Caller) -> None:
    run = _settled(["SMALL 0.9"])
    call(run.gate, model=MINI, messages=MSGS)
    assert run.analyzed == [] and _settings(run.workers[0]) == {}


def test_the_analyzer_request_never_takes_a_binding_setting(call: Caller) -> None:
    # The Analyzer runs on the same model SMALL binds, so a leak would show here.
    run = _settled(["SMALL 0.9"])
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    (asked,) = run.analyzed
    assert asked["model"] == MINI
    assert _settings(asked) == {"temperature": 0, "max_tokens": 16}


def test_the_same_routed_call_twice_is_a_cache_hit(call: Caller) -> None:
    run = _settled(["SMALL 0.9"])
    first = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    second = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert len(run.workers) == 1 and len(run.analyzed) == 1
    assert first.metadata is not None and first.metadata.cache_hit is False
    assert second.metadata is not None and second.metadata.cache_hit is True


def test_a_routed_stream_carries_the_chosen_tiers_settings(stream_call: StreamCaller) -> None:
    run = _settled(["MEDIUM 0.9"])
    result = stream_call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None and result.chunks
    assert run.models == [GPT] and _settings(run.workers[0]) == MEDIUM_SENT


def test_each_step_of_a_tool_loop_carries_the_settings_of_its_own_tier(call: Caller) -> None:
    run = _settled(["SMALL 0.9", "MEDIUM 0.9"])
    asked = [{"role": "user", "content": "look it up"}]
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=asked, tools=[TOOL])
    called = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    answered = {"role": "tool", "tool_call_id": "t1", "content": "found"}
    call(run.gate, model=TIER_ROUTING_ALIAS, messages=[*asked, called, answered], tools=[TOOL])
    assert run.models == [MINI, GPT] and len(run.analyzed) == 2
    assert [_settings(body) for body in run.workers] == [SMALL_SENT, MEDIUM_SENT]


SETUP = "setup validation runs when the instance is made, before any call"
SECRET_SEED = 918273645


@pytest.mark.mode_independent(SETUP)
@pytest.mark.parametrize("drops", [None, True, ["tools"]], ids=["off", "hints", "features"])
def test_a_setting_the_provider_cannot_honour_is_refused_at_setup(drops: Any) -> None:
    claude = {"model": "anthropic/claude-sonnet-5-5", "settings": {"seed": SECRET_SEED}}
    with pytest.raises(errors.BadRequestError) as caught:
        PrimaGate(drop_params=drops, **_setup({"SMALL": claude}))
    message = str(caught.value)
    assert "tier_bindings SMALL: settings.seed is not supported by anthropic" in message
    assert str(SECRET_SEED) not in message


@pytest.mark.mode_independent(SETUP)
def test_max_tokens_above_the_known_output_limit_is_refused_at_setup() -> None:
    over = {
        "model": "gpt-4o",
        "settings": {"max_tokens": 8192},
        "limits": {"max_output_tokens": 4096},
    }
    with pytest.raises(errors.BadRequestError) as caught:
        PrimaGate(**_setup({"SMALL": over}))
    message = str(caught.value)
    assert "SMALL" in message and "settings.max_tokens" in message
    assert "8192" not in message and "4096" not in message


@pytest.mark.mode_independent(SETUP)
def test_max_tokens_at_or_without_a_known_output_limit_is_accepted() -> None:
    at = {
        "model": "gpt-4o",
        "settings": {"max_tokens": 4096},
        "limits": {"max_output_tokens": 4096},
    }
    unknown = {"model": "gpt-4o-mini", "settings": {"max_tokens": 10**9}}
    PrimaGate(**_setup({"SMALL": unknown, "MEDIUM": at}))


@pytest.mark.mode_independent(SETUP)
def test_enabled_tiers_on_one_model_with_different_settings_are_refused_at_setup() -> None:
    # A bare name and its `provider/model` form are the same model.
    declared = {
        "SMALL": {"model": "gpt-4o", "settings": {"temperature": 0.25}},
        "MEDIUM": {"model": "openai/gpt-4o", "settings": {"temperature": 0.75}},
    }
    with pytest.raises(errors.BadRequestError) as caught:
        PrimaGate(**_setup(declared))
    message = str(caught.value)
    assert "SMALL" in message and "MEDIUM" in message and "openai/gpt-4o" in message
    assert "0.25" not in message and "0.75" not in message
    # A Tier that is not enabled is never routed to, so its settings cannot be lost.
    PrimaGate(**_setup(declared, enabled=["SMALL"]))


SHARED = {"model": GPT, "settings": {"temperature": 0.3, "max_tokens": 300}}


def test_enabled_tiers_on_one_model_with_the_same_settings_are_one_candidate(
    call: Caller,
) -> None:
    policy = {
        **POLICY,
        "enabled_tiers": ["SMALL", "MEDIUM"],
        "interim_binding": "SMALL",
        "ambiguous_context_default": "SMALL",
    }
    run = TierRun(["MEDIUM 0.9"], {}, policy=policy, bindings={"SMALL": SHARED, "MEDIUM": SHARED})
    result = call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    # One candidate, so the Analyzer is not asked and the first Tier on the model serves.
    assert run.analyzed == [] and run.models == [GPT]
    assert _settings(run.workers[0]) == SHARED["settings"]
    assert result.metadata is not None and result.metadata.tier == "SMALL"
