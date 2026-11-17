import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, _json_ok
from test_events import Recorder
from test_retry_routing import _answer as worker_answer

from primagate import PrimaGate
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.types import ChatCompletion

ALIAS = TIER_ROUTING_ALIAS
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}
MSGS = [{"role": "user", "content": "where is my order"}]
IMAGE_MSGS = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is in this photo"},
            {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
        ],
    }
]
TWO = {"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}}
POLICY: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
    "ambiguous_context_default": "SMALL",
}


def _analyzer_reply(text: str, usage: bool = True) -> httpx2.Response:
    body = _json_ok("openai")
    body["choices"][0]["message"]["content"] = text
    body["usage"] = (
        {"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104} if usage else None
    )
    return httpx2.Response(200, json=body, headers=DATE)


class Run:
    """A gate over a mock that tells Analyzer questions from worker calls by their prompt."""

    def __init__(
        self,
        verdicts: list[str | Callable[[], httpx2.Response]],
        policy: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.verdicts = list(verdicts)
        self.analyzed: list[dict[str, Any]] = []
        self.workers: list[dict[str, Any]] = []
        self.handler = Recorder()
        self.gate = PrimaGate(
            api_keys=KEYS,
            transport=httpx2.MockTransport(self._handle),
            tier_bindings=bindings or TWO,
            policy=policy or POLICY,
            analyzer="gpt-4o-mini",
            event_handler=self.handler,
            **kwargs,
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        system = str(body.get("messages", [{}])[0].get("content", ""))
        if "Prompt version: analyzer" in system:
            self.analyzed.append(body)
            verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
            return verdict() if callable(verdict) else _analyzer_reply(verdict)
        self.workers.append(body)
        return worker_answer("openai", bool(body.get("stream")))

    @property
    def decisions(self) -> list[events.RoutingDecision]:
        return [e for e in self.handler.seen if isinstance(e, events.RoutingDecision)]

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.handler.seen if isinstance(e, kind)]


def test_the_analyzers_tier_gets_the_call_and_the_decision_is_logged(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert [w["model"] for w in run.workers] == ["gpt-4o"] and len(run.analyzed) == 1
    (decision,) = run.decisions
    assert decision.binding_id == "MEDIUM:openai/gpt-4o"
    assert (
        decision.profile_id
        == decision.input_fingerprint
        == decision.prep_prompt_version
        == ("unknown")
    )
    assert decision.analyzer_prompt_version == "analyzer-1"
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"


def test_the_response_shows_the_analyzer_cost_apart_from_the_workers(call: Caller) -> None:
    result = call(Run(["MEDIUM 0.9"]).gate, model=ALIAS, messages=MSGS)
    assert result.metadata is not None and result.metadata.analyzer is not None
    line = result.metadata.analyzer
    assert (line.provider, line.model) == ("openai", "gpt-4o-mini")
    assert line.cost is not None and line.token_counts is not None
    assert line.token_counts.total_tokens == 104
    assert result.metadata.cost != line.cost


def test_a_below_threshold_answer_takes_the_default_and_says_so(call: Caller) -> None:
    run = Run(["MEDIUM 0.5"])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert [w["model"] for w in run.workers] == ["gpt-4o-mini"]
    assert len(run.of(events.LowConfidenceDowngrade)) == 1 and len(run.decisions) == 1
    assert run.of(events.PremiumRoutingSelected) == []
    assert result.metadata is not None and result.metadata.tier == "SMALL"


def test_a_tier_above_the_cheapest_eligible_is_logged_as_premium(call: Caller) -> None:
    run = Run(["MEDIUM 0.95"])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert len(run.of(events.PremiumRoutingSelected)) == 1
    assert run.of(events.LowConfidenceDowngrade) == []


def test_the_cheapest_tier_is_not_premium(call: Caller) -> None:
    run = Run(["SMALL 0.95"])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert run.of(events.PremiumRoutingSelected) == []


def test_events_come_before_the_dispatch_and_never_hold_content(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert run.handler.types[:2] == ["routing_decision", "premium_routing_selected"]
    assert run.handler.types.index("dispatch_started") > 1
    for event in run.of(events.RoutingDecision):
        assert "order" not in json.dumps(event.to_dict())


def test_a_tool_loop_makes_one_decision_per_step_each_with_its_own_tier(call: Caller) -> None:
    run = Run(["SMALL 0.9", "MEDIUM 0.9", "SMALL 0.95"], cache_max_entries=1)
    for step in range(3):
        call(run.gate, model=ALIAS, messages=[{"role": "user", "content": f"step {step}"}])
    assert [d.binding_id for d in run.decisions] == [
        "SMALL:openai/gpt-4o-mini",
        "MEDIUM:openai/gpt-4o",
        "SMALL:openai/gpt-4o-mini",
    ]
    assert len(run.analyzed) == 3


def test_one_eligible_tier_skips_the_analyzer_but_is_still_logged(call: Caller) -> None:
    one = Run(
        ["MEDIUM 0.9"],
        bindings={"MEDIUM": {"model": "gpt-4o"}},
        policy={
            **POLICY,
            "enabled_tiers": ["MEDIUM"],
            "interim_binding": "MEDIUM",
            "ambiguous_context_default": "MEDIUM",
        },
    )
    result = call(one.gate, model=ALIAS, messages=MSGS)
    assert one.analyzed == [] and len(one.decisions) == 1
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert result.metadata.analyzer is None
    assert one.decisions[0].analyzer_prompt_version == "unknown"


def test_only_tiers_that_can_take_the_request_are_offered_and_a_filtered_default_moves_up(
    call: Caller,
) -> None:
    bindings = {
        "SMALL": {"model": "claude-sonnet-5-5"},
        "MEDIUM": {"model": "gpt-4o"},
        "POWERFUL": {"model": "gpt-4.1"},
    }
    policy = {**POLICY, "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"]}
    run = Run(["MEDIUM 0.1"], policy=policy, bindings=bindings)
    result = call(run.gate, model=ALIAS, messages=IMAGE_MSGS, drop_params=False)
    system = run.analyzed[0]["messages"][0]["content"]
    assert "MEDIUM | POWERFUL" in system and "SMALL" not in system
    # The default (SMALL) cannot take images, so the nearest Tier above it is used.
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert [w["model"] for w in run.workers] == ["gpt-4o"]


def test_a_filtered_default_with_nothing_stronger_falls_back_to_the_strongest_able(
    call: Caller,
) -> None:
    bindings = {
        "MEDIUM": {"model": "gpt-4o"},
        "POWERFUL": {"model": "claude-sonnet-5-5"},
        "SMALL": {"model": "gpt-4.1"},
    }
    policy = {
        **POLICY,
        "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"],
        "ambiguous_context_default": "POWERFUL",
    }
    run = Run(["SMALL 0.1"], policy=policy, bindings=bindings)
    result = call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    # POWERFUL cannot take images and nothing is stronger, so the strongest able Tier answers.
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert [w["model"] for w in run.workers] == ["gpt-4o"]


def test_an_analyzer_name_is_trimmed_at_setup() -> None:
    gate = PrimaGate(tier_bindings=TWO, policy=POLICY, analyzer="  gpt-4o-mini ")
    assert gate._tier_routing is not None and gate._tier_routing.analyzer == "gpt-4o-mini"


def test_a_repeat_is_answered_from_memory_without_asking_the_analyzer_again(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    call(run.gate, model=ALIAS, messages=MSGS)
    second = call(run.gate, model=ALIAS, messages=MSGS)
    assert second.metadata is not None and second.metadata.cache_hit
    assert len(run.analyzed) == 1 and len(run.decisions) == 1
    assert second.metadata.tier is None and second.metadata.analyzer is None


def test_the_analyzer_sees_the_bounded_conversation_not_the_system_prompt(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    messages = [
        {"role": "system", "content": "SECRET-INSTRUCTIONS"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    call(run.gate, model=ALIAS, messages=messages)
    sent = json.dumps(run.analyzed[0])
    assert "second question" in sent and "first answer" in sent
    assert "SECRET-INSTRUCTIONS" not in sent
    assert [m["role"] for m in run.workers[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_the_policy_in_force_when_the_call_starts_is_the_one_it_uses(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])

    def swap() -> httpx2.Response:
        # Mid-call, the gate's own settings change; the running call must not notice.
        for driver in (run.gate._driver, run.gate._async_driver):
            if driver is not None:
                driver._tier_routing = None
        return _analyzer_reply("MEDIUM 0.9")

    run.verdicts = [swap]
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"


def test_a_stream_asks_the_analyzer_then_opens_on_the_chosen_tier(
    stream_call: StreamCaller,
) -> None:
    run = Run(["MEDIUM 0.9"])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert len(run.analyzed) == 1 and [w["model"] for w in run.workers] == ["gpt-4o"]


def test_the_alias_as_a_plain_call_still_never_asks_the_analyzer(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    plain = call(run.gate, model="gpt-4o", messages=MSGS)
    assert run.analyzed == [] and run.decisions == []
    assert plain.metadata is not None and plain.metadata.tier is None
    assert plain.metadata.analyzer is None


def test_two_enabled_tiers_need_an_analyzer_at_setup() -> None:
    with pytest.raises(errors.BadRequestError, match="analyzer"):
        PrimaGate(tier_bindings=TWO, policy=POLICY)


def test_one_enabled_tier_may_leave_the_analyzer_out() -> None:
    one = {**POLICY, "enabled_tiers": ["MEDIUM"], "ambiguous_context_default": "MEDIUM"}
    PrimaGate(tier_bindings=TWO, policy=one)


@pytest.mark.parametrize("bad", [ALIAS, "", "  ", 5, "nope/model", "openai/"])
def test_a_bad_analyzer_is_refused_at_setup(bad: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        PrimaGate(tier_bindings=TWO, policy=POLICY, analyzer=bad)


def test_an_analyzer_with_no_policy_is_refused() -> None:
    with pytest.raises(errors.BadRequestError, match="policy"):
        PrimaGate(analyzer="gpt-4o-mini")


def test_the_response_metadata_round_trips_to_plain_data(call: Caller) -> None:
    result: ChatCompletion = call(Run(["MEDIUM 0.9"]).gate, model=ALIAS, messages=MSGS)
    shown = result.to_dict()["metadata"]
    assert shown["tier"] == "MEDIUM" and shown["analyzer"]["model"] == "gpt-4o-mini"
