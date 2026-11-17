from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_tier_routing import IMAGE_MSGS, MSGS, POLICY, Run

from primagate.schema import events
from primagate.schema.policy import TIER_ROUTING_ALIAS

ALIAS = TIER_ROUTING_ALIAS


def _down() -> httpx2.Response:
    return httpx2.Response(503, json={"error": {"message": "x", "type": "server_error"}})


def _timeout() -> httpx2.Response:
    raise httpx2.ReadTimeout("slow")


def _degraded(run: Run) -> list[events.RoutingDegraded]:
    return run.of(events.RoutingDegraded)


def _models(run: Run) -> list[str]:
    return [w["model"] for w in run.workers]


CASES: list[tuple[str, Any, str]] = [
    ("5xx", _down, "provider_error"),
    ("timeout", _timeout, "provider_error"),
    ("malformed", "I would say MEDIUM 0.9", "malformed_output"),
    ("empty", "", "malformed_output"),
]


@pytest.mark.parametrize(("label", "verdict", "reason"), CASES, ids=[c[0] for c in CASES])
def test_a_failing_or_unusable_analyzer_uses_the_interim_binding_once(
    call: Caller, label: str, verdict: Any, reason: str
) -> None:
    run = Run([verdict])
    result = call(run.gate, model=ALIAS, messages=MSGS)
    # The Interim Binding here is MEDIUM; one Analyzer question, one worker call, no retry.
    assert len(run.analyzed) == 1 and _models(run) == ["gpt-4o"] and len(run.workers) == 1
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    (event,) = _degraded(run)
    assert event.reason == reason and event.binding_id == "MEDIUM:openai/gpt-4o"
    assert event.analyzer_prompt_version == "analyzer-1"


def test_the_events_come_in_order_before_the_dispatch(call: Caller) -> None:
    run = Run(["nonsense"])
    call(run.gate, model=ALIAS, messages=MSGS)
    # Interim here is MEDIUM, above the cheapest able Tier, so it is also a premium choice.
    assert run.handler.types[:4] == [
        "routing_decision",
        "routing_degraded",
        "premium_routing_selected",
        "dispatch_started",
    ]
    assert run.of(events.LowConfidenceDowngrade) == []


def test_a_degraded_event_is_a_warning_with_ids_and_a_fixed_reason_only(call: Caller) -> None:
    run = Run([_down])
    call(run.gate, model=ALIAS, messages=MSGS)
    (event,) = _degraded(run)
    assert event.log_level > events.RoutingDecision.log_level
    shown = event.to_dict()
    assert shown["reason"] == "provider_error"
    assert "order" not in str(shown) and "server_error" not in str(shown)


def test_a_malformed_answer_still_reports_what_the_analyzer_cost(call: Caller) -> None:
    result = call(Run(["nonsense"]).gate, model=ALIAS, messages=MSGS)
    assert result.metadata is not None and result.metadata.analyzer is not None
    assert result.metadata.analyzer.cost is not None


def test_a_failed_request_to_the_analyzer_has_no_cost_line(call: Caller) -> None:
    result = call(Run([_down]).gate, model=ALIAS, messages=MSGS)
    assert result.metadata is not None and result.metadata.analyzer is None


def test_a_latest_turn_over_the_budget_never_asks_the_analyzer(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"], policy={**POLICY, "context_token_budget": 2})
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.analyzed == [] and _models(run) == ["gpt-4o"]
    (event,) = _degraded(run)
    assert event.reason == "over_budget" and event.analyzer_prompt_version == "unknown"
    assert result.metadata is not None and result.metadata.analyzer is None


def test_a_turn_with_nothing_the_analyzer_can_read_is_degraded(call: Caller) -> None:
    only_image = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}],
        }
    ]
    run = Run(["MEDIUM 0.9"])
    call(run.gate, model=ALIAS, messages=only_image)
    assert run.analyzed == [] and len(run.workers) == 1
    assert _degraded(run)[0].reason == "no_conversation"


def test_an_answer_naming_a_tier_that_cannot_take_the_request_is_degraded(call: Caller) -> None:
    bindings = {
        "SMALL": {"model": "claude-sonnet-5-5"},
        "MEDIUM": {"model": "gpt-4o"},
        "POWERFUL": {"model": "gpt-4.1"},
    }
    policy = {**POLICY, "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"]}
    run = Run(["SMALL 0.99"], policy=policy, bindings=bindings)
    call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    assert _degraded(run)[0].reason == "tier_not_enabled" and _models(run) == ["gpt-4o"]


def test_an_interim_tier_that_cannot_take_the_request_moves_to_the_nearest_able_one(
    call: Caller,
) -> None:
    bindings = {
        "SMALL": {"model": "claude-sonnet-5-5"},
        "MEDIUM": {"model": "gpt-4o"},
        "POWERFUL": {"model": "gpt-4.1"},
    }
    policy = {
        **POLICY,
        "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"],
        "interim_binding": "SMALL",
    }
    run = Run(["nonsense"], policy=policy, bindings=bindings)
    result = call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert _models(run) == ["gpt-4o"] and len(run.analyzed) == 1


def test_a_degraded_answer_is_not_stored_so_the_next_call_asks_again(call: Caller) -> None:
    run = Run(["nonsense"])
    first = call(run.gate, model=ALIAS, messages=MSGS)
    second = call(run.gate, model=ALIAS, messages=MSGS)
    assert second.metadata is not None and not second.metadata.cache_hit
    assert len(run.analyzed) == 2 and len(run.workers) == 2 and first.choices


def test_a_good_answer_is_neither_degraded_nor_kept_from_the_cache(call: Caller) -> None:
    run = Run(["MEDIUM 0.9"])
    call(run.gate, model=ALIAS, messages=MSGS)
    again = call(run.gate, model=ALIAS, messages=MSGS)
    assert _degraded(run) == []
    assert again.metadata is not None and again.metadata.cache_hit


def test_a_below_threshold_answer_is_a_downgrade_not_a_degrade(call: Caller) -> None:
    run = Run(["MEDIUM 0.5"])
    call(run.gate, model=ALIAS, messages=MSGS)
    assert _degraded(run) == [] and len(run.of(events.LowConfidenceDowngrade)) == 1


def test_a_degraded_stream_opens_on_the_interim_tier(stream_call: StreamCaller) -> None:
    run = Run(["nonsense"])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert _models(run) == ["gpt-4o"] and _degraded(run)[0].reason == "malformed_output"
