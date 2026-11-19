from typing import Any

import pytest
from conftest import Caller, StreamCaller
from test_cache import ANSWERED, MSGS, OPENAI, _answer
from test_events import Recorder
from test_retry_routing import PRIMARY, Run

from primagate import PrimaGate
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS

ALIAS = TIER_ROUTING_ALIAS
ANTHROPIC = PRIMARY["anthropic"]
IMAGE_MSGS = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}],
    }
]


def _setup(bindings: dict[str, Any], enabled: list[str] | None = None) -> dict[str, Any]:
    names = enabled or list(bindings)
    return {
        "analyzer": "gpt-4o-mini",
        "tier_bindings": bindings,
        "policy": {
            "enabled_tiers": names,
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            "interim_binding": names[0],
            "ambiguous_context_default": names[0],
        },
    }


ONE = _setup({"MEDIUM": {"model": "gpt-4o"}})
CLAUDE_ONLY = _setup({"SMALL": {"model": "claude-sonnet-5-5"}})
TWO = _setup({"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}})


@pytest.mark.fr("FR15")
def test_the_alias_sends_the_call_to_the_only_enabled_tiers_model(call: Caller) -> None:
    run = Run({OPENAI: [_answer]}, **ONE)
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert result.choices[0].message.content == ANSWERED
    assert len(run.sent) == 1 and b'"model":"gpt-4o"' in run.sent[0].content.replace(b" ", b"")
    assert result.metadata is not None
    assert [row.model for row in result.metadata.attempt_chain] == ["openai/gpt-4o"]


@pytest.mark.fr("FR25")
def test_the_alias_picks_the_one_tier_that_can_do_the_job(call: Caller) -> None:
    both = _setup({"SMALL": {"model": "claude-sonnet-5-5"}, "MEDIUM": {"model": "gpt-4o"}})
    run = Run({OPENAI: [_answer], ANTHROPIC: [_answer]}, **both)
    call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    assert run.hosts == [OPENAI]


@pytest.mark.fr("FR15")
def test_a_stream_through_the_alias_goes_to_the_only_enabled_tier(
    stream_call: StreamCaller,
) -> None:
    run = Run({OPENAI: [_answer]}, **ONE)
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert run.hosts == [OPENAI]


@pytest.mark.fr("FR25")
def test_no_enabled_tier_able_to_take_the_request_fails_before_any_dispatch(
    call: Caller,
) -> None:
    handler = Recorder()
    run = Run({ANTHROPIC: [_answer]}, event_handler=handler, **CLAUDE_ONLY)
    with pytest.raises(errors.UnsupportedParamsError) as caught:
        call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    assert run.sent == []
    assert caught.value.attempt_chain == []
    assert "SMALL" in str(caught.value) and "image" in str(caught.value)
    assert handler.types == ["call_failed"]


@pytest.mark.fr("FR25")
def test_a_stream_with_no_able_tier_fails_before_any_dispatch(
    stream_call: StreamCaller,
) -> None:
    run = Run({ANTHROPIC: [_answer]}, **CLAUDE_ONLY)
    result = stream_call(run.gate, model=ALIAS, messages=IMAGE_MSGS)
    assert isinstance(result.call_error, errors.UnsupportedParamsError)
    assert run.sent == []


@pytest.mark.fr("FR5", "FR6", "FR25")
def test_a_caller_who_opts_in_to_dropping_images_gets_the_tier_and_a_record(
    call: Caller,
) -> None:
    run = Run({ANTHROPIC: [_answer]}, **CLAUDE_ONLY)
    result = call(run.gate, model=ALIAS, messages=IMAGE_MSGS, drop_params=["image"])
    assert run.hosts == [ANTHROPIC]
    assert result.metadata is not None and result.metadata.dropped_params == ["image"]


@pytest.mark.fr("FR22")
def test_two_able_tiers_with_an_unusable_analyzer_answer_use_the_interim_binding(
    call: Caller,
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler, **TWO)
    result = call(run.gate, model=ALIAS, messages=MSGS)
    # One question to the Analyzer, then one worker call on the Interim Tier (SMALL).
    assert len(run.sent) == 2 and result.choices
    assert b'"model":"gpt-4o-mini"' in run.sent[1].content.replace(b" ", b"")
    assert handler.types[:3] == ["routing_decision", "routing_degraded", "dispatch_started"]


@pytest.mark.fr("FR15")
def test_the_alias_without_a_policy_is_refused_before_any_dispatch(call: Caller) -> None:
    run = Run({OPENAI: [_answer]})
    with pytest.raises(errors.BadRequestError, match="policy"):
        call(run.gate, model=ALIAS, messages=MSGS)
    assert run.sent == []


@pytest.mark.fr("FR15")
def test_a_plain_model_name_bypasses_tier_routing(call: Caller) -> None:
    run = Run({OPENAI: [_answer]}, **TWO)
    result = call(run.gate, model="gpt-4o-mini", messages=MSGS)
    assert result.choices and len(run.sent) == 1
    assert b'"model":"gpt-4o-mini"' in run.sent[0].content.replace(b" ", b"")


@pytest.mark.fr("FR31")
def test_a_repeat_of_an_alias_call_is_served_from_memory_with_the_cache_event(
    call: Caller,
) -> None:
    handler = Recorder()
    run = Run({OPENAI: [_answer]}, event_handler=handler, **ONE)
    call(run.gate, model=ALIAS, messages=MSGS)
    second = call(run.gate, model=ALIAS, messages=MSGS)
    assert len(run.sent) == 1
    assert second.metadata is not None and second.metadata.cache_hit
    hit = [e for e in handler.seen if isinstance(e, events.CacheHit)]
    assert len(hit) == 1 and hit[0].model == ALIAS


@pytest.mark.fr("FR31")
def test_a_cache_hit_returns_before_the_filter_and_the_analyzer_are_consulted(
    call: Caller,
) -> None:
    # The Cache is read before the Tier is decided, so a hit never depends on the filter.
    first = Run({OPENAI: [_answer]}, **ONE)
    call(first.gate, model=ALIAS, messages=MSGS)
    # A gate that would now ask the Analyzer for this call, but which shares the filled Cache.
    second = Run({OPENAI: [_answer]}, **TWO)
    second.gate._cache = first.gate._cache
    call(second.gate, model=ALIAS, messages=MSGS, cache=False)
    asked = len(second.sent)
    again = call(second.gate, model=ALIAS, messages=MSGS)
    assert again.metadata is not None and again.metadata.cache_hit
    assert len(second.sent) == asked


@pytest.mark.fr("FR31")
def test_a_degraded_answer_is_never_cached(call: Caller) -> None:
    run = Run({OPENAI: [_answer]}, **TWO)
    for _ in range(2):
        call(run.gate, model=ALIAS, messages=MSGS)
    # Each call asked the Analyzer and then a worker: nothing was stored to replay.
    assert len(run.sent) == 4


@pytest.mark.fr("FR15")
def test_a_plain_call_with_a_fallback_list_is_unaffected_by_tier_setup(call: Caller) -> None:
    run = Run({OPENAI: [_answer]}, default_fallbacks=["gpt-4o-mini"], **TWO)
    assert call(run.gate, model="gpt-4o", messages=MSGS).choices
    assert len(run.sent) == 1


@pytest.mark.fr("FR7")
@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_a_fallback_entry_naming_the_alias_is_refused_at_setup() -> None:
    with pytest.raises(errors.BadRequestError, match="primagate/auto"):
        PrimaGate(fallbacks={"gpt-4o": [ALIAS]})
    with pytest.raises(errors.BadRequestError, match="primagate/auto"):
        PrimaGate(default_fallbacks=[ALIAS])
