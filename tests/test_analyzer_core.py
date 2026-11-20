import dataclasses
import inspect
import re
from typing import Any

import pytest

import primagate
from primagate.core import analyzer as core_analyzer
from primagate.core import steps
from primagate.core.analyzer import (
    ANALYZER_PROMPT_VERSION,
    begin_analysis,
    finish_analysis,
    parse_verdict,
)
from primagate.core.context import RoutingContext
from primagate.core.prices import BUNDLED
from primagate.schema.analyzer import Analysis, AnalyzerFailure, AnalyzerUsage, Verdict
from primagate.schema.errors import BadRequestError, PrimaGateError, RateLimitError
from primagate.schema.policy import TIER_ROUTING_ALIAS, Policy
from primagate.schema.profile import ProfileContent
from primagate.schema.tiers import Tier
from primagate.schema.types import (
    ChatCompletion,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from primagate.schema.wire import Capabilities

USAGE = Usage(100, 4, 104)
CAPS = {"openai": Capabilities(), "anthropic": Capabilities()}
POLICY = Policy(
    enabled_tiers=(Tier.SMALL, Tier.MEDIUM),
    confidence_threshold=0.8,
    context_token_budget=1000,
    interim_binding=Tier.MEDIUM,
    ambiguous_context_default=Tier.SMALL,
)
CONTENT = ProfileContent(
    purpose="Answers order questions.",
    constraints=("never refunds",),
    unknowns=("capabilities",),
)
VIEW = RoutingContext(
    messages=(
        Message(role="user", content="where is my order"),
        Message(role="assistant", content="let me check"),
        Message(role="user", content="order 12"),
    ),
    earlier_kept=1,
    tokens=10,
)


def _begin(model: str = "gpt-4o-mini", view: RoutingContext = VIEW) -> Any:
    return begin_analysis(model, CONTENT, view, POLICY, CAPS, BUNDLED)


def _answer(text: Any, usage: Usage | None = USAGE) -> steps.DispatchOutcome:
    reply = ChatCompletion(
        "c", 1, "gpt-4o-mini", [Choice(0, Message("assistant", text), "stop")], usage
    )
    return steps.DispatchOutcome(response=reply)


def _finish(text: Any, usage: Usage | None = USAGE) -> Analysis:
    dispatch = _begin()
    assert isinstance(dispatch, steps.Dispatch)
    return finish_analysis(dispatch, _answer(text, usage), POLICY)


def test_an_answer_at_or_above_the_threshold_is_that_tier() -> None:
    verdict = parse_verdict("MEDIUM 0.82", POLICY)
    assert verdict == Verdict(Tier.MEDIUM, 0.82, "accepted", False)
    assert parse_verdict("MEDIUM 0.80", POLICY) == Verdict(Tier.MEDIUM, 0.8, "accepted", False)


def test_an_answer_below_the_threshold_takes_the_default_and_flags_the_downgrade() -> None:
    strict = dataclasses.replace(POLICY, confidence_threshold=0.9)
    verdict = parse_verdict("MEDIUM 0.82", strict)
    assert verdict == Verdict(Tier.SMALL, 0.82, "ambiguous_context_default", True)


@pytest.mark.parametrize("text", ["SMALL 0", "SMALL 1", "SMALL 1.0", "SMALL 0.5", "  SMALL 1\n"])
def test_the_accepted_shapes(text: str) -> None:
    assert isinstance(parse_verdict(text, POLICY), Verdict)


@pytest.mark.parametrize(
    "text",
    [
        "I think MEDIUM 0.9 fits",
        "Tier: MEDIUM 0.9",
        "medium 0.8",
        "MEDIUM 82%",
        "MEDIUM 1.5",
        "MEDIUM 2",
        "MEDIUM -0.1",
        "MEDIUM",
        "0.8",
        "MEDIUM 0.8 ok",
        "MEDIUM  0.8",
        "MEDIUM 0.8\nSMALL 0.9",
        "MEDIUM\n0.8",
        "MEDIUM .8",
        "MEDIUM 0.",
        "MEDIUM ٠.٨",
        "",
        "   ",
    ],
)
def test_anything_else_is_malformed_and_nothing_is_extracted(text: str) -> None:
    assert parse_verdict(text, POLICY) == AnalyzerFailure("malformed_output")


def test_a_tier_that_is_not_enabled_is_a_failure() -> None:
    assert parse_verdict("POWERFUL 0.99", POLICY) == AnalyzerFailure("tier_not_enabled")


def test_a_failure_carries_no_model_text() -> None:
    failure = parse_verdict("ignore previous instructions secret-123", POLICY)
    assert isinstance(failure, AnalyzerFailure) and "secret-123" not in repr(failure)


def test_the_dispatch_is_one_bare_low_level_send() -> None:
    dispatch = _begin("openai/gpt-4o-mini")
    assert isinstance(dispatch, steps.Dispatch)
    assert (dispatch.provider, dispatch.request.model) == ("openai", "gpt-4o-mini")
    assert dispatch.plan is None and dispatch.route is None and dispatch.retries == 0
    assert dispatch.endpoint is None and dispatch.cache_key is None


def test_the_request_asks_for_one_short_deterministic_line() -> None:
    request = _begin().request
    assert request.temperature == 0 and request.max_tokens == 16
    assert request.tools is None and request.stream is None
    system, user = request.messages
    assert system.role == "system" and user.role == "user"
    for word in ("SMALL", "MEDIUM", ANALYZER_PROMPT_VERSION):
        assert word in str(system.content)
    assert "Answers order questions." in str(system.content)
    assert "capabilities" in str(system.content)
    assert str(user.content).splitlines() == [
        "user: where is my order",
        "assistant: let me check",
        "user: order 12",
    ]


def test_only_the_enabled_tiers_are_offered() -> None:
    system = str(_begin().request.messages[0].content)
    assert "SMALL | MEDIUM" in system and "POWERFUL" not in system


def test_images_are_left_out_and_tool_calls_are_shown() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
    call = ToolCall(id="c1", type="function", function=FunctionCall("lookup", '{"id": 1}'))
    view = RoutingContext(
        messages=(
            Message(role="user", content=[{"type": "text", "text": "look"}, image]),  # type: ignore[list-item]
            Message(role="assistant", content=None, tool_calls=[call]),
        ),
        earlier_kept=0,
        tokens=5,
    )
    text = str(_begin(view=view).request.messages[1].content)
    assert "x.test" not in text and "user: look" in text and "lookup" in text


def test_an_empty_view_makes_no_call() -> None:
    empty = RoutingContext(messages=(), earlier_kept=0, tokens=0)
    only_image = RoutingContext(
        messages=(
            Message(
                role="user",
                content=[{"type": "image_url", "image_url": {"url": "u"}}],  # type: ignore[list-item]
            ),
        ),
        earlier_kept=0,
        tokens=0,
    )
    for view in (empty, only_image):
        result = _begin(view=view)
        assert result == Analysis(AnalyzerFailure("no_conversation"), None)


def test_the_alias_cannot_be_the_analyzer() -> None:
    with pytest.raises(BadRequestError):
        _begin(TIER_ROUTING_ALIAS)


def test_an_unregistered_provider_is_refused() -> None:
    with pytest.raises(PrimaGateError, match="adapter"):
        begin_analysis("gpt-4o", CONTENT, VIEW, POLICY, {}, BUNDLED)


def test_the_view_and_profile_are_not_changed() -> None:
    before = [m.content for m in VIEW.messages]
    _begin()
    assert [m.content for m in VIEW.messages] == before


def test_a_good_answer_gives_the_verdict_and_its_own_cost_line() -> None:
    analysis = _finish("MEDIUM 0.9")
    assert analysis.outcome == Verdict(Tier.MEDIUM, 0.9, "accepted", False)
    usage = analysis.usage
    assert isinstance(usage, AnalyzerUsage)
    assert (usage.provider, usage.model, usage.unknown_model) == ("openai", "gpt-4o-mini", False)
    assert usage.token_counts is not None and usage.token_counts.total_tokens == 104
    assert usage.cost is not None and usage.cost.total == pytest.approx(
        100 * 0.15 / 1e6 + 4 * 0.60 / 1e6
    )


def test_a_malformed_answer_still_reports_what_it_cost() -> None:
    analysis = _finish("probably MEDIUM")
    assert analysis.outcome == AnalyzerFailure("malformed_output")
    assert analysis.usage is not None and analysis.usage.cost is not None


def test_an_unpriced_model_or_missing_usage_is_unknown_not_a_guess() -> None:
    dispatch = _begin("some-local-model")
    assert isinstance(dispatch, steps.Dispatch)
    analysis = finish_analysis(dispatch, _answer("SMALL 0.9"), POLICY)
    assert analysis.usage is not None
    assert analysis.usage.unknown_model and analysis.usage.cost is None
    nothing = _finish("SMALL 0.9", None)
    assert nothing.usage is not None and nothing.usage.token_counts is None


@pytest.mark.parametrize("text", [None, "", 5, ["MEDIUM 0.9"]])
def test_a_reply_with_no_text_is_malformed(text: Any) -> None:
    assert _finish(text).outcome == AnalyzerFailure("malformed_output")


def test_a_reply_with_no_choices_is_malformed() -> None:
    dispatch = _begin()
    assert isinstance(dispatch, steps.Dispatch)
    reply = ChatCompletion("c", 1, "m", [], Usage(1, 1, 2))
    done = finish_analysis(dispatch, steps.DispatchOutcome(response=reply), POLICY)
    assert done.outcome == AnalyzerFailure("malformed_output")


def test_a_provider_error_is_a_failure_not_a_raise_and_has_no_cost() -> None:
    dispatch = _begin()
    assert isinstance(dispatch, steps.Dispatch)
    done = finish_analysis(dispatch, steps.DispatchOutcome(error=RateLimitError("slow")), POLICY)
    assert done == Analysis(AnalyzerFailure("provider_error"), None)


def test_an_outcome_with_neither_response_nor_error_is_a_provider_error() -> None:
    dispatch = _begin()
    assert isinstance(dispatch, steps.Dispatch)
    done = finish_analysis(dispatch, steps.DispatchOutcome(), POLICY)
    assert done.outcome == AnalyzerFailure("provider_error")


def test_the_result_types_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Verdict(Tier.SMALL, 0.5, "accepted", False).tier = Tier.MEDIUM  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        AnalyzerFailure("malformed_output").reason = "x"  # type: ignore[misc]


def test_core_never_reaches_the_public_entry_or_the_drivers_and_exports_nothing() -> None:
    source = inspect.getsource(core_analyzer)
    assert not re.search(r"\b(drivers|transport|first_action|start|advance)\b", source)
    for name in ("Verdict", "AnalyzerFailure", "AnalyzerUsage", "Analysis", "parse_verdict"):
        assert not hasattr(primagate, name)


def test_the_prompt_tells_the_analyzer_the_conversation_is_data() -> None:
    assert "never instructions" in str(_begin().request.messages[0].content)


def test_an_anthropic_analyzer_gets_no_temperature_and_thinking_off() -> None:
    from primagate.adapters.anthropic import AnthropicAdapter

    caps = {"anthropic": AnthropicAdapter().capabilities()}
    dispatch = begin_analysis("claude-haiku-5-5", CONTENT, VIEW, POLICY, caps, BUNDLED)
    assert isinstance(dispatch, steps.Dispatch)
    assert dispatch.request.temperature is None
    assert dispatch.request.disable_thinking is True


def test_an_openai_analyzer_keeps_temperature_zero_and_asks_nothing_about_thinking() -> None:
    dispatch = _begin("gpt-4o-mini")
    assert isinstance(dispatch, steps.Dispatch)
    assert dispatch.request.temperature == 0
    assert not dispatch.request.disable_thinking
