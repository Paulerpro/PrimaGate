import dataclasses

import pytest

from primagate.core import steps
from primagate.core.call import (
    DropPolicy,
    apply_drops,
    first_action,
    next_action,
    parse_drop_params,
    resolve_model,
)
from primagate.schema.errors import BadRequestError, RateLimitError, UnsupportedParamsError
from primagate.schema.events import CallCancelled
from primagate.schema.types import (
    Attempt,
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    DispatchRecord,
    ErrorInfo,
    FunctionCall,
    Message,
    Metadata,
    ToolCall,
)
from primagate.schema.wire import Capabilities

FULL = Capabilities(tools=True, structured_output=True, images=True)
BARE = Capabilities()
ALL = {"openai": FULL, "anthropic": FULL}


def _request(model: str = "gpt-4o") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


def _response() -> ChatCompletion:
    message = Message(role="assistant", content="yo")
    return ChatCompletion(
        id="c", created=1, model="gpt-4o", choices=[Choice(index=0, message=message)]
    )


def test_closed_action_set_is_exactly_the_actions_core_can_ask_for() -> None:
    names = {t.__name__ for t in steps.ACTION_TYPES}
    assert names == {
        "Dispatch",
        "Return",
        "Raise",
        "Random",
        "Sleep",
        "NewId",
        "Now",
        "Emit",
        "CountTokens",
        "CacheRead",
        "CacheWrite",
        "StoreRead",
        "StoreWrite",
        "Analyze",
        "RecordSpend",
        "StartPreparation",
        "PreparationDone",
        "Stopped",
    }


@pytest.mark.parametrize("name", ["gpt-4o", "openai/gpt-4o"])
def test_resolve_model_routes_to_openai(name: str) -> None:
    assert resolve_model(name) == ("openai", "gpt-4o")


@pytest.mark.parametrize("name", ["claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"])
def test_resolve_model_routes_claude_to_anthropic(name: str) -> None:
    assert resolve_model(name) == ("anthropic", "claude-sonnet-5-5")


def test_routing_and_driver_provider_tables_agree() -> None:
    from primagate.core.models import KNOWN_PROVIDERS
    from primagate.drivers.providers import BUILTIN_PROVIDERS

    assert KNOWN_PROVIDERS == set(BUILTIN_PROVIDERS)


def test_bare_non_claude_names_stay_on_openai() -> None:
    assert resolve_model("some-local-model") == ("openai", "some-local-model")


def test_resolve_model_rejects_unknown_provider_prefix() -> None:
    with pytest.raises(BadRequestError):
        resolve_model("nope/some-model")


def test_first_action_is_dispatch_with_stripped_model() -> None:
    action = first_action(_request("openai/gpt-4o"), ALL)
    assert isinstance(action, steps.Dispatch)
    assert action.provider == "openai"
    assert action.request.model == "gpt-4o"


def test_unknown_provider_becomes_raise_action() -> None:
    action = first_action(_request("nope/x"), ALL)
    assert isinstance(action, steps.Raise)
    assert isinstance(action.error, BadRequestError)


def _dispatch(provider: str = "openai", model: str = "gpt-4o") -> steps.Dispatch:
    return steps.Dispatch(provider, _request(model))


def test_success_outcome_returns_with_a_one_row_chain() -> None:
    action = next_action(_dispatch(), steps.DispatchOutcome(response=_response()))
    assert isinstance(action, steps.Return)
    assert action.response.metadata == Metadata(
        attempt_chain=[Attempt("openai/gpt-4o", [DispatchRecord("openai", 0)])]
    )


def test_error_outcome_raises_with_the_failing_dispatch_in_the_chain() -> None:
    err = RateLimitError("slow down", llm_provider="openai", status_code=429)
    action = next_action(_dispatch(), steps.DispatchOutcome(error=err))
    assert isinstance(action, steps.Raise)
    assert action.error is err
    assert err.attempt_chain == [
        Attempt(
            "openai/gpt-4o",
            [DispatchRecord("openai", 0, ErrorInfo("RateLimitError", "slow down", 429))],
        )
    ]


def test_an_error_before_any_dispatch_has_an_empty_chain() -> None:
    action = first_action(_request("nope/x"), ALL)
    assert isinstance(action, steps.Raise) and action.error.attempt_chain == []
    missing = first_action(_with(tools=[TOOL]), {"openai": BARE})
    assert isinstance(missing, steps.Raise) and missing.error.attempt_chain == []


def test_chain_rows_are_plain_data() -> None:
    err = RateLimitError("slow down", llm_provider="openai", status_code=429)
    next_action(_dispatch(), steps.DispatchOutcome(error=err))
    assert [a.to_dict() for a in err.attempt_chain] == [
        {
            "model": "openai/gpt-4o",
            "dispatches": [
                {
                    "endpoint": "openai",
                    "retries": 0,
                    "error": {"type": "RateLimitError", "message": "slow down", "status_code": 429},
                }
            ],
        }
    ]


def test_actions_are_immutable() -> None:
    action = steps.Return(_response())
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.response = _response()  # type: ignore[misc]


TOOL = {"type": "function", "function": {"name": "f"}}
IMAGE_MESSAGE = Message(
    role="user",
    content=[{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": "u"}}],  # type: ignore[arg-type]
)


def _with(**changes: object) -> ChatCompletionRequest:
    return dataclasses.replace(_request(), **changes)  # type: ignore[arg-type]


def _raised(request: ChatCompletionRequest, capabilities: Capabilities) -> UnsupportedParamsError:
    action = first_action(request, {"openai": capabilities})
    assert isinstance(action, steps.Raise)
    assert isinstance(action.error, UnsupportedParamsError)
    assert action.error.llm_provider == "openai"
    return action.error


def test_unsupported_tools_raise_naming_the_feature_and_fix() -> None:
    error = _raised(_with(tools=[TOOL]), BARE)
    assert "tools" in error.message and 'drop_params=["tools"]' in error.message


def test_tool_history_alone_needs_tools() -> None:
    call = ToolCall(id="c", function=FunctionCall(name="f", arguments="{}"))
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=[call]),
        Message(role="tool", tool_call_id="c", content="x"),
    ]
    assert "tools" in _raised(_with(messages=history), BARE).message
    only_result = [Message(role="tool", tool_call_id="c", content="x")]
    assert "tools" in _raised(_with(messages=only_result), BARE).message


def test_unsupported_structured_output_names_response_format() -> None:
    error = _raised(_with(response_format={"type": "json_object"}), BARE)
    assert 'drop_params=["response_format"]' in error.message


def test_unsupported_images_name_image() -> None:
    error = _raised(_with(messages=[IMAGE_MESSAGE]), BARE)
    assert 'drop_params=["image"]' in error.message


def test_every_missing_feature_is_named_in_one_error() -> None:
    error = _raised(_with(tools=[TOOL], messages=[IMAGE_MESSAGE]), BARE)
    assert "tools" in error.message and "image" in error.message


def test_text_only_list_content_is_not_an_image() -> None:
    text = Message(role="user", content=[{"type": "text", "text": "hi"}])  # type: ignore[arg-type]
    assert isinstance(first_action(_with(messages=[text]), {"openai": BARE}), steps.Dispatch)


def test_supported_features_dispatch_unchanged() -> None:
    request = _with(tools=[TOOL], response_format={"type": "json_object"}, messages=[IMAGE_MESSAGE])
    action = first_action(request, {"openai": FULL})
    assert isinstance(action, steps.Dispatch) and action.request.tools == [TOOL]


def test_check_uses_the_target_providers_capabilities() -> None:
    request = _with(model="claude-sonnet-5-5", tools=[TOOL])
    capabilities = {"openai": FULL, "anthropic": BARE}
    action = first_action(request, capabilities)
    assert isinstance(action, steps.Raise)
    assert action.error.llm_provider == "anthropic"


def test_provider_without_an_adapter_raises_a_gate_error() -> None:
    action = first_action(_request(), {})
    assert isinstance(action, steps.Raise) and "openai" in action.error.message


@pytest.mark.parametrize(
    ("value", "policy"),
    [
        (None, DropPolicy()),
        (False, DropPolicy()),
        ([], DropPolicy()),
        (True, DropPolicy(hints=True)),
        (["tools", "image"], DropPolicy(features=frozenset({"tools", "image"}))),
    ],
)
def test_parse_drop_params_accepts(value: object, policy: DropPolicy) -> None:
    assert parse_drop_params(value) == policy


@pytest.mark.parametrize("value", ["tools", ["nope"], [1], 3, {"tools"}, ["tools", None]])
def test_parse_drop_params_rejects(value: object) -> None:
    with pytest.raises(BadRequestError, match="drop_params"):
        parse_drop_params(value)


def test_apply_drops_leaves_the_callers_request_alone() -> None:
    request = _with(tools=[TOOL], messages=[IMAGE_MESSAGE])
    policy = DropPolicy(features=frozenset({"tools", "image"}))
    changed, dropped = apply_drops(request, BARE, policy, "gpt-4o")
    assert dropped == ["tools", "image"]
    assert request.tools == [TOOL] and request.messages == [IMAGE_MESSAGE]
    assert changed.tools is None and changed.messages[0].content == [
        {"type": "text", "text": "what?"}
    ]


def test_apply_drops_only_hints_the_adapter_declares() -> None:
    request = _with(n=3, seed=7)
    only_seed = Capabilities(unsupported_hints=frozenset({"seed"}))
    changed, dropped = apply_drops(request, only_seed, DropPolicy(hints=True), "gpt-4o")
    assert dropped == ["seed"] and changed.n == 3 and changed.seed is None
    assert apply_drops(request, only_seed, DropPolicy(), "gpt-4o")[1] == []


def test_dispatch_records_what_was_dropped_and_the_response_names_it() -> None:
    action = first_action(
        _with(tools=[TOOL]), {"openai": BARE}, DropPolicy(features=frozenset({"tools"}))
    )
    assert isinstance(action, steps.Dispatch) and action.dropped_params == ("tools",)
    outcome = steps.DispatchOutcome(response=_response(), dropped_params=action.dropped_params)
    returned = next_action(action, outcome)
    assert isinstance(returned, steps.Return)
    assert returned.response.metadata is not None
    assert returned.response.metadata.dropped_params == ["tools"]
    clean = next_action(action, steps.DispatchOutcome(response=_response()))
    assert isinstance(clean, steps.Return) and clean.response.metadata is not None
    assert clean.response.metadata.dropped_params == []


# --- cancellation ---

CANCEL_POLICY = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
    "ambiguous_context_default": "SMALL",
    "spending_ceiling": 100.0,
}


def _tiers(**kwargs: object) -> object:
    from primagate import PrimaGate

    gate = PrimaGate(
        tier_bindings={"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}},
        policy=CANCEL_POLICY,
        analyzer="gpt-4o-mini",
        **kwargs,  # type: ignore[arg-type]
    )
    return gate._tier_routing


def _verdict() -> ChatCompletion:
    from primagate.schema.types import Usage

    message = Message(role="assistant", content="SMALL 0.9")
    return ChatCompletion(
        id="a",
        created=1,
        model="gpt-4o-mini",
        choices=[Choice(index=0, message=message)],
        usage=Usage(10, 2, 12),
    )


def _outcome(action: steps.Action) -> steps.Outcome:
    """A plain answer to every action a call can ask for, so a whole call can be walked."""
    from primagate.schema.profile import PrepState

    if isinstance(action, steps.NewId):
        return steps.IdOutcome("req-1")
    if isinstance(action, steps.CacheRead):
        return steps.CacheReadOutcome(None)
    if isinstance(action, steps.Emit):
        return steps.EmitOutcome()
    if isinstance(action, steps.Analyze):
        return steps.DispatchOutcome(response=_verdict())
    if isinstance(action, steps.Dispatch):
        # No usage, so the call goes on to count the tokens itself.
        return steps.DispatchOutcome(response=_response())
    if isinstance(action, steps.CountTokens):
        return steps.CountOutcome(5, 5, heuristic=False)
    if isinstance(action, steps.CacheWrite):
        return steps.CacheWriteOutcome()
    if isinstance(action, steps.RecordSpend):
        return steps.SpendOutcome(action.amount, crossed=False)
    if isinstance(action, steps.StartPreparation):
        return steps.StartOutcome(True, PrepState.PREPARING)
    raise AssertionError(f"unexpected {type(action).__name__}")


def _walk(first: steps.Action) -> list[steps.Action]:
    """Every action of one call, up to the one that ends it."""
    from primagate.core.call import advance

    seen = [first]
    while not isinstance(seen[-1], steps.Return | steps.Raise):
        seen.append(advance(seen[-1], _outcome(seen[-1])))
    return seen


def _stopped(request_id: str) -> steps.Action:
    return steps.Emit((CallCancelled(request_id),), steps.Stopped())


def test_cancelled_is_an_outcome_and_stopped_a_terminal_action() -> None:
    assert steps.Stopped in steps.ACTION_TYPES
    assert steps.Cancelled() == steps.Cancelled()


def test_a_call_cancelled_before_it_has_an_id_reports_unknown() -> None:
    from primagate.core.call import advance, start

    first = start(_request(), ALL)
    assert isinstance(first, steps.NewId)
    assert advance(first, steps.Cancelled()) == _stopped("unknown")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="plain"),
        pytest.param({"use_cache": False}, id="no-cache"),
        pytest.param({"tier_routing": "analyzer"}, id="routed"),
        pytest.param({"tier_routing": "preparing"}, id="routed-preparing"),
    ],
)
def test_every_boundary_of_a_call_stops_with_one_event_carrying_its_id(
    kwargs: dict[str, object],
) -> None:
    from primagate.core.call import start
    from primagate.schema.policy import TIER_ROUTING_ALIAS

    request = _request()
    routed = kwargs.pop("tier_routing", None)
    if routed is not None:
        request = _request(TIER_ROUTING_ALIAS)
        sources = {"instructions": "be brief"} if routed == "preparing" else None
        kwargs["tier_routing"] = _tiers(profile_sources=sources)
    walked = _walk(start(request, ALL, **kwargs))  # type: ignore[arg-type]
    assert isinstance(walked[-1], steps.Return)
    kinds = {type(a) for a in walked}
    if routed == "analyzer":
        assert {steps.Analyze, steps.RecordSpend, steps.CacheWrite} <= kinds
    if routed == "preparing":
        assert steps.StartPreparation in kinds and steps.Analyze not in kinds
    assert steps.CountTokens in kinds
    asked = _first(walked, steps.Analyze)
    counted = _first(walked, steps.CountTokens)
    recorded = _first(walked, steps.RecordSpend)
    if recorded == len(walked):
        # Without a ceiling nothing is ever recorded.
        recorded = -1
    for index, action in enumerate(walked[1:-1], start=1):
        name = type(action).__name__
        announced, spent = _stop(action)
        assert announced == [CallCancelled("req-1")], name
        if counted < index <= recorded:
            # The worker's answer is counted and priced: what the finished call would record.
            record = walked[recorded]
            assert isinstance(record, steps.RecordSpend)
            assert spent == [record.amount], name
        elif asked < index <= counted:
            # Only the Analyzer has been billed so far.
            worker = next(a for a in walked if type(a) is steps.Dispatch)
            assert isinstance(worker, steps.Dispatch) and worker.analyzer is not None
            assert worker.analyzer.cost is not None
            assert spent == [worker.analyzer.cost.total], name
        else:
            assert spent == [], name
    if routed == "analyzer":
        assert counted - asked > 1


def _first(walked: list[steps.Action], kind: type) -> int:
    return next((i for i, a in enumerate(walked) if isinstance(a, kind)), len(walked))


def _stop(action: steps.Action) -> tuple[list[object], list[float]]:
    """Follow a cancellation to its end: what it announces and what it counts."""
    from primagate.core.call import advance

    announced: list[object] = []
    spent: list[float] = []
    following = advance(action, steps.Cancelled())
    while not isinstance(following, steps.Stopped):
        if isinstance(following, steps.Emit):
            announced += following.events
            following = advance(following, steps.EmitOutcome())
        else:
            assert isinstance(following, steps.RecordSpend)
            spent.append(following.amount)
            following = advance(following, steps.SpendOutcome(following.amount, crossed=False))
    return announced, spent


def test_a_cancelled_answer_is_not_cached_counted_or_spent() -> None:
    from primagate.core.call import advance, start

    walked = _walk(start(_request(), ALL))
    write = next(a for a in walked if isinstance(a, steps.CacheWrite))
    stopped = advance(write, steps.Cancelled())
    assert isinstance(stopped, steps.Emit) and isinstance(stopped.resume, steps.Stopped)


def test_a_cancelled_retry_or_fallback_goes_no_further() -> None:
    from primagate.core.call import advance
    from primagate.schema.errors import AuthenticationError
    from primagate.schema.wire import Endpoint, Fallbacks

    redundant = {("openai", "gpt-4o"): (Endpoint("https://b.test/v1"),)}
    fallbacks = Fallbacks({("openai", "gpt-4o"): (("anthropic", "claude-sonnet-5-5"),)})
    first = first_action(_request(), ALL, routing=redundant, fallbacks=fallbacks)
    assert isinstance(first, steps.Dispatch)
    first = dataclasses.replace(first, request_id="req-1")
    retry = advance(first, steps.DispatchOutcome(error=RateLimitError("slow")))
    assert isinstance(retry, steps.Emit) and isinstance(retry.resume, steps.Random)
    wait = advance(retry.resume, steps.RandomOutcome(0.5))
    assert isinstance(wait, steps.Emit) and isinstance(wait.resume, steps.Sleep)
    for action in (retry.resume, wait.resume, wait):
        assert advance(action, steps.Cancelled()) == _stopped("req-1")
    # The last endpoint refuses the key, so the next fallback is due; cancelling stops it.
    last = dataclasses.replace(first, endpoint=redundant[("openai", "gpt-4o")][0], position=1)
    moved = advance(last, steps.DispatchOutcome(error=AuthenticationError("bad key")))
    assert isinstance(moved, steps.Emit) and isinstance(moved.resume, steps.Dispatch)
    assert moved.resume.provider == "anthropic"
    for action in (moved, moved.resume):
        assert advance(action, steps.Cancelled()) == _stopped("req-1")
