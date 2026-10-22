from types import MappingProxyType

import pytest

from primagate.core import steps
from primagate.core.call import NO_DROPS, DropPolicy, first_action, next_action
from primagate.schema import errors
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    DispatchRecord,
    ErrorInfo,
    Message,
)
from primagate.schema.wire import Capabilities, Endpoint, Fallbacks, RetryPolicy

FULL = Capabilities(tools=True, structured_output=True, images=True)
BARE = Capabilities()
ALL = {"openai": FULL, "anthropic": FULL}
PRIMARY = ("openai", "gpt-4o")
FB1 = ("anthropic", "claude-sonnet-5-5")
FB2 = ("openai", "gpt-4o-mini")
SPARE = Endpoint("https://spare.test/v1")
POLICY = RetryPolicy(max_retries=1)
TOOL = {"type": "function", "function": {"name": "f"}}


def _lists(default: tuple[tuple[str, str], ...]) -> Fallbacks:
    return Fallbacks(MappingProxyType({}), default)


def _request(**fields: object) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hi")],
        **fields,  # type: ignore[arg-type]
    )


def _start(
    default: tuple[tuple[str, str], ...] = (FB1, FB2),
    *,
    caps: dict[str, Capabilities] | None = None,
    routing: dict | None = None,  # type: ignore[type-arg]
    request: ChatCompletionRequest | None = None,
    policy: DropPolicy = NO_DROPS,
) -> steps.Dispatch:
    action = first_action(
        request or _request(), caps or ALL, policy, routing, POLICY, _lists(default)
    )
    assert isinstance(action, steps.Dispatch)
    return action


def _fail(action: steps.Action, error: errors.PrimaGateError) -> steps.Action:
    return next_action(action, steps.DispatchOutcome(error=error))


def _down() -> errors.PrimaGateError:
    return errors.ServiceUnavailableError("down", status_code=503)


def _ok() -> ChatCompletion:
    message = Message(role="assistant", content="yo")
    return ChatCompletion(id="c", created=1, model="m", choices=[Choice(index=0, message=message)])


def test_a_failed_attempt_starts_the_next_fallback_as_its_own_attempt() -> None:
    nxt = _fail(_start(), _down())
    assert isinstance(nxt, steps.Dispatch)
    assert (nxt.provider, nxt.request.model) == FB1
    assert nxt.endpoint is None and nxt.retries == 0 and nxt.history == ()
    (past,) = nxt.past
    assert past.model == "openai/gpt-4o"
    assert past.dispatches == [
        DispatchRecord("openai", 0, ErrorInfo("ServiceUnavailableError", "down", 503))
    ]


def test_fallbacks_are_tried_in_order_and_the_last_error_is_raised_with_every_attempt() -> None:
    second = _fail(_start(), _down())
    last = errors.RateLimitError("slow", status_code=429)
    third = _fail(second, errors.Timeout("t"))
    assert isinstance(third, steps.Dispatch) and (third.provider, third.request.model) == FB2
    result = _fail(third, last)
    assert isinstance(result, steps.Raise) and result.error is last
    assert [a.model for a in last.attempt_chain] == [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-5-5",
        "openai/gpt-4o-mini",
    ]


def test_with_routing_off_a_retry_error_moves_on_without_retrying() -> None:
    nxt = _fail(_start(), errors.Timeout("t"))
    assert isinstance(nxt, steps.Dispatch)
    assert (nxt.provider, nxt.request.model) == FB1 and nxt.retries == 0


def test_an_advance_error_moves_to_the_next_fallback_at_once() -> None:
    nxt = _fail(_start(), errors.AuthenticationError("no", status_code=401))
    assert isinstance(nxt, steps.Dispatch) and nxt.provider == "anthropic"


@pytest.mark.parametrize("which", ["primary", "fallback"])
def test_a_raise_error_ends_the_chain(which: str) -> None:
    action: steps.Action = _start()
    if which == "fallback":
        action = _fail(action, _down())
    bad = errors.BadRequestError("bad", status_code=400)
    result = _fail(action, bad)
    assert isinstance(result, steps.Raise) and result.error is bad
    assert len(bad.attempt_chain) == (1 if which == "primary" else 2)


def test_the_primarys_routing_is_exhausted_before_the_first_fallback() -> None:
    routing = {PRIMARY: (SPARE,)}
    action: steps.Action = _start(routing=routing)
    action = _fail(action, errors.AuthenticationError("a"))
    assert isinstance(action, steps.Dispatch) and action.endpoint == SPARE
    action = _fail(action, errors.AuthenticationError("b"))
    assert isinstance(action, steps.Dispatch) and (action.provider, action.request.model) == FB1
    (past,) = action.past
    assert [d.endpoint for d in past.dispatches] == ["openai", SPARE.base_url]


def test_a_fallback_gets_its_own_routing_and_retries_before_the_next_entry() -> None:
    routing = {FB1: (SPARE,)}
    action = _fail(_start(routing=routing), _down())
    assert isinstance(action, steps.Dispatch) and action.route is not None
    random = _fail(action, _down())
    assert isinstance(random, steps.Random) and random.resume.retries == 1
    sleep = next_action(random, steps.RandomOutcome(1.0))
    retry = next_action(sleep, steps.SleepOutcome())
    assert isinstance(retry, steps.Dispatch)
    spare = _fail(retry, _down())
    assert isinstance(spare, steps.Dispatch) and spare.endpoint == SPARE
    assert len(spare.past) == 1 and len(spare.history) == 1
    after = _fail(spare, errors.AuthenticationError("a"))
    assert isinstance(after, steps.Dispatch) and (after.provider, after.request.model) == FB2
    assert len(after.past) == 2


def test_the_primary_without_routing_is_not_retried_even_when_a_fallback_has_it() -> None:
    action = _start(routing={FB1: (SPARE,)})
    assert action.route is None


def test_a_fallback_that_cannot_take_the_request_is_passed_over() -> None:
    caps = {"openai": FULL, "anthropic": BARE}
    action = _fail(_start(caps=caps, request=_request(tools=[TOOL])), _down())
    assert isinstance(action, steps.Dispatch) and (action.provider, action.request.model) == FB2
    assert len(action.past) == 1


def test_a_fallback_with_no_adapter_is_passed_over() -> None:
    action = _fail(_start(caps={"openai": FULL}), _down())
    assert isinstance(action, steps.Dispatch) and (action.provider, action.request.model) == FB2


def test_when_no_fallback_can_take_the_request_the_error_is_raised() -> None:
    caps = {"openai": FULL, "anthropic": BARE}
    error = _down()
    result = _fail(_start((FB1,), caps=caps, request=_request(tools=[TOOL])), error)
    assert isinstance(result, steps.Raise) and result.error is error
    assert len(error.attempt_chain) == 1


def test_drop_params_are_applied_against_each_fallbacks_own_provider() -> None:
    caps = {"openai": FULL, "anthropic": Capabilities(unsupported_hints=frozenset({"seed"}))}
    start = _start(caps=caps, request=_request(seed=7), policy=DropPolicy(hints=True))
    assert start.dropped_params == ()
    nxt = _fail(start, _down())
    assert isinstance(nxt, steps.Dispatch)
    assert nxt.request.seed is None and nxt.dropped_params == ("seed",)
    done = next_action(
        nxt, steps.DispatchOutcome(response=_ok(), dropped_params=nxt.dropped_params)
    )
    assert isinstance(done, steps.Return) and done.response.metadata is not None
    assert done.response.metadata.dropped_params == ["seed"]
    assert [a.model for a in done.response.metadata.attempt_chain] == [
        "openai/gpt-4o",
        "anthropic/claude-sonnet-5-5",
    ]


def test_no_fallbacks_behaves_as_before() -> None:
    error = _down()
    result = _fail(_start(()), error)
    assert isinstance(result, steps.Raise) and result.error is error
    assert len(error.attempt_chain) == 1


def test_the_primary_is_never_repeated_as_its_own_fallback() -> None:
    result = _fail(_start((PRIMARY, FB1)), _down())
    assert isinstance(result, steps.Dispatch) and result.provider == "anthropic"
