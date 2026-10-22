from types import MappingProxyType

import pytest

from primagate.core import steps
from primagate.core.call import NO_DROPS, first_action, next_action
from primagate.schema import errors
from primagate.schema.types import ChatCompletionRequest, Message
from primagate.schema.wire import Capabilities, Endpoint, Fallbacks, RetryPolicy

FULL = Capabilities(tools=True, structured_output=True, images=True)
BARE = Capabilities()
PRIMARY = ("openai", "gpt-4o")
FB1 = ("anthropic", "claude-sonnet-5-5")
FB2 = ("openai", "gpt-4o-mini")
SPARE = Endpoint("https://spare.test/v1")
TOOL = {"type": "function", "function": {"name": "f"}}


def _start(
    fallbacks: tuple[tuple[str, str], ...] = (FB1,),
    *,
    caps: dict[str, Capabilities] | None = None,
    routing: dict | None = None,  # type: ignore[type-arg]
    tools: bool = False,
) -> steps.Dispatch:
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hi")],
        tools=[TOOL] if tools else None,
    )
    lists = Fallbacks(MappingProxyType({}), fallbacks)
    action = first_action(
        request,
        caps or {"openai": FULL, "anthropic": FULL},
        NO_DROPS,
        routing,
        RetryPolicy(max_retries=0),
        lists,
    )
    assert isinstance(action, steps.Dispatch)
    return action


def _fail(action: steps.Action, error: errors.PrimaGateError) -> steps.Action:
    return next_action(action, steps.DispatchOutcome(error=error))


def _down() -> errors.PrimaGateError:
    return errors.ServiceUnavailableError("down", llm_provider="openai", status_code=503)


def _aggregate(result: steps.Action) -> errors.AttemptsExhaustedError:
    assert isinstance(result, steps.Raise)
    assert isinstance(result.error, errors.AttemptsExhaustedError)
    return result.error


def test_it_is_a_primagate_error() -> None:
    assert issubclass(errors.AttemptsExhaustedError, errors.PrimaGateError)


def test_two_failed_attempts_raise_the_aggregate_with_both_in_it() -> None:
    second = _fail(_start(), _down())
    last = errors.RateLimitError("slow", llm_provider="anthropic", status_code=429)
    error = _aggregate(_fail(second, last))
    assert [a.model for a in error.attempts] == ["openai/gpt-4o", "anthropic/claude-sonnet-5-5"]
    assert error.attempt_chain == last.attempt_chain == error.attempts
    first_row = error.attempts[0].dispatches[0].error
    assert first_row is not None and first_row.status_code == 503
    assert error.last_error is last and error.__cause__ is last
    assert (error.llm_provider, error.status_code) == ("anthropic", 429)


def test_a_raise_class_error_that_ends_the_chain_still_gives_the_aggregate() -> None:
    last = errors.BadRequestError("bad", status_code=400)
    error = _aggregate(_fail(_fail(_start(), _down()), last))
    assert error.last_error is last and len(error.attempts) == 2


def test_the_message_names_the_models_and_the_last_error() -> None:
    last = errors.RateLimitError("slow down", status_code=429)
    error = _aggregate(_fail(_fail(_start(), _down()), last))
    for part in ("openai/gpt-4o", "anthropic/claude-sonnet-5-5", "RateLimitError", "slow down"):
        assert part in error.message
    assert str(error) == error.message


def test_one_attempt_with_several_endpoints_stays_the_plain_error() -> None:
    action: steps.Action = _start((), routing={PRIMARY: (SPARE,)})
    action = _fail(action, errors.AuthenticationError("a"))
    last = errors.AuthenticationError("b")
    result = _fail(action, last)
    assert isinstance(result, steps.Raise) and result.error is last
    assert len(last.attempt_chain) == 1 and len(last.attempt_chain[0].dispatches) == 2


def test_no_fallbacks_means_the_plain_error() -> None:
    error = _down()
    result = _fail(_start(()), error)
    assert isinstance(result, steps.Raise) and result.error is error


def test_a_lone_raise_class_error_is_plain_even_with_fallbacks_declared() -> None:
    bad = errors.BadRequestError("bad", status_code=400)
    result = _fail(_start(), bad)
    assert isinstance(result, steps.Raise) and result.error is bad


def test_one_attempt_followed_only_by_skipped_candidates_gives_the_aggregate() -> None:
    caps = {"openai": FULL, "anthropic": BARE}
    last = _down()
    error = _aggregate(_fail(_start(caps=caps, tools=True), last))
    assert len(error.attempts) == 1 and len(error.attempt_chain) == 2
    assert error.attempt_chain[1].skipped is not None
    assert "anthropic/claude-sonnet-5-5" in error.message and "skipped" in error.message
    assert error.last_error is last


def test_a_failure_before_any_dispatch_is_plain_with_an_empty_chain() -> None:
    caps = {"openai": BARE, "anthropic": FULL}
    request = ChatCompletionRequest(
        model="gpt-4o", messages=[Message(role="user", content="hi")], tools=[TOOL]
    )
    result = first_action(
        request, caps, NO_DROPS, None, RetryPolicy(), Fallbacks(MappingProxyType({}), (FB1,))
    )
    assert isinstance(result, steps.Raise)
    assert type(result.error) is errors.UnsupportedParamsError
    assert result.error.attempt_chain == []


def test_the_message_holds_no_key() -> None:
    last = errors.AuthenticationError("bad key sk-secret", status_code=401).redact(["sk-secret"])
    error = _aggregate(_fail(_fail(_start(), _down()), last))
    assert "sk-secret" not in error.message


@pytest.mark.parametrize("attribute", ["attempts", "attempt_chain", "last_error"])
def test_the_documented_attributes_exist(attribute: str) -> None:
    error = _aggregate(_fail(_fail(_start(), _down()), _down()))
    assert hasattr(error, attribute)


def test_the_aggregate_survives_pickle_and_copy() -> None:
    import copy
    import pickle

    error = _aggregate(_fail(_fail(_start(), _down()), errors.RateLimitError("slow")))
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error), copy.deepcopy(error)):
        assert isinstance(clone, errors.AttemptsExhaustedError)
        assert clone.message == error.message and str(clone) == str(error)
        assert clone.attempt_chain == error.attempt_chain
        assert clone.attempts == error.attempts
        assert type(clone.last_error) is errors.RateLimitError
