import dataclasses

import pytest

from primagate.core import steps
from primagate.core.call import first_action, next_action, resolve_model
from primagate.schema.errors import BadRequestError, RateLimitError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Choice, Message


def _request(model: str = "gpt-4o") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[Message(role="user", content="hi")])


def _response() -> ChatCompletion:
    message = Message(role="assistant", content="yo")
    return ChatCompletion(
        id="c", created=1, model="gpt-4o", choices=[Choice(index=0, message=message)]
    )


def test_closed_action_set_is_dispatch_return_raise() -> None:
    names = {t.__name__ for t in steps.ACTION_TYPES}
    assert names == {"Dispatch", "Return", "Raise"}


@pytest.mark.parametrize("name", ["gpt-4o", "openai/gpt-4o"])
def test_resolve_model_routes_to_openai(name: str) -> None:
    assert resolve_model(name) == ("openai", "gpt-4o")


@pytest.mark.parametrize("name", ["claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"])
def test_resolve_model_routes_claude_to_anthropic(name: str) -> None:
    assert resolve_model(name) == ("anthropic", "claude-sonnet-5-5")


def test_routing_and_driver_provider_tables_agree() -> None:
    from primagate.core.call import KNOWN_PROVIDERS
    from primagate.drivers.providers import BUILTIN_PROVIDERS

    assert KNOWN_PROVIDERS == set(BUILTIN_PROVIDERS)


def test_bare_non_claude_names_stay_on_openai() -> None:
    assert resolve_model("some-local-model") == ("openai", "some-local-model")


def test_resolve_model_rejects_unknown_provider_prefix() -> None:
    with pytest.raises(BadRequestError):
        resolve_model("nope/some-model")


def test_first_action_is_dispatch_with_stripped_model() -> None:
    action = first_action(_request("openai/gpt-4o"))
    assert isinstance(action, steps.Dispatch)
    assert action.provider == "openai"
    assert action.request.model == "gpt-4o"


def test_unknown_provider_becomes_raise_action() -> None:
    action = first_action(_request("nope/x"))
    assert isinstance(action, steps.Raise)
    assert isinstance(action.error, BadRequestError)


def test_success_outcome_returns() -> None:
    resp = _response()
    action = next_action(steps.DispatchOutcome(response=resp))
    assert action == steps.Return(resp)


def test_error_outcome_raises() -> None:
    err = RateLimitError("slow down", llm_provider="openai", status_code=429)
    action = next_action(steps.DispatchOutcome(error=err))
    assert isinstance(action, steps.Raise)
    assert action.error is err


def test_actions_are_immutable() -> None:
    action = steps.Return(_response())
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.response = _response()  # type: ignore[misc]
