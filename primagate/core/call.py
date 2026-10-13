from __future__ import annotations

import dataclasses

from primagate.core.steps import Action, Dispatch, DispatchOutcome, Raise, Return
from primagate.schema.errors import BadRequestError, PrimaGateError
from primagate.schema.types import ChatCompletionRequest

DEFAULT_PROVIDER = "openai"
KNOWN_PROVIDERS = frozenset({"openai"})


def resolve_model(name: str) -> tuple[str, str]:
    """Split a model string into (provider, model). A bare name means the default provider."""
    prefix, sep, rest = name.partition("/")
    if not sep:
        return DEFAULT_PROVIDER, name
    if prefix in KNOWN_PROVIDERS:
        return prefix, rest
    raise BadRequestError(f"Unknown provider prefix '{prefix}' in model '{name}'")


def first_action(request: ChatCompletionRequest) -> Action:
    try:
        provider, model = resolve_model(request.model)
    except PrimaGateError as err:
        return Raise(err)
    return Dispatch(provider, dataclasses.replace(request, model=model))


def next_action(outcome: DispatchOutcome) -> Action:
    if outcome.error is not None:
        return Raise(outcome.error)
    if outcome.response is None:
        return Raise(PrimaGateError("Dispatch produced neither a response nor an error"))
    return Return(outcome.response)
