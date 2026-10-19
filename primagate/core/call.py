from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from primagate.core.steps import Action, Dispatch, DispatchOutcome, Raise, Return
from primagate.schema.errors import BadRequestError, PrimaGateError, UnsupportedParamsError
from primagate.schema.types import ChatCompletionRequest
from primagate.schema.wire import Capabilities

DEFAULT_PROVIDER = "openai"
KNOWN_PROVIDERS = frozenset({"openai", "anthropic"})

# Behavior features a request can need, in the order errors list them, and the name a caller
# uses to opt in to dropping each.
_FEATURE_DROP_NAMES = {"tools": "tools", "structured_output": "response_format", "images": "image"}
_IMAGE_PART_TYPES = frozenset({"image_url", "image"})


def resolve_model(name: str) -> tuple[str, str]:
    """Split a model string into (provider, model).

    A bare `claude-*` name is Anthropic's; any other bare name is the default provider's, so
    OpenAI-compatible servers with unusual model names keep working.
    """
    prefix, sep, rest = name.partition("/")
    if not sep:
        return ("anthropic" if name.startswith("claude-") else DEFAULT_PROVIDER), name
    if prefix in KNOWN_PROVIDERS:
        return prefix, rest
    raise BadRequestError(f"Unknown provider prefix '{prefix}' in model '{name}'")


def required_features(request: ChatCompletionRequest) -> list[str]:
    """Behavior features the request cannot work without."""
    needed: list[str] = []
    # Tool history needs tool support even when no new tools are offered.
    if request.tools or any(m.tool_calls or m.role == "tool" for m in request.messages):
        needed.append("tools")
    if request.response_format:
        needed.append("structured_output")
    if any(_has_image(m.content) for m in request.messages):
        needed.append("images")
    return needed


def _has_image(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(part, Mapping) and part.get("type") in _IMAGE_PART_TYPES for part in content
    )


def first_action(
    request: ChatCompletionRequest, capabilities: Mapping[str, Capabilities]
) -> Action:
    try:
        provider, model = resolve_model(request.model)
    except PrimaGateError as err:
        return Raise(err)
    declared = capabilities.get(provider)
    if declared is None:
        return Raise(PrimaGateError(f"No adapter is registered for provider '{provider}'"))
    missing = [f for f in required_features(request) if not getattr(declared, f)]
    if missing:
        names = [_FEATURE_DROP_NAMES[f] for f in missing]
        listed = ", ".join(f'"{n}"' for n in names)
        return Raise(
            UnsupportedParamsError(
                f"{provider} does not support {', '.join(names)} for model '{model}'. "
                f"Pass drop_params=[{listed}] to drop them and continue without",
                llm_provider=provider,
            )
        )
    return Dispatch(provider, dataclasses.replace(request, model=model))


def next_action(outcome: DispatchOutcome) -> Action:
    if outcome.error is not None:
        return Raise(outcome.error)
    if outcome.response is None:
        return Raise(PrimaGateError("Dispatch produced neither a response nor an error"))
    return Return(outcome.response)
