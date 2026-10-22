from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from primagate.core.classify import Fate, classify
from primagate.core.fallbacks import fallback_candidates
from primagate.core.steps import (
    Action,
    Dispatch,
    DispatchOutcome,
    DropPolicy,
    Outcome,
    Plan,
    Raise,
    Random,
    RandomOutcome,
    Return,
    Route,
    Sleep,
    SleepOutcome,
)
from primagate.schema.errors import (
    AttemptsExhaustedError,
    BadRequestError,
    PrimaGateError,
    UnsupportedParamsError,
)
from primagate.schema.types import (
    Attempt,
    ChatCompletionRequest,
    DispatchRecord,
    ErrorInfo,
    Message,
    Metadata,
)
from primagate.schema.wire import (
    DEFAULT_RETRY,
    NO_FALLBACKS,
    Capabilities,
    Endpoint,
    Fallbacks,
    RetryPolicy,
)

DEFAULT_PROVIDER = "openai"
KNOWN_PROVIDERS = frozenset({"openai", "anthropic"})

# Behavior features a request can need, in the order errors list them, and the name a caller
# uses to opt in to dropping each.
_FEATURE_DROP_NAMES = {"tools": "tools", "structured_output": "response_format", "images": "image"}
_IMAGE_PART_TYPES = frozenset({"image_url", "image"})
DROPPABLE_FEATURES = frozenset({"tools", "response_format", "image"})
# Hint-type parameters in the order they are recorded, each with how to tell the caller set it.
_HINTS: dict[str, Callable[[ChatCompletionRequest], bool]] = {
    "n": lambda r: r.n is not None and r.n > 1,
    "seed": lambda r: r.seed is not None,
}


NO_DROPS = DropPolicy()


def parse_drop_params(value: Any) -> DropPolicy:
    if value is None or value is False:
        return DropPolicy()
    if value is True:
        return DropPolicy(hints=True)
    if isinstance(value, list | tuple) and all(
        isinstance(v, str) and v in DROPPABLE_FEATURES for v in value
    ):
        return DropPolicy(features=frozenset(value))
    allowed = ", ".join(f'"{n}"' for n in sorted(DROPPABLE_FEATURES))
    raise BadRequestError(f"drop_params must be True, False, None or a list drawn from {allowed}")


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


def _without_images(message: Message) -> Message:
    parts = [
        p
        for p in message.content or []
        if not (isinstance(p, Mapping) and p.get("type") in _IMAGE_PART_TYPES)
    ]
    # A message that held only images keeps its turn as empty text.
    return dataclasses.replace(message, content=parts or "")  # type: ignore[arg-type]


def apply_drops(
    request: ChatCompletionRequest, declared: Capabilities, policy: DropPolicy
) -> tuple[ChatCompletionRequest, list[str]]:
    """Remove what the policy allows and the adapter cannot honor; report what was removed."""
    dropped: list[str] = []
    changes: dict[str, Any] = {}
    if "tools" in policy.features and request.tools and not declared.tools:
        changes["tools"] = None
        dropped.append("tools")
    if "response_format" in policy.features and request.response_format:
        if not declared.structured_output:
            changes["response_format"] = None
            dropped.append("response_format")
    if "image" in policy.features and not declared.images:
        if any(_has_image(m.content) for m in request.messages):
            changes["messages"] = [
                _without_images(m) if _has_image(m.content) else m for m in request.messages
            ]
            dropped.append("image")
    if policy.hints:
        for name, is_set in _HINTS.items():
            if name in declared.unsupported_hints and is_set(request):
                changes[name] = None
                dropped.append(name)
    return dataclasses.replace(request, **changes), dropped


def _build(
    request: ChatCompletionRequest,
    provider: str,
    model: str,
    capabilities: Mapping[str, Capabilities],
    drops: DropPolicy,
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]],
    retry: RetryPolicy,
) -> Dispatch:
    """The first dispatch of one model's Attempt, or the reason it cannot be made."""
    declared = capabilities.get(provider)
    if declared is None:
        raise PrimaGateError(f"No adapter is registered for provider '{provider}'")
    request, dropped = apply_drops(request, declared, drops)
    missing = [f for f in required_features(request) if not getattr(declared, f)]
    if missing:
        names = [_FEATURE_DROP_NAMES[f] for f in missing]
        listed = ", ".join(f'"{n}"' for n in names)
        raise UnsupportedParamsError(
            f"{provider} does not support {', '.join(names)} for model '{model}'. "
            f"Pass drop_params=[{listed}] to drop them and continue without",
            llm_provider=provider,
        )
    redundant = routing.get((provider, model))
    return Dispatch(
        provider,
        dataclasses.replace(request, model=model),
        tuple(dropped),
        route=Route(redundant, retry) if redundant else None,
    )


def first_action(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    policy: DropPolicy = NO_DROPS,
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]] | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    fallbacks: Fallbacks = NO_FALLBACKS,
) -> Action:
    routes = routing or {}
    try:
        provider, model = resolve_model(request.model)
        first = _build(request, provider, model, capabilities, policy, routes, retry)
    except PrimaGateError as err:
        return Raise(err)
    later = fallback_candidates((provider, model), fallbacks)
    if not later:
        return first
    plan = Plan(request, capabilities, policy, routes, retry, later)
    return dataclasses.replace(first, plan=plan)


def _label(action: Dispatch) -> str:
    return action.endpoint.base_url if action.endpoint else action.provider


def _record(action: Dispatch, error: PrimaGateError | None = None) -> DispatchRecord:
    info = ErrorInfo(type(error).__name__, error.message, error.status_code) if error else None
    return DispatchRecord(_label(action), action.retries, info)


def attempt_chain(action: Dispatch, error: PrimaGateError | None = None) -> list[Attempt]:
    """The chain so far: the Attempts already over, then this one with a row per endpoint."""
    model = f"{action.provider}/{action.request.model}"
    return [*action.past, Attempt(model, [*action.history, _record(action, error)])]


def _delay(policy: RetryPolicy, retry: int, random_value: float) -> float:
    """Exponential backoff before retry number `retry` (1-based), with equal jitter.

    Half the delay is fixed so growth stays visible; the other half is spread by the caller's
    random value so simultaneous callers do not retry in lockstep.
    """
    # The exponent is bounded because 2.0 ** 1024 overflows before the cap can apply.
    nominal = min(policy.backoff_max, policy.backoff_base * 2.0 ** min(retry - 1, 62))
    return nominal / 2 + nominal / 2 * random_value


def count_attempts(chain: Sequence[Attempt]) -> int:
    """How many rows of `chain` are Attempts; a skipped candidate was never dispatched."""
    return sum(1 for row in chain if row.skipped is None)


def _next_attempt(
    action: Dispatch, plan: Plan, error: PrimaGateError
) -> tuple[Dispatch | None, list[Attempt]]:
    """Start the first remaining fallback that can take the request, and say which were passed.

    Each candidate that cannot take the request leaves a skipped row, using the same message
    the primary path would raise, so the two paths cannot drift apart.
    """
    chain = attempt_chain(action, error)
    skipped: list[Attempt] = []
    for index, (provider, model) in enumerate(plan.later):
        try:
            built = _build(
                plan.request,
                provider,
                model,
                plan.capabilities,
                plan.drops,
                plan.routing,
                plan.retry,
            )
        except PrimaGateError as reason:
            skipped.append(
                Attempt(f"{provider}/{model}", [], skipped=reason.message or type(reason).__name__)
            )
            continue
        rest = plan.later[index + 1 :]
        following = dataclasses.replace(
            built,
            past=(*chain, *skipped),
            plan=dataclasses.replace(plan, later=rest) if rest else None,
        )
        return following, skipped
    return None, skipped


def _exhausted(chain: list[Attempt], last: PrimaGateError) -> AttemptsExhaustedError:
    tried = [row.model for row in chain if row.skipped is None]
    passed = [row.model for row in chain if row.skipped is not None]
    parts = [f"{count_attempts(chain)} attempt(s) failed ({', '.join(tried)})"]
    if passed:
        parts.append(f"skipped: {', '.join(passed)}")
    parts.append(f"last error: {type(last).__name__}: {last.message}")
    return AttemptsExhaustedError("; ".join(parts), attempt_chain=chain, last_error=last)


def _after_failure(action: Dispatch, error: PrimaGateError) -> Action:
    fate = classify(error)
    route = action.route
    if route is not None:
        if fate is Fate.RETRY and action.retries < route.policy.max_retries:
            return Random(dataclasses.replace(action, retries=action.retries + 1))
        if fate in (Fate.RETRY, Fate.ADVANCE) and action.position < len(route.redundant):
            return dataclasses.replace(
                action,
                endpoint=route.redundant[action.position],
                position=action.position + 1,
                retries=0,
                history=(*action.history, _record(action, error)),
            )
    # The model's Attempt is over; only a RAISE-class error ends the whole chain.
    passed: list[Attempt] = []
    if fate in (Fate.RETRY, Fate.ADVANCE) and action.plan is not None:
        following, passed = _next_attempt(action, action.plan, error)
        if following is not None:
            return following
    chain = [*attempt_chain(action, error), *passed]
    error.attempt_chain = chain
    # One row is a single Attempt's story and the mapped error tells it; more than one row,
    # whether Attempts or candidates passed over, needs the whole chain to be understood.
    if len(chain) > 1:
        return Raise(_exhausted(chain, error))
    return Raise(error)


def next_action(action: Action, outcome: Outcome) -> Action:
    """Decide what follows `outcome`, the result of performing `action`."""
    if isinstance(action, Dispatch) and isinstance(outcome, DispatchOutcome):
        if outcome.error is not None:
            return _after_failure(action, outcome.error)
        if outcome.response is None:
            return Raise(PrimaGateError("Dispatch produced neither a response nor an error"))
        metadata = Metadata(
            dropped_params=list(outcome.dropped_params), attempt_chain=attempt_chain(action)
        )
        return Return(dataclasses.replace(outcome.response, metadata=metadata))
    if isinstance(action, Random) and isinstance(outcome, RandomOutcome):
        resume = action.resume
        policy = resume.route.policy if resume.route else DEFAULT_RETRY
        return Sleep(_delay(policy, resume.retries, outcome.value), resume)
    if isinstance(action, Sleep) and isinstance(outcome, SleepOutcome):
        return action.resume
    return Raise(
        PrimaGateError(f"{type(outcome).__name__} does not answer {type(action).__name__}")
    )
