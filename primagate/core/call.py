from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from primagate.core.cache import cache_key
from primagate.core.classify import Fate, classify
from primagate.core.counting import completion_text, prompt_text
from primagate.core.fallbacks import fallback_candidates
from primagate.core.prices import PriceTable, estimate
from primagate.core.steps import (
    Action,
    CacheRead,
    CacheReadOutcome,
    CacheWrite,
    CacheWriteOutcome,
    CountOutcome,
    CountTokens,
    Dispatch,
    DispatchOutcome,
    DropPolicy,
    Emit,
    EmitOutcome,
    IdOutcome,
    NewId,
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
from primagate.core.tiers import narrowed
from primagate.schema import events
from primagate.schema.errors import (
    AttemptsExhaustedError,
    BadRequestError,
    PrimaGateError,
    UnsupportedParamsError,
)
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.types import (
    Attempt,
    ChatCompletion,
    ChatCompletionRequest,
    DispatchRecord,
    ErrorInfo,
    Message,
    Metadata,
    Usage,
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


def _route_tier(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    drops: DropPolicy,
    routes: Mapping[tuple[str, str], tuple[Endpoint, ...]],
    retry: RetryPolicy,
    tiers: TierRouting | None,
) -> Dispatch:
    """The Compatibility Filter for Tier Routing, and the dispatch when only one Tier can serve.

    Each enabled Tier goes through `_build`, the check the plain path and the Fallback List use
    too, with the adapter's capabilities narrowed by that Tier's binding, so all three agree on
    what a model can do and on dropping before filtering.
    """
    if tiers is None:
        raise BadRequestError(
            f"model '{TIER_ROUTING_ALIAS}' is the Tier Routing alias; "
            "declare tier_bindings and a policy before using it"
        )
    able: list[Dispatch] = []
    lacking: list[str] = []
    for tier in tiers.policy.enabled_tiers:
        binding = tiers.bindings[tier]
        provider, model = binding.provider, binding.model_id
        declared = capabilities.get(provider)
        seen_by_tier = (
            capabilities
            if declared is None
            else {
                **capabilities,
                provider: narrowed(declared, binding),
            }
        )
        try:
            able.append(_build(request, provider, model, seen_by_tier, drops, routes, retry))
        except UnsupportedParamsError as reason:
            # Only a missing feature means "this Tier cannot take it"; anything else is a fault
            # to report as it is, not to hide behind a feature message.
            lacking.append(f"{tier.name} ({provider}/{model}): {reason.message}")
    # After the filter, so a model removed by one Tier's limits can still serve under another's.
    first_per_model: dict[tuple[str, str], Dispatch] = {}
    for built in able:
        first_per_model.setdefault((built.provider, built.request.model), built)
    unique = list(first_per_model.values())
    if not unique:
        raise UnsupportedParamsError("No enabled Tier can take this request. " + "; ".join(lacking))
    if len(unique) > 1:
        names = ", ".join(f"{d.provider}/{d.request.model}" for d in unique)
        raise PrimaGateError(
            f"{len(unique)} Tiers can take this request ({names}); choosing between them needs "
            "the Analyzer, which is not part of PrimaGate yet"
        )
    return unique[0]


def first_action(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    policy: DropPolicy = NO_DROPS,
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]] | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    fallbacks: Fallbacks = NO_FALLBACKS,
    tier_routing: TierRouting | None = None,
) -> Action:
    routes = routing or {}
    try:
        if request.model == TIER_ROUTING_ALIAS:
            return _route_tier(request, capabilities, policy, routes, retry, tier_routing)
        provider, model = resolve_model(request.model)
        first = _build(request, provider, model, capabilities, policy, routes, retry)
    except PrimaGateError as err:
        return Raise(err)
    later = fallback_candidates((provider, model), fallbacks)
    if not later:
        return first
    plan = Plan(request, capabilities, policy, routes, retry, later)
    return dataclasses.replace(first, plan=plan)


def _record(action: Dispatch, error: PrimaGateError | None = None) -> DispatchRecord:
    info = ErrorInfo(type(error).__name__, error.message, error.status_code) if error else None
    return DispatchRecord(_shown(action), action.retries, info)


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
            request_id=action.request_id,
            payload_retention=action.payload_retention,
            prices=action.prices,
            cache_key=action.cache_key,
            no_cache=action.no_cache,
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
        priced = estimate(
            action.provider, action.request.model, outcome.response.usage, action.prices
        )
        metadata = Metadata(
            dropped_params=list(outcome.dropped_params),
            attempt_chain=attempt_chain(action),
            token_counts=priced.token_counts,
            cost=priced.cost,
            unknown_model=priced.unknown_model,
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


def start(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    policy: DropPolicy = NO_DROPS,
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]] | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    fallbacks: Fallbacks = NO_FALLBACKS,
    payload_retention: bool = False,
    prices: PriceTable | None = None,
    use_cache: bool = True,
    cache_salt: bytes = b"",
    tier_routing: TierRouting | None = None,
) -> Action:
    """The first action of a call: ask for its request id, then announce and begin.

    Content reaches an event only when `payload_retention` is true; with false it is never built,
    so no later stage can leak it.
    """

    # A stream never touches the Cache, and a call may opt out of it.
    cacheable = use_cache and not request.stream

    def begin(request_id: str, key: str | None = None, read_already: bool = False) -> Action:
        first = first_action(request, capabilities, policy, routing, retry, fallbacks, tier_routing)
        if isinstance(first, Dispatch):
            first = dataclasses.replace(
                first, request_id=request_id, payload_retention=payload_retention
            )
            if prices is not None:
                first = dataclasses.replace(first, prices=prices)
            if not cacheable:
                return Emit((_started(first),), first)
            if not read_already:
                # Read only once the request is known to be valid, so an invalid one still raises.
                key = cache_key(request, policy, cache_salt)
            if key is None:
                return Emit((_started(first),), first)
            first = dataclasses.replace(first, cache_key=key)
            if read_already:
                return Emit((_started(first),), first)
            return CacheRead(key, lambda read: _after_read(first, read))
        if isinstance(first, Raise):
            return Emit((_call_failed(request_id, first.error),), first)
        return first

    def begin_routed(request_id: str) -> Action:
        # A hit returns before the Tier is chosen, so it never depends on the filter or, later, on
        # an Analyzer. Anything a past answer was cached for would pass the filter again.
        key = cache_key(request, policy, cache_salt)
        if key is None:
            return begin(request_id)

        def after(read: CacheReadOutcome) -> Action:
            if read.response is None:
                return begin(request_id, key, read_already=True)
            hit = dataclasses.replace(read.response, metadata=Metadata(cache_hit=True))
            return Emit((events.CacheHit(request_id, request.model, key),), Return(hit))

        return CacheRead(key, after)

    routed = request.model == TIER_ROUTING_ALIAS and tier_routing is not None and cacheable
    return NewId(begin_routed if routed else begin)


def advance(action: Action, outcome: Outcome) -> Action:
    """Like `next_action`, and also says what happened so the driver can deliver the events."""
    if isinstance(action, NewId) and isinstance(outcome, IdOutcome):
        return action.resume(outcome.value)
    if isinstance(action, Emit) and isinstance(outcome, EmitOutcome):
        return action.resume
    if isinstance(action, CountTokens) and isinstance(outcome, CountOutcome):
        return action.resume(outcome)
    if isinstance(action, CacheRead) and isinstance(outcome, CacheReadOutcome):
        return action.resume(outcome)
    if isinstance(action, CacheWrite) and isinstance(outcome, CacheWriteOutcome):
        return action.resume
    following = next_action(action, outcome)
    if (
        isinstance(action, Dispatch)
        and isinstance(outcome, DispatchOutcome)
        and isinstance(following, Return)
        and following.response.usage is None
    ):
        answered = following.response
        return CountTokens(
            action.request.model,
            prompt_text(action.request),
            completion_text(answered),
            lambda counted: _announced(action, outcome, _counted(action, answered, counted)),
        )
    return _announced(action, outcome, following)


def _announced(action: Action, outcome: Outcome, following: Action) -> Action:
    announced = _announce(action, outcome, following)
    after = _written(action, following)
    return Emit(announced, after) if announced else after


def _written(action: Action, following: Action) -> Action:
    """Store a final answer before it is returned, so the next identical call can reuse it."""
    if (
        isinstance(action, Dispatch)
        and action.cache_key
        and not action.no_cache
        and isinstance(following, Return)
    ):
        return CacheWrite(action.cache_key, following.response, following)
    return following


def _after_read(first: Dispatch, read: CacheReadOutcome) -> Action:
    """A hit is returned as a copy that says it was not dispatched; a miss goes on to dispatch."""
    if read.response is None:
        return Emit((_started(first),), first)
    hit = dataclasses.replace(read.response, metadata=Metadata(cache_hit=True))
    announced = events.CacheHit(first.request_id, _model(first), first.cache_key or "")
    return Emit((announced,), Return(hit))


def _counted(action: Dispatch, answered: ChatCompletion, counted: CountOutcome) -> Return:
    """The response with counts the SDK made itself, and the cost priced from them."""
    usage = Usage(
        counted.prompt_tokens,
        counted.completion_tokens,
        counted.prompt_tokens + counted.completion_tokens,
    )
    priced = estimate(
        action.provider, action.request.model, usage, action.prices, heuristic=counted.heuristic
    )
    metadata = dataclasses.replace(
        answered.metadata or Metadata(),
        token_counts=priced.token_counts,
        cost=priced.cost,
        unknown_model=priced.unknown_model,
    )
    return Return(dataclasses.replace(answered, metadata=metadata))


def _request_id(action: Action) -> str:
    if isinstance(action, Dispatch):
        return action.request_id
    if isinstance(action, Random | Sleep):
        return action.resume.request_id
    return ""


def _model(action: Dispatch) -> str:
    return f"{action.provider}/{action.request.model}"


def _shown(action: Dispatch) -> str:
    """The endpoint as events, logs and the attempt chain show it.

    A base URL may carry a key in its user info or query string, so only scheme, host, port and
    path are kept. A URL that cannot be cut down reliably is not shown at all.
    """
    if action.endpoint is None:
        return action.provider
    try:
        parts = urlsplit(action.endpoint.base_url)
    except ValueError:
        return action.provider
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return action.provider
    host = parts.netloc.rpartition("@")[2]
    return f"{parts.scheme}://{host}{parts.path}"


def _started(action: Dispatch) -> events.DispatchStarted:
    messages = (
        tuple(m.to_dict() for m in action.request.messages) if action.payload_retention else None
    )
    return events.DispatchStarted(
        action.request_id, _model(action), _shown(action), action.retries, messages
    )


def _call_failed(request_id: str, error: PrimaGateError) -> events.CallFailed:
    attempts = count_attempts(error.attempt_chain)
    return events.CallFailed(request_id, type(error).__name__, error.status_code, attempts)


def _announce(action: Action, outcome: Outcome, following: Action) -> tuple[events.Event, ...]:
    """Events for one step. Built from the step's own result so they cannot disagree with it."""
    rid = _request_id(action)
    if isinstance(action, Dispatch) and isinstance(outcome, DispatchOutcome):
        if outcome.error is not None:
            return _after_dispatch_failure(action, outcome.error, following)
        if isinstance(following, Return):
            done = (_call_succeeded(action, following),)
            metadata = following.response.metadata
            if metadata is not None and metadata.unknown_model:
                return (events.CostUnknown(rid, _model(action)), *done)
            return done
    if isinstance(action, Random) and isinstance(following, Sleep):
        resume = following.resume
        return (
            events.RetryScheduled(
                rid, _model(resume), _shown(resume), resume.retries, following.seconds
            ),
        )
    if isinstance(action, Sleep) and isinstance(following, Dispatch):
        return (_started(following),)
    return ()


def _call_succeeded(action: Dispatch, done: Return) -> events.CallSucceeded:
    metadata = done.response.metadata
    chain = metadata.attempt_chain if metadata else []
    dropped = tuple(metadata.dropped_params) if metadata else ()
    model = next((row.model for row in reversed(chain) if row.skipped is None), "")
    reply = (
        tuple(c.message.to_dict() for c in done.response.choices)
        if action.payload_retention
        else None
    )
    return events.CallSucceeded(action.request_id, model, count_attempts(chain), dropped, reply)


def _after_dispatch_failure(
    action: Dispatch, error: PrimaGateError, following: Action
) -> tuple[events.Event, ...]:
    rid = action.request_id
    before = len(attempt_chain(action, error))
    next_step: events.NextStep
    skipped: list[Attempt] = []
    if isinstance(following, Random):
        next_step = "retry"
    elif isinstance(following, Dispatch):
        moved_on = len(following.past) > len(action.past)
        next_step = "next_attempt" if moved_on else "next_endpoint"
        skipped = list(following.past[before:])
    else:
        next_step = "raise"
        skipped = error.attempt_chain[before:]
    announced: list[events.Event] = [
        events.DispatchFailed(
            rid, _model(action), _shown(action), type(error).__name__, error.status_code, next_step
        )
    ]
    announced += [events.AttemptSkipped(rid, row.model, row.skipped or "") for row in skipped]
    if isinstance(following, Dispatch):
        announced.append(_started(following))
    elif isinstance(following, Raise):
        announced.append(_call_failed(rid, following.error))
    return tuple(announced)
