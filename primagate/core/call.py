from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from primagate.core.analyzer import (
    begin_analysis,
    decision_events,
    finish_analysis,
    resolve_tier,
)
from primagate.core.cache import cache_key
from primagate.core.classify import Fate, classify
from primagate.core.context import OverBudget, bounded_context
from primagate.core.counting import completion_text, prompt_text
from primagate.core.fallbacks import fallback_candidates
from primagate.core.models import resolve_model
from primagate.core.prices import BUNDLED, PriceTable, estimate
from primagate.core.profile import PLACEHOLDER_UNKNOWNS
from primagate.core.steps import (
    Action,
    Analyze,
    CacheRead,
    CacheReadOutcome,
    CacheWrite,
    CacheWriteOutcome,
    Cancelled,
    CountOutcome,
    CountTokens,
    Declined,
    Dispatch,
    DispatchOutcome,
    DropPolicy,
    Emit,
    EmitOutcome,
    IdOutcome,
    NewId,
    Now,
    NowOutcome,
    Outcome,
    Plan,
    Raise,
    Random,
    RandomOutcome,
    RecordSpend,
    Return,
    Route,
    Sleep,
    SleepOutcome,
    SpendOutcome,
    StartOutcome,
    StartPreparation,
    Stopped,
    StoreRead,
    StoreReadOutcome,
    StoreWrite,
    StoreWriteOutcome,
)
from primagate.core.tiers import narrowed
from primagate.schema import events
from primagate.schema.analyzer import Analysis, AnalyzerFailure
from primagate.schema.errors import (
    AttemptsExhaustedError,
    BadRequestError,
    PrimaGateError,
    UnsupportedParamsError,
)
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.profile import PrepState, Profile, ProfileContent
from primagate.schema.tiers import Tier
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


def _with_settings(
    request: ChatCompletionRequest, settings: Mapping[str, Any]
) -> ChatCompletionRequest:
    """`request` with a binding's settings filled in only where the Caller left a field unset.

    The Caller's own value always wins, so a per-request override needs no extra syntax.
    """
    filled: dict[str, Any] = {
        # A binding stores `stop` as a tuple so it cannot be changed; the request takes a list.
        name: list(value) if isinstance(value, tuple) else value
        for name, value in settings.items()
        if getattr(request, name) is None
    }
    return dataclasses.replace(request, **filled) if filled else request


@dataclasses.dataclass(frozen=True)
class Routing:
    """The Tiers that can take a request, each with the dispatch that would serve it."""

    candidates: tuple[tuple[Tier, Dispatch], ...]
    tiers: TierRouting
    # Every Tier the filter let through, kept even when de-duplicated by model, and the reason
    # for each it removed, so Tier Failover reuses the filter's verdict instead of redoing it.
    built: Mapping[Tier, Dispatch] = dataclasses.field(default_factory=dict)
    lacking: Mapping[Tier, str] = dataclasses.field(default_factory=dict)


def _route_tier(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    drops: DropPolicy,
    routes: Mapping[tuple[str, str], tuple[Endpoint, ...]],
    retry: RetryPolicy,
    tiers: TierRouting | None,
) -> Routing:
    """The Compatibility Filter for Tier Routing: every Tier that can serve the request.

    Each enabled Tier goes through `_build`, the check the plain path and the Fallback List use
    too, with the adapter's capabilities narrowed by that Tier's binding, so all three agree on
    what a model can do and on dropping before filtering. Each Tier's request carries that
    binding's settings, so whatever reuses a dispatch built here, Tier Failover included, sends
    the settings of the Tier it serves.
    """
    if tiers is None:
        raise BadRequestError(
            f"model '{TIER_ROUTING_ALIAS}' is the Tier Routing alias; "
            "declare tier_bindings and a policy before using it"
        )
    able: list[tuple[Tier, Dispatch]] = []
    lacking: dict[Tier, str] = {}
    shown: list[str] = []
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
        # Filled in before `_build`, so a binding's setting meets the same drop and
        # unsupported-parameter rules as one the Caller sent.
        asked = _with_settings(request, binding.settings)
        try:
            built = _build(asked, provider, model, seen_by_tier, drops, routes, retry)
            able.append((tier, built))
        except UnsupportedParamsError as reason:
            # Only a missing feature means "this Tier cannot take it"; anything else is a fault
            # to report as it is, not to hide behind a feature message.
            lacking[tier] = reason.message
            shown.append(f"{tier.name} ({provider}/{model}): {reason.message}")
    # After the filter, so a model removed by one Tier's limits can still serve under another's.
    first_per_model: dict[tuple[str, str], tuple[Tier, Dispatch]] = {}
    for tier, built in able:
        first_per_model.setdefault((built.provider, built.request.model), (tier, built))
    unique = tuple(first_per_model.values())
    if not unique:
        raise UnsupportedParamsError("No enabled Tier can take this request. " + "; ".join(shown))
    return Routing(unique, tiers, dict(able), lacking)


def _tier_failover(routing: Routing, tier: Tier, chosen: Dispatch) -> Dispatch | Attempt | None:
    """MEDIUM as the one Tier Failover for a call resolved to `tier`, or the row saying why it
    cannot run. None for MEDIUM itself, which has nothing to fail over to.

    The dispatch is the one the filter built for MEDIUM, with that Tier's own narrowed limits and
    drops, so failover never judges MEDIUM differently from routing.
    """
    if tier is Tier.MEDIUM:
        return None
    binding = routing.tiers.bindings.get(Tier.MEDIUM)
    shown = f"{binding.provider}/{binding.model_id}" if binding else Tier.MEDIUM.name
    if Tier.MEDIUM not in routing.tiers.policy.enabled_tiers:
        return Attempt(shown, [], skipped="MEDIUM is not in enabled_tiers")
    if Tier.MEDIUM in routing.lacking:
        return Attempt(shown, [], skipped=routing.lacking[Tier.MEDIUM])
    medium = routing.built[Tier.MEDIUM]
    # A model runs at most once per chain: its own Provider Routing has already been spent.
    if (medium.provider, medium.request.model) == (chosen.provider, chosen.request.model):
        return Attempt(shown, [], skipped="same model as the resolved Tier")
    return medium


def _routed_fallbacks(
    resolved: tuple[str, str], failover: Dispatch | Attempt | None, fallbacks: Fallbacks
) -> tuple[tuple[str, str], ...]:
    """The Fallback List a routed call tries after Tier Failover.

    The list is the one declared for the resolved Tier's model, as a plain call to that model
    would use. MEDIUM's model is dropped only when it really runs as the failover; a MEDIUM that
    was passed over was never tried, so the Caller's entry for it still gets its turn.
    """
    later = fallback_candidates(resolved, fallbacks)
    if isinstance(failover, Dispatch):
        ran = (failover.provider, failover.request.model)
        later = tuple(entry for entry in later if entry != ran)
    return later


def first_action(
    request: ChatCompletionRequest,
    capabilities: Mapping[str, Capabilities],
    policy: DropPolicy = NO_DROPS,
    routing: Mapping[tuple[str, str], tuple[Endpoint, ...]] | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    fallbacks: Fallbacks = NO_FALLBACKS,
    tier_routing: TierRouting | None = None,
) -> Action | Routing:
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


def _carried(built: Dispatch, action: Dispatch, past: Sequence[Attempt], plan: Plan) -> Dispatch:
    """`built` as the call's next Attempt, with what the call carries and what is left after.

    `tier` is not carried: only Tier Failover knows the Tier (MEDIUM) its Attempt serves, so a
    fallback's answer reports none. The spending ceiling is carried, so whatever answers a routed
    call is still counted against it.
    """
    return dataclasses.replace(
        built,
        request_id=action.request_id,
        payload_retention=action.payload_retention,
        prices=action.prices,
        cache_key=action.cache_key,
        no_cache=action.no_cache,
        # Already paid once when the Tier was chosen, so every later Attempt reports the same.
        analyzer=action.analyzer,
        spending_ceiling=action.spending_ceiling,
        past=tuple(past),
        plan=plan if plan.failover is not None or plan.later else None,
    )


def _next_attempt(
    action: Dispatch, plan: Plan, error: PrimaGateError
) -> tuple[Dispatch | None, list[Attempt]]:
    """Start the next candidate that can take the request, and say which were passed.

    Tier Failover comes first, then the fallbacks in order. Each candidate that cannot take the
    request leaves a skipped row, using the same message the primary path would raise, so the
    two paths cannot drift apart.
    """
    chain = attempt_chain(action, error)
    skipped: list[Attempt] = []
    if plan.failover is not None:
        rest = dataclasses.replace(plan, failover=None)
        if isinstance(plan.failover, Dispatch):
            medium = _carried(plan.failover, action, chain, rest)
            return dataclasses.replace(medium, tier=Tier.MEDIUM), skipped
        skipped.append(plan.failover)
        plan = rest
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
        rest = dataclasses.replace(plan, later=plan.later[index + 1 :])
        return _carried(built, action, (*chain, *skipped), rest), skipped
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
            tier=action.tier.name if action.tier else None,
            analyzer=action.analyzer,
        )
        return Return(dataclasses.replace(outcome.response, metadata=metadata))
    if (
        isinstance(action, Random)
        and isinstance(outcome, RandomOutcome)
        and isinstance(action.resume, Dispatch)
    ):
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
    routes = routing or {}

    def ready(first: Dispatch, request_id: str, key: str | None, read_already: bool) -> Action:
        """Make a chosen dispatch the call's own: id, price table, then the Cache, then send."""
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
        return CacheRead(key, lambda read: _after_read(first, read), request_id)

    def failed(request_id: str, error: PrimaGateError) -> Action:
        return Emit((_call_failed(request_id, error),), Raise(error))

    def decided(
        routing: Routing,
        tier: Tier,
        request_id: str,
        analysis: Analysis | None,
        key: str | None,
        read_already: bool,
        *,
        asked: bool | None = None,
        degraded: str | None = None,
        profile: Profile | None = None,
    ) -> Action:
        eligible = [t for t, _ in routing.candidates]
        chosen = dict(routing.candidates)[tier]
        binding = routing.tiers.bindings[tier]
        usage = analysis.usage if analysis else None
        failover = _tier_failover(routing, tier, chosen)
        later = _routed_fallbacks((binding.provider, binding.model_id), failover, fallbacks)
        plan = (
            None
            if failover is None and not later
            else Plan(
                request=request,
                capabilities=capabilities,
                drops=policy,
                routing=routes,
                retry=retry,
                later=later,
                failover=failover,
            )
        )
        # An answer from the Interim Binding is a stand-in, so it must never be replayed.
        chosen = dataclasses.replace(
            chosen,
            tier=tier,
            analyzer=usage,
            no_cache=degraded is not None,
            spending_ceiling=routing.tiers.policy.spending_ceiling,
            plan=plan,
        )
        announced = decision_events(
            request_id,
            tier,
            f"{tier.name}:{binding.provider}/{binding.model_id}",
            eligible,
            analysis,
            asked=asked,
            degraded=degraded,
            profile=profile,
        )
        return Emit(announced, ready(chosen, request_id, key, read_already))

    def degrade(
        routing: Routing,
        reason: str,
        request_id: str,
        key: str | None,
        read_already: bool,
        analysis: Analysis | None = None,
        *,
        asked: bool = False,
        profile: Profile | None = None,
    ) -> Action:
        """Give up on choosing: one dispatch on the Interim Binding, with the reason on record.

        Every cause that stops the Analyzer deciding comes here, so each degrades the same way.
        """
        eligible = tuple(t for t, _ in routing.candidates)
        tier = resolve_tier(routing.tiers.policy.interim_binding, eligible)
        return decided(
            routing,
            tier,
            request_id,
            analysis,
            key,
            read_already,
            asked=asked,
            degraded=reason,
            profile=profile,
        )

    def analyzing(routing: Routing, request_id: str, key: str | None, read_already: bool) -> Action:
        """Two or more Tiers can serve the request, so the Analyzer chooses between them.

        With sources declared, the call first asks for its Profile to be prepared, and keeps the
        state it is told for its whole run, so a preparation that ends mid-call changes nothing
        for it. It never waits on preparation: until a Profile is READY, or once preparation has
        failed, the Interim Binding serves it without asking the Analyzer, since the Analyzer
        without a Profile is what declaring sources asked to replace.
        """
        fingerprint = routing.tiers.fingerprint
        if fingerprint is None:
            return choosing(routing, request_id, key, read_already, None)

        def pinned(outcome: StartOutcome) -> Action:
            if outcome.state is PrepState.READY and outcome.profile is not None:
                return choosing(routing, request_id, key, read_already, outcome.profile)
            reason = "preparation_failed" if outcome.state is PrepState.FAILED else "preparing"
            return degrade(routing, reason, request_id, key, read_already)

        return StartPreparation(fingerprint, pinned, request_id)

    def choosing(
        routing: Routing,
        request_id: str,
        key: str | None,
        read_already: bool,
        profile: Profile | None,
    ) -> Action:
        """Ask the Analyzer, with the READY Profile when there is one, else the empty one."""
        tiers = routing.tiers
        eligible = tuple(t for t, _ in routing.candidates)
        # Only Tiers that can take the request are offered, so the answer is always usable.
        narrowed_policy = dataclasses.replace(tiers.policy, enabled_tiers=eligible)

        def give_up(reason: str, analysis: Analysis | None = None, asked: bool = False) -> Action:
            return degrade(
                routing,
                reason,
                request_id,
                key,
                read_already,
                analysis,
                asked=asked,
                profile=profile,
            )

        if tiers.analyzer is None:
            return give_up("no_analyzer")
        view = bounded_context(request, tiers.policy.context_token_budget)
        if isinstance(view, OverBudget):
            return give_up("over_budget")
        content = profile.content if profile else ProfileContent(unknowns=PLACEHOLDER_UNKNOWNS)
        try:
            begun = begin_analysis(
                tiers.analyzer,
                content,
                view,
                narrowed_policy,
                capabilities,
                BUNDLED if prices is None else prices,
            )
        except PrimaGateError:
            return give_up("analyzer_unavailable")
        if isinstance(begun, Analysis):
            return give_up(begun.outcome.reason)

        def after(outcome: DispatchOutcome | Declined) -> Action:
            if isinstance(outcome, Declined):
                return give_up(outcome.reason)
            analysis = finish_analysis(begun, outcome, narrowed_policy)
            if isinstance(analysis.outcome, AnalyzerFailure):
                return give_up(analysis.outcome.reason, analysis, asked=True)
            chosen = resolve_tier(analysis.outcome.tier, eligible)
            return decided(
                routing, chosen, request_id, analysis, key, read_already, profile=profile
            )

        return Analyze(dataclasses.replace(begun, request_id=request_id), after)

    def begin(request_id: str, key: str | None = None, read_already: bool = False) -> Action:
        first = first_action(request, capabilities, policy, routing, retry, fallbacks, tier_routing)
        if isinstance(first, Routing):
            if len(first.candidates) == 1:
                tier = first.candidates[0][0]
                return decided(first, tier, request_id, None, key, read_already)
            return analyzing(first, request_id, key, read_already)
        if isinstance(first, Dispatch):
            return ready(first, request_id, key, read_already)
        if isinstance(first, Raise):
            return failed(request_id, first.error)
        return first

    def begin_routed(request_id: str) -> Action:
        # A hit returns before the Tier is chosen, so it never depends on the filter or on the
        # Analyzer. Anything a past answer was cached for would pass the filter again. Binding
        # settings stay out of the key: they are fixed for the instance, like the bindings.
        key = cache_key(request, policy, cache_salt)
        if key is None:
            return begin(request_id)

        def after(read: CacheReadOutcome) -> Action:
            if read.response is None:
                return begin(request_id, key, read_already=True)
            hit = dataclasses.replace(read.response, metadata=Metadata(cache_hit=True))
            return Emit((events.CacheHit(request_id, request.model, key),), Return(hit))

        return CacheRead(key, after, request_id)

    routed = request.model == TIER_ROUTING_ALIAS and tier_routing is not None and cacheable
    return NewId(begin_routed if routed else begin)


def advance(action: Action, outcome: Outcome) -> Action:
    """Like `next_action`, and also says what happened so the driver can deliver the events."""
    # Answered before anything else, so no retry, fallback, cache write, spend record or
    # degrade can follow a cancellation, whatever the action was.
    if isinstance(outcome, Cancelled):
        return _cancelled(action)
    if isinstance(action, NewId) and isinstance(outcome, IdOutcome):
        return action.resume(outcome.value)
    if isinstance(action, Now) and isinstance(outcome, NowOutcome):
        return action.resume(outcome.value)
    # A wait that is not a call's retry carries its own next step, and no call events.
    if (
        isinstance(action, Random)
        and isinstance(outcome, RandomOutcome)
        and not isinstance(action.resume, Dispatch)
    ):
        return action.resume(outcome.value)
    if (
        isinstance(action, Sleep)
        and isinstance(outcome, SleepOutcome)
        and not isinstance(action.resume, Dispatch)
    ):
        return action.resume
    if isinstance(action, Emit) and isinstance(outcome, EmitOutcome):
        return action.resume
    if isinstance(action, CountTokens) and isinstance(outcome, CountOutcome):
        return action.resume(outcome)
    if isinstance(action, CacheRead) and isinstance(outcome, CacheReadOutcome):
        return action.resume(outcome)
    if isinstance(action, CacheWrite) and isinstance(outcome, CacheWriteOutcome):
        return action.resume
    if isinstance(action, StoreRead) and isinstance(outcome, StoreReadOutcome):
        return action.resume(outcome)
    if isinstance(action, StoreWrite) and isinstance(outcome, StoreWriteOutcome):
        return action.resume
    if isinstance(action, Analyze) and isinstance(outcome, DispatchOutcome | Declined):
        return action.resume(outcome)
    if isinstance(action, RecordSpend) and isinstance(outcome, SpendOutcome):
        return action.resume(outcome)
    if isinstance(action, StartPreparation) and isinstance(outcome, StartOutcome):
        return action.resume(outcome)
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
            action.request_id,
            action,
        )
    return _announced(action, outcome, following)


def _cancelled(action: Action) -> Action:
    """Stop the call: count what it was already billed for, then announce the cancellation."""
    request_id = _request_id(action) or "unknown"
    stopped = Emit((events.CallCancelled(request_id),), Stopped())
    owed = _owed(action)
    if owed is None:
        return stopped
    amount, ceiling = owed

    def after(counted: SpendOutcome) -> Action:
        if not counted.crossed:
            return stopped
        warning = events.SpendingCeilingExceeded(request_id, ceiling, counted.total)
        return Emit((warning,), stopped)

    return RecordSpend(amount, ceiling, after, request_id)


def _owed(action: Action) -> tuple[float, float] | None:
    """The spend a cancelled call was billed for and has not yet counted, with its ceiling.

    Follows the steps already decided until it meets the pending spend record, or the dispatch
    that carries the Analyzer's cost. A worker answer still being counted has no cost yet, so
    only the Analyzer's is owed then. Only a routed call carries a spending ceiling, so the
    ceiling alone decides, whichever model the call was on when it stopped.
    """
    while True:
        if isinstance(action, RecordSpend):
            return action.amount, action.ceiling
        if isinstance(action, Emit | CacheWrite):
            action = action.resume
        elif isinstance(action, Random | Sleep) and isinstance(action.resume, Dispatch):
            action = action.resume
        elif isinstance(action, CountTokens) and action.dispatch is not None:
            action = action.dispatch
        elif isinstance(action, Dispatch):
            break
        else:
            return None
    usage, ceiling = action.analyzer, action.spending_ceiling
    if ceiling is None or usage is None or usage.cost is None:
        return None
    paid = usage.cost.total
    # The same guard as a finished call: a bad price would poison the total for good.
    if not (math.isfinite(paid) and paid >= 0):
        return None
    return paid, ceiling


def _announced(action: Action, outcome: Outcome, following: Action) -> Action:
    announced = _announce(action, outcome, following)
    after = _written(action, following)
    return Emit(announced, after) if announced else after


def _written(action: Action, following: Action) -> Action:
    """Store a final answer before it is returned, so the next identical call can reuse it."""
    final = _spent(action, following)
    if (
        isinstance(action, Dispatch)
        and action.cache_key
        and not action.no_cache
        and isinstance(following, Return)
    ):
        return CacheWrite(action.cache_key, following.response, final, action.request_id)
    return final


def _spent(action: Action, following: Action) -> Action:
    """Count a finished routed call against the ceiling, if the Policy set one.

    Only a routed call carries a ceiling, so the ceiling alone decides: a fallback's answer has no
    Tier but is still the routed call's spend. The answer is returned whatever the count says: the
    ceiling only ever warns.
    """
    if not (
        isinstance(action, Dispatch)
        and action.spending_ceiling is not None
        and isinstance(following, Return)
    ):
        return following
    metadata = following.response.metadata
    if metadata is None:
        return following
    costs = [metadata.cost, metadata.analyzer.cost if metadata.analyzer else None]
    # A price that is not a plain non-negative number would poison the total for good.
    paid = [c.total for c in costs if c is not None and math.isfinite(c.total) and c.total >= 0]
    if not paid:
        return following
    ceiling = action.spending_ceiling

    def after(counted: SpendOutcome) -> Action:
        if not counted.crossed:
            return following
        warning = events.SpendingCeilingExceeded(action.request_id, ceiling, counted.total)
        return Emit((warning,), following)

    return RecordSpend(sum(paid), ceiling, after, action.request_id)


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
    """The id of the call `action` belongs to; empty before the call has one."""
    if isinstance(action, Dispatch):
        return action.request_id
    if isinstance(action, Random | Sleep) and isinstance(action.resume, Dispatch):
        return action.resume.request_id
    if isinstance(action, Analyze):
        return action.dispatch.request_id
    if isinstance(action, CacheRead | CacheWrite | CountTokens | RecordSpend | StartPreparation):
        return action.request_id
    if isinstance(action, Emit) and action.events:
        return action.events[0].request_id
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
    if (
        isinstance(action, Random)
        and isinstance(following, Sleep)
        and isinstance(following.resume, Dispatch)
    ):
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
