from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, ClassVar, Self

import httpx2

from primagate.core import steps
from primagate.core.analyzer import begin_analysis, finish_analysis
from primagate.core.call import advance, parse_drop_params, start
from primagate.core.context import RoutingContext
from primagate.core.ports import Adapter
from primagate.core.preparation import begin_preparation
from primagate.core.prices import BUNDLED, PriceTable
from primagate.core.profile import read_profile
from primagate.drivers._shared import (
    END,
    EventHandler,
    Finished,
    Prepared,
    Step,
    StreamState,
    credentials,
    deliver,
    logger,
    prepare,
    settle,
)
from primagate.drivers.cache import ResponseCache
from primagate.drivers.capacity import AnalyzerCap
from primagate.drivers.preparation import (
    PreparationRunner,
    PreparationStopped,
    Progress,
    StoreFailed,
)
from primagate.drivers.profile_store import ProfileStore, default_profile_store
from primagate.drivers.providers import (
    ProviderRouting,
    SettingsResolver,
    SettingsSource,
    builtin_adapters,
)
from primagate.drivers.spend import SpendTracker
from primagate.drivers.tokens import TokenCounting
from primagate.schema.analyzer import Analysis, AnalyzerFailure
from primagate.schema.errors import BadRequestError, PrimaGateError
from primagate.schema.events import PrepFailureReason
from primagate.schema.policy import Policy, TierRouting
from primagate.schema.profile import PrepState, Profile, ProfileContent, ProfileSources
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from primagate.schema.wire import (
    DEFAULT_RETRY,
    NO_FALLBACKS,
    Fallbacks,
    RawOutcome,
    RetryPolicy,
    StreamFailure,
)
from primagate.transport.sync import SyncEventStream, SyncTransport

_NOTHING = object()


class SyncStream:
    """Chunks of one streamed response. The driver opens it and reads up to the first chunk."""

    def __init__(self, source: SyncEventStream, state: StreamState) -> None:
        self._source = source
        self._events = source.events()
        self._state = state
        self._ahead: object = _NOTHING
        self._closed = False

    def _read(self) -> object:
        if self._closed:
            return END
        try:
            while True:
                item = self._state.next_item()
                if item is END:
                    self.close()
                if item is not None:
                    return item
                try:
                    self._state.feed(next(self._events))
                except StopIteration:
                    self._state.end()
                except StreamFailure as failure:
                    self._state.fail(failure.kind)
        except BaseException:
            self.close()
            raise

    def prime(self) -> None:
        self._ahead = self._read()

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> ChatCompletionChunk:
        item = END if self._closed else self._ahead
        if item is _NOTHING:
            item = self._read()
        self._ahead = _NOTHING
        if item is END:
            raise StopIteration
        assert isinstance(item, ChatCompletionChunk)
        return item

    def close(self) -> None:
        self._closed = True
        self._source.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class SyncDriver:
    """Performs actions and feeds outcomes back. All decisions live in core."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, Adapter],
        settings: SettingsSource,
        transport: SyncTransport,
        routing: ProviderRouting | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        fallbacks: Fallbacks = NO_FALLBACKS,
        event_handler: EventHandler | None = None,
        payload_retention: bool = False,
        price_overrides: PriceTable | None = None,
        counting: TokenCounting | None = None,
        cache: ResponseCache | None = None,
        tier_routing: TierRouting | None = None,
        analyzer_cap: AnalyzerCap | None = None,
        spend: SpendTracker | None = None,
        profile_store: ProfileStore | None = None,
        runner: PreparationRunner | None = None,
    ) -> None:
        self._adapters = adapters
        self._settings = settings
        self._transport = transport
        self._routing = routing
        self._retry = retry
        self._fallbacks = fallbacks
        self._event_handler = event_handler
        self._payload_retention = payload_retention
        self._prices = {**BUNDLED, **(price_overrides or {})}
        self._counting = counting or TokenCounting()
        self._cache = cache or ResponseCache()
        self._tier_routing = tier_routing
        self._analyzer_cap = analyzer_cap
        self._spend = spend or SpendTracker()
        self._profile_store = default_profile_store() if profile_store is None else profile_store
        # None means no sources were declared, so nothing is ever prepared in the background.
        self._runner = runner
        self._capabilities = {name: a.capabilities() for name, a in adapters.items()}

    @classmethod
    def with_builtin_adapters(
        cls,
        *,
        api_keys: Mapping[str, str] | None = None,
        base_urls: Mapping[str, str] | None = None,
        timeout: float,
        http_transport: httpx2.BaseTransport | None = None,
        routing: ProviderRouting | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        fallbacks: Fallbacks = NO_FALLBACKS,
        event_handler: EventHandler | None = None,
        payload_retention: bool = False,
        price_overrides: PriceTable | None = None,
        counting: TokenCounting | None = None,
        cache: ResponseCache | None = None,
        tier_routing: TierRouting | None = None,
        analyzer_cap: AnalyzerCap | None = None,
        spend: SpendTracker | None = None,
        profile_store: ProfileStore | None = None,
        runner: PreparationRunner | None = None,
    ) -> SyncDriver:
        return cls(
            adapters=builtin_adapters(),
            settings=SettingsResolver(api_keys=api_keys, base_urls=base_urls),
            transport=SyncTransport(timeout=timeout, transport=http_transport),
            routing=routing,
            retry=retry,
            fallbacks=fallbacks,
            event_handler=event_handler,
            payload_retention=payload_retention,
            price_overrides=price_overrides,
            counting=counting,
            cache=cache,
            tier_routing=tier_routing,
            analyzer_cap=analyzer_cap,
            spend=spend,
            profile_store=profile_store,
            runner=runner,
        )

    def close(self) -> None:
        self._transport.close()

    def _first(
        self, request: ChatCompletionRequest, drop_params: object, use_cache: bool = True
    ) -> steps.Action:
        return start(
            request,
            self._capabilities,
            parse_drop_params(drop_params),
            self._routing,
            self._retry,
            self._fallbacks,
            self._payload_retention,
            prices=self._prices,
            use_cache=use_cache,
            cache_salt=self._cache.salt,
            tier_routing=self._tier_routing,
        )

    def run(
        self, request: ChatCompletionRequest, drop_params: object = None, use_cache: bool = True
    ) -> ChatCompletion:
        action = self._first(request, drop_params, use_cache)
        while True:
            step = self.perform(action)
            if isinstance(step, Finished):
                return step.response
            if isinstance(step, Prepared):
                raise PrimaGateError("A call ended as a preparation")
            action = advance(action, step)

    def stream(self, request: ChatCompletionRequest, drop_params: object = None) -> SyncStream:
        action = self._first(request, drop_params)
        while True:
            if isinstance(action, steps.Dispatch):
                opened = self._open(action)
                if isinstance(opened, SyncStream):
                    return opened
                step: steps.Outcome = opened
            else:
                performed = self.perform(action)
                if isinstance(performed, Finished | Prepared):
                    raise PrimaGateError("Stream request produced no dispatch")
                step = performed
            action = advance(action, step)

    def _open(self, action: steps.Dispatch) -> SyncStream | steps.DispatchOutcome:
        """Open a stream and read to its first chunk; failing before that can be retried."""
        provider_request = prepare(self._adapters, self._settings, action)
        opened = self._transport.open_stream(provider_request)
        if isinstance(opened, RawOutcome):
            return settle(self._adapters, self._settings, action, opened)
        key = credentials(self._settings, action).api_key
        state = StreamState(self._adapters[action.provider], key, opened.headers, action)
        stream = SyncStream(opened, state)
        try:
            stream.prime()
        except PrimaGateError as err:
            return steps.DispatchOutcome(error=err)
        return stream

    def prepare(self, sources: ProfileSources, fingerprint: str) -> Prepared:
        """Prepare a Profile from `sources` now, in the foreground, and store it when READY.

        It uses this instance's Analyzer, prices and Policy ceiling. Provider and answer failures
        end as FAILED with events; an error raised by the Profile store reaches the caller
        unchanged. Once the instance is closed it stops before its next action and raises
        `PreparationStopped`.
        """
        return self._drive(self._begin(sources, fingerprint))

    def _begin(self, sources: ProfileSources, fingerprint: str) -> steps.Action:
        tiers = self._tier_routing
        if tiers is None or tiers.analyzer is None:
            raise BadRequestError("Preparing a Profile needs an analyzer")
        return begin_preparation(
            sources,
            fingerprint,
            tiers.analyzer,
            self._capabilities,
            self._prices,
            tiers.policy.spending_ceiling,
        )

    def _drive(self, action: steps.Action, progress: Progress | None = None) -> Prepared:
        """Run a preparation to its end. With `progress`, it is the background job's: what it
        shows is recorded, and a store error comes out as `StoreFailed`."""
        while True:
            # Checked before every action, so a closed instance never sends or writes again.
            if self._runner is not None and self._runner.is_closed:
                raise PreparationStopped
            if progress is not None:
                progress.observe(action)
            try:
                step = self.perform(action)
            except Exception as err:
                if progress is not None and isinstance(action, steps.StoreRead | steps.StoreWrite):
                    raise StoreFailed from err
                raise
            if isinstance(step, Prepared):
                return step
            if isinstance(step, Finished):
                raise PrimaGateError("A preparation ended as a call")
            action = advance(action, step)

    def _background(self, sources: ProfileSources, fingerprint: str) -> steps.Action:
        """A stored Profile that still fits is used as it is; only a miss is prepared."""

        def found(profile: Profile | None) -> steps.Action:
            if profile is not None:
                return steps.PreparationDone(PrepState.READY, profile)
            return self._begin(sources, fingerprint)

        return read_profile(fingerprint, found)

    def _job(self, sources: ProfileSources, fingerprint: str) -> None:
        """One background preparation, on its own thread. No `Exception` escapes it; anything
        rarer, such as `SystemExit`, is recorded as a failure and then let through."""
        runner = self._runner
        assert runner is not None
        progress = Progress()
        reason: PrepFailureReason
        try:
            prepared = self._drive(self._background(sources, fingerprint), progress)
        except PreparationStopped:
            return
        except StoreFailed:
            reason = "store_error"
        except Exception:
            # DEBUG only: the message may echo provider text, and the event already says it failed.
            logger.debug("preparation failed unexpectedly", exc_info=True)
            reason = "provider_error"
        except BaseException:
            # Recorded first, or the fingerprint would stay PREPARING for the life of the process.
            runner.finish(fingerprint, PrepState.FAILED)
            raise
        else:
            runner.finish(fingerprint, prepared.state, progress.reason)
            return
        # A closed instance stops quietly, even when closing it is what broke the job.
        if runner.is_closed:
            return
        try:
            deliver((progress.failed(fingerprint, reason),), self._event_handler)
        finally:
            runner.finish(fingerprint, PrepState.FAILED, reason)

    def _start_preparation(self, action: steps.StartPreparation) -> Step:
        runner, tiers = self._runner, self._tier_routing
        # Only the instance's own sources are ever prepared, under their own fingerprint.
        if (
            runner is None
            or tiers is None
            or tiers.sources is None
            or tiers.fingerprint != action.fingerprint
        ):
            return steps.StartOutcome(False)
        sources, fingerprint = tiers.sources, tiers.fingerprint
        started = runner.start_thread(fingerprint, lambda: self._job(sources, fingerprint))
        return steps.StartOutcome(started)

    def analyze(
        self, model: str, profile: ProfileContent, context: RoutingContext, policy: Policy
    ) -> Analysis:
        """Ask `model` which Tier the conversation needs, with exactly one low-level dispatch.

        It never goes through `run`, so routing, retries and fallbacks cannot reach it, and a
        failed answer is returned as a failure for the caller to decide on.
        """
        begun = begin_analysis(model, profile, context, policy, self._capabilities, self._prices)
        if isinstance(begun, Analysis):
            return begun
        try:
            outcome = self._dispatch(begun)
        except PrimaGateError:
            # A missing key is still the Analyzer failing, and routing must survive that.
            return Analysis(AnalyzerFailure("provider_error"), None)
        if not isinstance(outcome, steps.DispatchOutcome):
            raise PrimaGateError("Analyzer dispatch produced no outcome")
        return finish_analysis(begun, outcome, policy)

    def perform(self, action: steps.Action) -> Step:
        handler = self.HANDLERS.get(type(action))
        if handler is None:
            raise PrimaGateError(f"unhandled action: {type(action).__name__}")
        return handler(self, action)

    def _dispatch(self, action: steps.Dispatch) -> Step:
        provider_request = prepare(self._adapters, self._settings, action)
        raw = self._transport.send(provider_request)
        return settle(self._adapters, self._settings, action, raw)

    def _analyze(self, action: steps.Analyze) -> Step:
        # Preparation never takes a permit: the cap protects request latency, which preparation
        # does not affect.
        cap = self._analyzer_cap if action.capped else None
        # Over the limit means no request and no waiting: core sends the call to the Interim
        # Binding instead, so a slow Analyzer cannot hold callers up.
        if cap is not None and not cap.try_acquire():
            return steps.Declined("over_capacity")
        try:
            # The Analyzer failing is an outcome for core to judge, never an error out of here.
            return self._dispatch(action.dispatch)
        except PrimaGateError as err:
            return steps.DispatchOutcome(error=err)
        finally:
            if cap is not None:
                cap.release()

    def _record_spend(self, action: steps.RecordSpend) -> Step:
        total, crossed = self._spend.add(action.amount, action.ceiling)
        return steps.SpendOutcome(total, crossed)

    def _random(self, action: steps.Random) -> Step:
        return steps.RandomOutcome(random.random())

    def _sleep(self, action: steps.Sleep) -> Step:
        time.sleep(action.seconds)
        return steps.SleepOutcome()

    def _new_id(self, action: steps.NewId) -> Step:
        # Prefixed so a call's ids never collide with preparation ids.
        return steps.IdOutcome(f"{action.prefix}-{uuid.uuid4().hex}")

    def _now(self, action: steps.Now) -> Step:
        return steps.NowOutcome(datetime.now(UTC).isoformat())

    def _emit(self, action: steps.Emit) -> Step:
        deliver(action.events, self._event_handler)
        return steps.EmitOutcome()

    def _count(self, action: steps.CountTokens) -> Step:
        return self._counting.count(action.model, action.prompt, action.completion)

    def _cache_read(self, action: steps.CacheRead) -> Step:
        return steps.CacheReadOutcome(self._cache.get(action.key))

    def _cache_write(self, action: steps.CacheWrite) -> Step:
        self._cache.put(action.key, action.response)
        return steps.CacheWriteOutcome()

    def _store_read(self, action: steps.StoreRead) -> Step:
        return steps.StoreReadOutcome(self._profile_store.read())

    def _store_write(self, action: steps.StoreWrite) -> Step:
        self._profile_store.write(action.profile)
        return steps.StoreWriteOutcome()

    def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    def _preparation_done(self, action: steps.PreparationDone) -> Step:
        return Prepared(action.state, action.profile)

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Step]]] = {
        steps.Dispatch: _dispatch,
        steps.Analyze: _analyze,
        steps.RecordSpend: _record_spend,
        steps.StartPreparation: _start_preparation,
        steps.Return: _return,
        steps.Raise: _raise,
        steps.Random: _random,
        steps.Sleep: _sleep,
        steps.NewId: _new_id,
        steps.Now: _now,
        steps.Emit: _emit,
        steps.CountTokens: _count,
        steps.CacheRead: _cache_read,
        steps.CacheWrite: _cache_write,
        steps.StoreRead: _store_read,
        steps.StoreWrite: _store_write,
        steps.PreparationDone: _preparation_done,
    }
