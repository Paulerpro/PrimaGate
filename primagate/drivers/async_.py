from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any, ClassVar, Self

import httpx2

from primagate.core import steps
from primagate.core.call import advance, parse_drop_params, start
from primagate.core.ports import Adapter
from primagate.core.prices import BUNDLED, PriceTable
from primagate.drivers._shared import (
    END,
    EventHandler,
    Finished,
    Step,
    StreamState,
    credentials,
    deliver,
    prepare,
    settle,
)
from primagate.drivers.cache import ResponseCache
from primagate.drivers.providers import (
    ProviderRouting,
    SettingsResolver,
    SettingsSource,
    builtin_adapters,
)
from primagate.drivers.tokens import TokenCounting
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from primagate.schema.wire import (
    DEFAULT_RETRY,
    NO_FALLBACKS,
    Fallbacks,
    RawOutcome,
    RetryPolicy,
    StreamFailure,
)
from primagate.transport.async_ import AsyncEventStream, AsyncTransport

_NOTHING = object()


class AsyncStream:
    """Chunks of one streamed response. The driver opens it and reads up to the first chunk."""

    def __init__(self, source: AsyncEventStream, state: StreamState) -> None:
        self._source = source
        self._events = source.events()
        self._state = state
        self._ahead: object = _NOTHING
        self._closed = False

    async def _read(self) -> object:
        if self._closed:
            return END
        try:
            while True:
                item = self._state.next_item()
                if item is END:
                    await self.aclose()
                if item is not None:
                    return item
                try:
                    self._state.feed(await anext(self._events))
                except StopAsyncIteration:
                    self._state.end()
                except StreamFailure as failure:
                    self._state.fail(failure.kind)
        except BaseException:
            await self.aclose()
            raise

    async def prime(self) -> None:
        self._ahead = await self._read()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        item = END if self._closed else self._ahead
        if item is _NOTHING:
            item = await self._read()
        self._ahead = _NOTHING
        if item is END:
            raise StopAsyncIteration
        assert isinstance(item, ChatCompletionChunk)
        return item

    async def aclose(self) -> None:
        self._closed = True
        await self._events.aclose()
        await self._source.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class AsyncDriver:
    """Performs actions and feeds outcomes back. All decisions live in core."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, Adapter],
        settings: SettingsSource,
        transport: AsyncTransport,
        routing: ProviderRouting | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        fallbacks: Fallbacks = NO_FALLBACKS,
        event_handler: EventHandler | None = None,
        payload_retention: bool = False,
        price_overrides: PriceTable | None = None,
        counting: TokenCounting | None = None,
        cache: ResponseCache | None = None,
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
        self._capabilities = {name: a.capabilities() for name, a in adapters.items()}

    @classmethod
    def with_builtin_adapters(
        cls,
        *,
        api_keys: Mapping[str, str] | None = None,
        base_urls: Mapping[str, str] | None = None,
        timeout: float,
        http_transport: httpx2.AsyncBaseTransport | None = None,
        routing: ProviderRouting | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        fallbacks: Fallbacks = NO_FALLBACKS,
        event_handler: EventHandler | None = None,
        payload_retention: bool = False,
        price_overrides: PriceTable | None = None,
        counting: TokenCounting | None = None,
        cache: ResponseCache | None = None,
    ) -> AsyncDriver:
        return cls(
            adapters=builtin_adapters(),
            settings=SettingsResolver(api_keys=api_keys, base_urls=base_urls),
            transport=AsyncTransport(timeout=timeout, transport=http_transport),
            routing=routing,
            retry=retry,
            fallbacks=fallbacks,
            event_handler=event_handler,
            payload_retention=payload_retention,
            price_overrides=price_overrides,
            counting=counting,
            cache=cache,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

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
        )

    async def run(
        self, request: ChatCompletionRequest, drop_params: object = None, use_cache: bool = True
    ) -> ChatCompletion:
        action = self._first(request, drop_params, use_cache)
        while True:
            step = await self.perform(action)
            if isinstance(step, Finished):
                return step.response
            action = advance(action, step)

    async def stream(
        self, request: ChatCompletionRequest, drop_params: object = None
    ) -> AsyncStream:
        action = self._first(request, drop_params)
        while True:
            if isinstance(action, steps.Dispatch):
                opened = await self._open(action)
                if isinstance(opened, AsyncStream):
                    return opened
                step: steps.Outcome = opened
            else:
                performed = await self.perform(action)
                if isinstance(performed, Finished):
                    raise PrimaGateError("Stream request produced no dispatch")
                step = performed
            action = advance(action, step)

    async def _open(self, action: steps.Dispatch) -> AsyncStream | steps.DispatchOutcome:
        """Open a stream and read to its first chunk; failing before that can be retried."""
        provider_request = prepare(self._adapters, self._settings, action)
        opened = await self._transport.open_stream(provider_request)
        if isinstance(opened, RawOutcome):
            return settle(self._adapters, self._settings, action, opened)
        key = credentials(self._settings, action).api_key
        state = StreamState(self._adapters[action.provider], key, opened.headers, action)
        stream = AsyncStream(opened, state)
        try:
            await stream.prime()
        except PrimaGateError as err:
            return steps.DispatchOutcome(error=err)
        return stream

    async def perform(self, action: steps.Action) -> Step:
        handler = self.HANDLERS.get(type(action))
        if handler is None:
            raise PrimaGateError(f"unhandled action: {type(action).__name__}")
        return await handler(self, action)

    async def _dispatch(self, action: steps.Dispatch) -> Step:
        provider_request = prepare(self._adapters, self._settings, action)
        raw = await self._transport.send(provider_request)
        return settle(self._adapters, self._settings, action, raw)

    async def _random(self, action: steps.Random) -> Step:
        return steps.RandomOutcome(random.random())

    async def _sleep(self, action: steps.Sleep) -> Step:
        await asyncio.sleep(action.seconds)
        return steps.SleepOutcome()

    async def _new_id(self, action: steps.NewId) -> Step:
        # Prefixed so a call's ids never collide with background-preparation ids.
        return steps.IdOutcome(f"req-{uuid.uuid4().hex}")

    async def _emit(self, action: steps.Emit) -> Step:
        deliver(action.events, self._event_handler)
        return steps.EmitOutcome()

    async def _count(self, action: steps.CountTokens) -> Step:
        # Off the event loop: tiktoken reads files and may download a vocabulary on first use.
        return await asyncio.to_thread(
            self._counting.count, action.model, action.prompt, action.completion
        )

    async def _cache_read(self, action: steps.CacheRead) -> Step:
        return steps.CacheReadOutcome(self._cache.get(action.key))

    async def _cache_write(self, action: steps.CacheWrite) -> Step:
        self._cache.put(action.key, action.response)
        return steps.CacheWriteOutcome()

    async def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    async def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Awaitable[Step]]]] = {
        steps.Dispatch: _dispatch,
        steps.Return: _return,
        steps.Raise: _raise,
        steps.Random: _random,
        steps.Sleep: _sleep,
        steps.NewId: _new_id,
        steps.Emit: _emit,
        steps.CountTokens: _count,
        steps.CacheRead: _cache_read,
        steps.CacheWrite: _cache_write,
    }
