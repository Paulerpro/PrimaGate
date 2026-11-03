from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable, Mapping
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
from primagate.drivers.providers import (
    ProviderRouting,
    SettingsResolver,
    SettingsSource,
    builtin_adapters,
)
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
        )

    def close(self) -> None:
        self._transport.close()

    def _first(self, request: ChatCompletionRequest, drop_params: object) -> steps.Action:
        return start(
            request,
            self._capabilities,
            parse_drop_params(drop_params),
            self._routing,
            self._retry,
            self._fallbacks,
            self._payload_retention,
            prices=self._prices,
        )

    def run(self, request: ChatCompletionRequest, drop_params: object = None) -> ChatCompletion:
        action = self._first(request, drop_params)
        while True:
            step = self.perform(action)
            if isinstance(step, Finished):
                return step.response
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
                if isinstance(performed, Finished):
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

    def perform(self, action: steps.Action) -> Step:
        handler = self.HANDLERS.get(type(action))
        if handler is None:
            raise PrimaGateError(f"unhandled action: {type(action).__name__}")
        return handler(self, action)

    def _dispatch(self, action: steps.Dispatch) -> Step:
        provider_request = prepare(self._adapters, self._settings, action)
        raw = self._transport.send(provider_request)
        return settle(self._adapters, self._settings, action, raw)

    def _random(self, action: steps.Random) -> Step:
        return steps.RandomOutcome(random.random())

    def _sleep(self, action: steps.Sleep) -> Step:
        time.sleep(action.seconds)
        return steps.SleepOutcome()

    def _new_id(self, action: steps.NewId) -> Step:
        # Prefixed so a call's ids never collide with background-preparation ids.
        return steps.IdOutcome(f"req-{uuid.uuid4().hex}")

    def _emit(self, action: steps.Emit) -> Step:
        deliver(action.events, self._event_handler)
        return steps.EmitOutcome()

    def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Step]]] = {
        steps.Dispatch: _dispatch,
        steps.Return: _return,
        steps.Raise: _raise,
        steps.Random: _random,
        steps.Sleep: _sleep,
        steps.NewId: _new_id,
        steps.Emit: _emit,
    }
