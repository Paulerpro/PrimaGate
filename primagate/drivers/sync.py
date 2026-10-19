from __future__ import annotations

from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, ClassVar, Self

import httpx2

from primagate.core import steps
from primagate.core.call import first_action, next_action
from primagate.core.ports import Adapter
from primagate.drivers._shared import END, Finished, Step, StreamState, prepare, settle
from primagate.drivers.providers import SettingsResolver, SettingsSource, builtin_adapters
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from primagate.schema.wire import RawOutcome, StreamFailure
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
    ) -> None:
        self._adapters = adapters
        self._settings = settings
        self._transport = transport

    @classmethod
    def with_builtin_adapters(
        cls,
        *,
        api_keys: Mapping[str, str] | None = None,
        base_urls: Mapping[str, str] | None = None,
        timeout: float,
        http_transport: httpx2.BaseTransport | None = None,
    ) -> SyncDriver:
        return cls(
            adapters=builtin_adapters(),
            settings=SettingsResolver(api_keys=api_keys, base_urls=base_urls),
            transport=SyncTransport(timeout=timeout, transport=http_transport),
        )

    def close(self) -> None:
        self._transport.close()

    def run(self, request: ChatCompletionRequest) -> ChatCompletion:
        action = first_action(request)
        while True:
            step = self.perform(action)
            if isinstance(step, Finished):
                return step.response
            action = next_action(step)

    def stream(self, request: ChatCompletionRequest) -> SyncStream:
        action = first_action(request)
        if not isinstance(action, steps.Dispatch):
            self.perform(action)
            raise PrimaGateError("Stream request produced no dispatch")
        provider_request = prepare(self._adapters, self._settings, action)
        opened = self._transport.open_stream(provider_request)
        if isinstance(opened, RawOutcome):
            self.perform(next_action(settle(self._adapters, self._settings, action, opened)))
            raise PrimaGateError("Stream open failed without an error")
        key = self._settings[action.provider].api_key
        state = StreamState(self._adapters[action.provider], key, opened.headers)
        stream = SyncStream(opened, state)
        stream.prime()
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

    def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Step]]] = {
        steps.Dispatch: _dispatch,
        steps.Return: _return,
        steps.Raise: _raise,
    }
