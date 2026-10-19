from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
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
    ) -> None:
        self._adapters = adapters
        self._settings = settings
        self._transport = transport
        self._capabilities = {name: a.capabilities() for name, a in adapters.items()}

    @classmethod
    def with_builtin_adapters(
        cls,
        *,
        api_keys: Mapping[str, str] | None = None,
        base_urls: Mapping[str, str] | None = None,
        timeout: float,
        http_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> AsyncDriver:
        return cls(
            adapters=builtin_adapters(),
            settings=SettingsResolver(api_keys=api_keys, base_urls=base_urls),
            transport=AsyncTransport(timeout=timeout, transport=http_transport),
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def run(self, request: ChatCompletionRequest) -> ChatCompletion:
        action = first_action(request, self._capabilities)
        while True:
            step = await self.perform(action)
            if isinstance(step, Finished):
                return step.response
            action = next_action(step)

    async def stream(self, request: ChatCompletionRequest) -> AsyncStream:
        action = first_action(request, self._capabilities)
        if not isinstance(action, steps.Dispatch):
            await self.perform(action)
            raise PrimaGateError("Stream request produced no dispatch")
        provider_request = prepare(self._adapters, self._settings, action)
        opened = await self._transport.open_stream(provider_request)
        if isinstance(opened, RawOutcome):
            await self.perform(next_action(settle(self._adapters, self._settings, action, opened)))
            raise PrimaGateError("Stream open failed without an error")
        key = self._settings[action.provider].api_key
        state = StreamState(self._adapters[action.provider], key, opened.headers)
        stream = AsyncStream(opened, state)
        await stream.prime()
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

    async def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    async def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Awaitable[Step]]]] = {
        steps.Dispatch: _dispatch,
        steps.Return: _return,
        steps.Raise: _raise,
    }
