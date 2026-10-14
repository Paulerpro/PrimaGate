from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar

import httpx2

from primagate.adapters.openai import PROVIDER as OPENAI
from primagate.adapters.openai import OpenAIAdapter
from primagate.core import steps
from primagate.core.call import first_action, next_action
from primagate.core.ports import Adapter
from primagate.drivers._shared import Finished, Step, prepare, settle
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest
from primagate.schema.wire import ProviderSettings
from primagate.transport.async_ import AsyncTransport


class AsyncDriver:
    """Performs actions and feeds outcomes back. All decisions live in core."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, Adapter],
        settings: Mapping[str, ProviderSettings],
        transport: AsyncTransport,
    ) -> None:
        self._adapters = adapters
        self._settings = settings
        self._transport = transport

    @classmethod
    def with_builtin_adapters(
        cls,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        http_transport: httpx2.AsyncBaseTransport | None = None,
    ) -> AsyncDriver:
        return cls(
            adapters={OPENAI: OpenAIAdapter()},
            settings={OPENAI: ProviderSettings(api_key=api_key, base_url=base_url)},
            transport=AsyncTransport(timeout=timeout, transport=http_transport),
        )

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def run(self, request: ChatCompletionRequest) -> ChatCompletion:
        action = first_action(request)
        while True:
            step = await self.perform(action)
            if isinstance(step, Finished):
                return step.response
            action = next_action(step)

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
