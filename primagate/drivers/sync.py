from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import httpx2

from primagate.core import steps
from primagate.core.call import first_action, next_action
from primagate.core.ports import Adapter
from primagate.drivers._shared import Finished, Step, prepare, settle
from primagate.drivers.providers import SettingsResolver, SettingsSource, builtin_adapters
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest
from primagate.transport.sync import SyncTransport


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
