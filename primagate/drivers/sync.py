from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx2

from primagate.adapters.openai import PROVIDER as OPENAI
from primagate.adapters.openai import OpenAIAdapter
from primagate.core import steps
from primagate.core.call import first_action, next_action
from primagate.core.ports import Adapter
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest
from primagate.schema.wire import ProviderSettings
from primagate.transport.sync import SyncTransport


@dataclass(frozen=True)
class Finished:
    response: ChatCompletion


Step = steps.DispatchOutcome | Finished


class SyncDriver:
    """Performs actions and feeds outcomes back. All decisions live in core."""

    def __init__(
        self,
        *,
        adapters: Mapping[str, Adapter],
        settings: Mapping[str, ProviderSettings],
        transport: SyncTransport,
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
        http_transport: httpx2.BaseTransport | None = None,
    ) -> SyncDriver:
        # The registry is per instance so one caller's adapters never leak to another.
        return cls(
            adapters={OPENAI: OpenAIAdapter()},
            settings={OPENAI: ProviderSettings(api_key=api_key, base_url=base_url)},
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
        adapter = self._adapters[action.provider]
        settings = self._settings[action.provider]
        provider_request = adapter.build_request(
            action.request, api_key=settings.api_key, base_url=settings.base_url
        )
        raw = self._transport.send(provider_request)
        try:
            if raw.ok:
                return steps.DispatchOutcome(response=adapter.parse_response(raw))
            error = adapter.map_error(raw)
        except PrimaGateError as err:
            error = err
        return steps.DispatchOutcome(error=error.redact([settings.api_key]))

    def _return(self, action: steps.Return) -> Step:
        return Finished(action.response)

    def _raise(self, action: steps.Raise) -> Step:
        raise action.error

    HANDLERS: ClassVar[dict[type, Callable[[Any, Any], Step]]] = {
        steps.Dispatch: _dispatch,
        steps.Return: _return,
        steps.Raise: _raise,
    }
