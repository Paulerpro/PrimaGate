from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from primagate.core import steps
from primagate.core.ports import Adapter
from primagate.drivers.providers import SettingsSource
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, Metadata
from primagate.schema.wire import ProviderRequest, RawEvent, RawOutcome


@dataclass(frozen=True)
class Finished:
    response: ChatCompletion


Step = steps.DispatchOutcome | Finished


def prepare(
    adapters: Mapping[str, Adapter],
    settings: SettingsSource,
    action: steps.Dispatch,
) -> ProviderRequest:
    adapter = adapters[action.provider]
    config = settings[action.provider]
    return adapter.build_request(action.request, api_key=config.api_key, base_url=config.base_url)


def settle(
    adapters: Mapping[str, Adapter],
    settings: SettingsSource,
    action: steps.Dispatch,
    raw: RawOutcome,
) -> steps.DispatchOutcome:
    """Turn what transport saw into a response or error, with the key scrubbed."""
    adapter = adapters[action.provider]
    try:
        if raw.ok:
            return steps.DispatchOutcome(
                response=adapter.parse_response(raw), dropped_params=action.dropped_params
            )
        error = adapter.map_error(raw)
    except PrimaGateError as err:
        error = err
    return steps.DispatchOutcome(error=error.redact([settings[action.provider].api_key]))


END: Final = object()


class StreamState:
    """Sans-IO bookkeeping for one stream: queued chunks, then at most one error, then the end."""

    def __init__(
        self,
        adapter: Adapter,
        api_key: str,
        headers: Mapping[str, str],
        dropped_params: tuple[str, ...] = (),
    ) -> None:
        self._adapter = adapter
        self._dropped = list(dropped_params)
        self._api_key = api_key
        self._parser = adapter.new_stream_parser(headers)
        self._queue: deque[ChatCompletionChunk] = deque()
        self._error: PrimaGateError | None = None
        self._finished = False

    def feed(self, event: RawEvent) -> None:
        self._produce(lambda: self._parser.feed(event))

    def end(self) -> None:
        self._produce(self._parser.finish)
        self._finished = True

    def fail(self, kind: str) -> None:
        raw = RawOutcome(status=None, headers={}, body=b"", failure=kind)
        self._error = self._adapter.map_error(raw).redact([self._api_key])
        self._finished = True

    def _produce(self, make: Callable[[], list[ChatCompletionChunk]]) -> None:
        try:
            chunks = make()
            for chunk in chunks:
                # The final chunk is where finish reason and usage already live.
                if self._dropped and any(c.finish_reason for c in chunk.choices):
                    chunk.metadata = Metadata(dropped_params=list(self._dropped))
            self._queue.extend(chunks)
        except PrimaGateError as err:
            self._error = err.redact([self._api_key])
            self._finished = True

    def next_item(self) -> ChatCompletionChunk | object | None:
        """A chunk, `END` once the stream is over, or None when more input is needed."""
        if self._queue:
            return self._queue.popleft()
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return END if self._finished else None
