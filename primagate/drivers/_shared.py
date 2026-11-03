from __future__ import annotations

import dataclasses
import logging
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from primagate.core import steps
from primagate.core.call import attempt_chain
from primagate.core.ports import Adapter
from primagate.core.prices import estimate
from primagate.drivers.providers import SettingsSource
from primagate.schema.errors import PrimaGateError
from primagate.schema.events import Event
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, Metadata
from primagate.schema.wire import ProviderRequest, ProviderSettings, RawEvent, RawOutcome

logger = logging.getLogger("primagate")

EventHandler = Callable[[Event], None]


def deliver(events: tuple[Event, ...], handler: EventHandler | None) -> None:
    """Write each event to the logger, then give it to the handler, exactly once.

    The logger goes first so a faulty handler cannot hide what happened. A handler's own
    exception is logged and dropped because observing a call must never change it; only
    `BaseException`s that are not `Exception`s (cancellation, interrupts) get through.
    """
    for event in events:
        stamped = dataclasses.replace(
            event, timestamp=datetime.now(UTC).isoformat(timespec="milliseconds")
        )
        fields = " ".join(
            f"{k}={v}"
            for k, v in stamped.to_dict().items()
            if k not in ("event_type", "request_id")
        )
        logger.log(
            stamped.log_level, "%s request_id=%s %s", stamped.event_type, stamped.request_id, fields
        )
        if handler is None:
            continue
        try:
            handler(stamped)
        except Exception:
            logger.exception("event handler failed on %s", stamped.event_type)


@dataclass(frozen=True)
class Finished:
    response: ChatCompletion


Step = steps.Outcome | Finished


def credentials(settings: SettingsSource, action: steps.Dispatch) -> ProviderSettings:
    """The URL and key this dispatch is sent with; a redundant endpoint may bring its own."""
    endpoint = action.endpoint
    if endpoint is not None and endpoint.api_key:
        # Its own key means the provider's configured key is never needed, or looked up.
        return ProviderSettings(api_key=endpoint.api_key, base_url=endpoint.base_url)
    configured = settings[action.provider]
    if endpoint is None:
        return configured
    return ProviderSettings(api_key=configured.api_key, base_url=endpoint.base_url)


def prepare(
    adapters: Mapping[str, Adapter],
    settings: SettingsSource,
    action: steps.Dispatch,
) -> ProviderRequest:
    adapter = adapters[action.provider]
    config = credentials(settings, action)
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
    return steps.DispatchOutcome(error=error.redact([credentials(settings, action).api_key]))


END: Final = object()


class StreamState:
    """Sans-IO bookkeeping for one stream: queued chunks, then at most one error, then the end."""

    def __init__(
        self,
        adapter: Adapter,
        api_key: str,
        headers: Mapping[str, str],
        action: steps.Dispatch,
    ) -> None:
        self._adapter = adapter
        self._action = action
        self._dropped = list(action.dropped_params)
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
        self._error = self._with_chain(self._adapter.map_error(raw))
        self._finished = True

    def _with_chain(self, error: PrimaGateError) -> PrimaGateError:
        error.redact([self._api_key])
        error.attempt_chain = attempt_chain(self._action, error)
        return error

    def _produce(self, make: Callable[[], list[ChatCompletionChunk]]) -> None:
        try:
            chunks = make()
            for chunk in chunks:
                # The final chunk is where finish reason and usage already live.
                if any(c.finish_reason for c in chunk.choices):
                    priced = estimate(
                        self._action.provider,
                        self._action.request.model,
                        chunk.usage,
                        self._action.prices,
                    )
                    chunk.metadata = Metadata(
                        dropped_params=list(self._dropped),
                        attempt_chain=attempt_chain(self._action),
                        token_counts=priced.token_counts,
                        cost=priced.cost,
                        unknown_model=priced.unknown_model,
                    )
            self._queue.extend(chunks)
        except PrimaGateError as err:
            self._error = self._with_chain(err)
            self._finished = True

    def next_item(self) -> ChatCompletionChunk | object | None:
        """A chunk, `END` once the stream is over, or None when more input is needed."""
        if self._queue:
            return self._queue.popleft()
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return END if self._finished else None
