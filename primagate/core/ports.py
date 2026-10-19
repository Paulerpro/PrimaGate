from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from primagate.schema.wire import Capabilities, ProviderRequest, RawEvent, RawOutcome


class StreamParser(Protocol):
    """Stateful translation of one stream's events into unified chunks."""

    def feed(self, event: RawEvent) -> list[ChatCompletionChunk]: ...

    def finish(self) -> list[ChatCompletionChunk]:
        """Flush at end of input; raises if the stream never reached its terminal event."""
        ...


class Adapter(Protocol):
    """Pure translation for one provider. Implementations perform no I/O."""

    def build_request(
        self, request: ChatCompletionRequest, *, api_key: str, base_url: str
    ) -> ProviderRequest: ...

    def parse_response(self, outcome: RawOutcome) -> ChatCompletion: ...

    def new_stream_parser(self, headers: Mapping[str, str]) -> StreamParser: ...

    def map_error(self, outcome: RawOutcome) -> PrimaGateError: ...

    def capabilities(self) -> Capabilities: ...
