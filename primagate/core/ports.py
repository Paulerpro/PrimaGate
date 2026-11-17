from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from primagate.schema.errors import PrimaGateError
from primagate.schema.profile import Profile
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


class ProfileStore(Protocol):
    """One slot holding the latest Profile written. Whether it still fits is core's call.

    `non_blocking` is true only when `read` and `write` never wait on I/O, so the async driver
    can call them on the event loop instead of in a worker thread.
    """

    non_blocking: bool

    def read(self) -> Profile | None: ...

    def write(self, profile: Profile) -> None:
        """Replace the slot; a reader must see the old Profile or the new one, never a mix."""
        ...


class ModelRegistry(Protocol):
    """What features a model supports, from a source other than its adapter."""

    non_blocking: bool

    def supported_features(self, provider: str, model: str) -> frozenset[str] | None:
        """The model's features, or None when the registry does not know the model."""
        ...
