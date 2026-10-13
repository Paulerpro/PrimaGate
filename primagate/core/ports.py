from __future__ import annotations

from typing import Any, Protocol

from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest
from primagate.schema.wire import Capabilities, ProviderRequest, RawOutcome


class Adapter(Protocol):
    """Pure translation for one provider. Implementations perform no I/O."""

    def build_request(
        self, request: ChatCompletionRequest, *, api_key: str, base_url: str
    ) -> ProviderRequest: ...

    def parse_response(self, outcome: RawOutcome) -> ChatCompletion: ...

    def new_stream_parser(self) -> Any: ...

    def map_error(self, outcome: RawOutcome) -> PrimaGateError: ...

    def capabilities(self) -> Capabilities: ...
