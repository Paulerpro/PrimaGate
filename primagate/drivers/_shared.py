from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from primagate.core import steps
from primagate.core.ports import Adapter
from primagate.drivers.providers import SettingsSource
from primagate.schema.errors import PrimaGateError
from primagate.schema.types import ChatCompletion
from primagate.schema.wire import ProviderRequest, RawOutcome


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
            return steps.DispatchOutcome(response=adapter.parse_response(raw))
        error = adapter.map_error(raw)
    except PrimaGateError as err:
        error = err
    return steps.DispatchOutcome(error=error.redact([settings[action.provider].api_key]))
