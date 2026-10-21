from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from primagate.adapters import anthropic, openai
from primagate.core.call import parse_drop_params, resolve_model
from primagate.core.ports import Adapter
from primagate.schema.errors import AuthenticationError, BadRequestError
from primagate.schema.wire import Endpoint, ProviderSettings


@dataclass(frozen=True)
class ProviderSpec:
    adapter: Callable[[], Adapter]
    default_base_url: str
    key_env: str


BUILTIN_PROVIDERS: Mapping[str, ProviderSpec] = {
    openai.PROVIDER: ProviderSpec(openai.OpenAIAdapter, openai.DEFAULT_BASE_URL, openai.KEY_ENV),
    anthropic.PROVIDER: ProviderSpec(
        anthropic.AnthropicAdapter, anthropic.DEFAULT_BASE_URL, anthropic.KEY_ENV
    ),
}


class SettingsSource(Protocol):
    def __getitem__(self, provider: str, /) -> ProviderSettings: ...


def builtin_adapters() -> dict[str, Adapter]:
    # Built per instance so one caller's adapters never leak to another.
    return {name: spec.adapter() for name, spec in BUILTIN_PROVIDERS.items()}


def check_provider_names(label: str, names: Mapping[str, object]) -> None:
    unknown = sorted(set(names) - set(BUILTIN_PROVIDERS))
    if unknown:
        known = ", ".join(sorted(BUILTIN_PROVIDERS))
        raise BadRequestError(f"{label} names unknown provider '{unknown[0]}' (known: {known})")


def check_drop_params(value: object) -> None:
    parse_drop_params(value)


ProviderRouting = Mapping[tuple[str, str], tuple[Endpoint, ...]]
_ENDPOINT_FIELDS = frozenset({"base_url", "api_key"})


def parse_provider_routing(value: object) -> ProviderRouting:
    """Validate the per-model endpoint declaration; a listed model has Provider Routing on.

    Messages name the model and the problem but never echo an endpoint's contents, which may
    hold a key.
    """
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise BadRequestError("provider_routing must map each model to its redundant endpoints")
    parsed: dict[tuple[str, str], tuple[Endpoint, ...]] = {}
    for name, endpoints in value.items():
        if not isinstance(name, str):
            raise BadRequestError("provider_routing keys must be model names")
        key = resolve_model(name)
        if key in parsed:
            raise BadRequestError(f"provider_routing lists model '{name}' more than once")
        if not isinstance(endpoints, Sequence) or isinstance(endpoints, str | bytes):
            raise BadRequestError(f"provider_routing for '{name}' must be a list of endpoints")
        if not endpoints:
            raise BadRequestError(
                f"Provider Routing is on for '{name}' but no redundant endpoint is declared"
            )
        parsed[key] = tuple(_endpoint(name, e) for e in endpoints)
    return MappingProxyType(parsed)


def _endpoint(model: str, raw: object) -> Endpoint:
    if not isinstance(raw, Mapping):
        raise BadRequestError(f"An endpoint for '{model}' must be a mapping with a base_url")
    unknown = sorted(str(k) for k in set(raw) - _ENDPOINT_FIELDS)
    if unknown:
        raise BadRequestError(f"An endpoint for '{model}' has unknown field '{unknown[0]}'")
    base_url, api_key = raw.get("base_url"), raw.get("api_key")
    if not isinstance(base_url, str) or not base_url:
        raise BadRequestError(f"An endpoint for '{model}' needs a non-empty base_url")
    if api_key is not None and not isinstance(api_key, str):
        raise BadRequestError(f"An endpoint for '{model}' has an api_key that is not a string")
    return Endpoint(base_url=base_url, api_key=api_key)


class SettingsResolver:
    """Resolves each provider's key and base URL on first use, then keeps the answer.

    A key is looked up only for the provider being called, so a missing key for one provider
    never blocks another, and a key is never handed to a different provider.
    """

    def __init__(
        self,
        *,
        api_keys: Mapping[str, str] | None = None,
        base_urls: Mapping[str, str] | None = None,
    ) -> None:
        self._api_keys = dict(api_keys or {})
        self._base_urls = dict(base_urls or {})
        self._resolved: dict[str, ProviderSettings] = {}
        self._lock = threading.Lock()

    def __getitem__(self, provider: str) -> ProviderSettings:
        with self._lock:
            if provider not in self._resolved:
                self._resolved[provider] = self._resolve(provider)
            return self._resolved[provider]

    def _resolve(self, provider: str) -> ProviderSettings:
        spec = BUILTIN_PROVIDERS[provider]
        key = self._api_keys.get(provider) or os.environ.get(spec.key_env)
        if not key:
            raise AuthenticationError(
                f"No {provider} API key: pass api_keys={{'{provider}': ...}} or set {spec.key_env}",
                llm_provider=provider,
            )
        return ProviderSettings(
            api_key=key, base_url=self._base_urls.get(provider, spec.default_base_url)
        )

    def __repr__(self) -> str:
        return f"SettingsResolver(resolved={sorted(self._resolved)})"
