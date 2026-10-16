from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from primagate.adapters import anthropic, openai
from primagate.core.ports import Adapter
from primagate.schema.errors import AuthenticationError, BadRequestError
from primagate.schema.wire import ProviderSettings


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
