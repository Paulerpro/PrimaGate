from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from primagate.adapters import anthropic, openai
from primagate.core import fallbacks as core_fallbacks
from primagate.core import tiers as core_tiers
from primagate.core.call import parse_drop_params, resolve_model
from primagate.core.ports import Adapter
from primagate.core.prices import Price
from primagate.schema.errors import AuthenticationError, BadRequestError
from primagate.schema.tiers import (
    FEATURES,
    UNKNOWN,
    BindingLimits,
    BindingPricing,
    Tier,
    TierBinding,
    Unknown,
)
from primagate.schema.wire import Endpoint, Fallbacks, ProviderSettings


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


def parse_fallbacks(per_model: object, default: object) -> Fallbacks:
    """Validate the Fallback List declaration; nothing declared is the same as an empty list."""
    parsed: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    if per_model is not None:
        if not isinstance(per_model, Mapping):
            raise BadRequestError("fallbacks must map each model to a list of fallback models")
        for name, entries in per_model.items():
            key = _fallback_model("fallbacks", name)
            if key in parsed:
                raise BadRequestError(f"fallbacks lists model '{name}' more than once")
            parsed[key] = _fallback_list(f"fallbacks for '{name}'", entries)
    return Fallbacks(
        MappingProxyType(parsed),
        () if default is None else _fallback_list("default_fallbacks", default),
    )


def _fallback_list(label: str, value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise BadRequestError(f"{label} must be a list of model names")
    return tuple(_fallback_model(label, entry, reserved=True) for entry in value)


def _fallback_model(label: str, name: object, *, reserved: bool = False) -> tuple[str, str]:
    if not isinstance(name, str) or not name.strip():
        raise BadRequestError(f"{label} must name models with non-empty strings (got {name!r})")
    if reserved and name in core_fallbacks.RESERVED_MODEL_NAMES:
        raise BadRequestError(
            f"{label} names '{name}', which is the Tier Routing alias; "
            "a fallback must be a concrete model"
        )
    provider, model = resolve_model(name)
    if not model.strip():
        raise BadRequestError(f"{label} names '{name}' without a model")
    return provider, model


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


_PRICE_FIELDS = frozenset({"input", "output"})


def parse_price_overrides(value: object) -> dict[tuple[str, str], Price]:
    """Validate the caller's per-model prices (USD per million tokens)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BadRequestError("price_overrides must map each model to its input and output price")
    parsed: dict[tuple[str, str], Price] = {}
    for name, raw in value.items():
        if not isinstance(name, str):
            raise BadRequestError("price_overrides keys must be model names")
        if not isinstance(raw, Mapping) or set(raw) != _PRICE_FIELDS:
            raise BadRequestError(
                f"price_overrides for '{name}' must be a mapping with 'input' and 'output'"
            )
        try:
            key = resolve_model(name)
        except BadRequestError as err:
            raise BadRequestError(f"price_overrides: {err.message}") from None
        if key in parsed:
            raise BadRequestError(f"price_overrides lists model '{name}' more than once")
        parsed[key] = Price(
            _price_number(name, "input", raw["input"]), _price_number(name, "output", raw["output"])
        )
    return parsed


def _price_number(model: str, label: str, value: object) -> float:
    # bool is an int subclass, and `True` as a price is a mistake worth catching.
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise BadRequestError(
            f"price_overrides for '{model}' needs a finite {label} price of 0 or more"
        )
    return float(value)


def layer_prices(
    bindings: Mapping[Tier, TierBinding], overrides: Mapping[tuple[str, str], Price]
) -> dict[tuple[str, str], Price]:
    """The prices that replace the bundled table: binding prices, then the caller's own on top."""
    return {**core_tiers.binding_prices(bindings.values()), **overrides}


_BINDING_FIELDS = frozenset({"model", "settings", "limits", "pricing"})
_SETTING_FIELDS = frozenset({"temperature", "top_p", "max_tokens", "seed", "stop"})
_LIMIT_FIELDS = frozenset({"context_window", "max_output_tokens", "supported_features"})
_UNKNOWN_WORD = "unknown"


def parse_tier_bindings(value: object) -> dict[Tier, TierBinding]:
    """Validate the Tier Bindings declaration into frozen data.

    Messages name the Tier and the field but never echo a value, which could be a secret.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BadRequestError("tier_bindings must map each Tier label to its binding")
    parsed: dict[Tier, TierBinding] = {}
    for label, raw in value.items():
        # Exact upper-case words only: a near miss such as "small" is a mistake to surface.
        if not isinstance(label, str) or label not in Tier.__members__:
            raise BadRequestError("tier_bindings labels must be SMALL, MEDIUM or POWERFUL")
        parsed[Tier[label]] = _binding(label, raw)
    core_tiers.binding_prices(parsed.values())
    return parsed


def _binding(label: str, raw: object) -> TierBinding:
    where = f"tier_bindings for {label}"
    if not isinstance(raw, Mapping):
        raise BadRequestError(f"{where} must be a mapping with a model")
    _only(where, raw, _BINDING_FIELDS)
    name = raw.get("model")
    # Padding would be stored as part of the model id, so the binding's price would never match.
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise BadRequestError(f"{where} needs a non-empty model without surrounding spaces")
    try:
        provider, model_id = resolve_model(name)
    except BadRequestError:
        raise BadRequestError(f"{where} names a model with an unknown provider") from None
    if not model_id.strip():
        raise BadRequestError(f"{where} names a provider without a model")
    return TierBinding(
        Tier[label],
        provider,
        model_id,
        # An absent section is empty, but one given as None is a mistake, like any other bad type.
        _settings(where, raw.get("settings", {})),
        _limits(where, raw.get("limits", {})),
        _pricing(where, raw.get("pricing", {})),
    )


def _only(where: str, raw: Mapping[object, object], allowed: frozenset[str]) -> None:
    # Only the count is reported for an unrecognised key, because the key itself may be a secret.
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise BadRequestError(f"{where} has {len(unknown)} unrecognised field(s)")


def _section(where: str, name: str, raw: object, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise BadRequestError(f"{where}: {name} must be a mapping")
    _only(f"{where}: {name}", raw, allowed)
    return raw


def _settings(where: str, raw: object) -> Mapping[str, Any]:
    given = _section(where, "settings", raw, _SETTING_FIELDS)
    checked: dict[str, Any] = {}
    for key, value in given.items():
        at = f"{where}: settings.{key}"
        if key in ("temperature", "top_p"):
            checked[key] = _number(at, value)
        elif key == "max_tokens":
            checked[key] = _whole(at, value, minimum=1)
        elif key == "seed":
            checked[key] = _whole(at, value, minimum=None)
        else:
            checked[key] = _stop(at, value)
    return MappingProxyType(checked)


def _stop(where: str, value: object) -> str | tuple[str, ...]:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise BadRequestError(f"{where} must be a string or a non-empty list of strings")


def _limits(where: str, raw: object) -> BindingLimits:
    given = _section(where, "limits", raw, _LIMIT_FIELDS)
    return BindingLimits(
        _maybe_whole(f"{where}: limits.context_window", given.get("context_window", UNKNOWN)),
        _maybe_whole(f"{where}: limits.max_output_tokens", given.get("max_output_tokens", UNKNOWN)),
        _features(f"{where}: limits.supported_features", given.get("supported_features", UNKNOWN)),
    )


def _features(where: str, value: object) -> frozenset[str] | Unknown:
    if value is UNKNOWN or value == _UNKNOWN_WORD:
        return UNKNOWN
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not all(isinstance(v, str) and v in FEATURES for v in value)
    ):
        raise BadRequestError(f"{where} must be a list drawn from {', '.join(FEATURES)}")
    return frozenset(value)


def _pricing(where: str, raw: object) -> BindingPricing:
    given = _section(where, "pricing", raw, _PRICE_FIELDS)
    return BindingPricing(
        _maybe_number(f"{where}: pricing.input", given.get("input", UNKNOWN)),
        _maybe_number(f"{where}: pricing.output", given.get("output", UNKNOWN)),
    )


def _maybe_whole(where: str, value: object) -> int | Unknown:
    if value is UNKNOWN or value == _UNKNOWN_WORD:
        return UNKNOWN
    return _whole(where, value, minimum=1)


def _maybe_number(where: str, value: object) -> float | Unknown:
    if value is UNKNOWN or value == _UNKNOWN_WORD:
        return UNKNOWN
    return _number(where, value)


def _number(where: str, value: object) -> float:
    # bool is an int subclass, and `True` as a number is a mistake worth catching.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BadRequestError(f"{where} must be a finite number of 0 or more")
    try:
        number = float(value)
    except OverflowError:
        # An integer too large for a float is no usable number, and must not leak OverflowError.
        raise BadRequestError(f"{where} must be a finite number of 0 or more") from None
    if not math.isfinite(number) or number < 0:
        raise BadRequestError(f"{where} must be a finite number of 0 or more")
    return number


def _whole(where: str, value: object, *, minimum: int | None) -> int:
    too_small = minimum is not None and isinstance(value, int) and value < minimum
    if not isinstance(value, int) or isinstance(value, bool) or too_small:
        bound = "" if minimum is None else f" of {minimum} or more"
        raise BadRequestError(f"{where} must be a whole number{bound}")
    return value
