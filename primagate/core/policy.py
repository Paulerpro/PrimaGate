from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from primagate.schema.errors import BadRequestError
from primagate.schema.policy import COST_PREFERENCES, Policy
from primagate.schema.tiers import Tier, TierBinding

_FIELDS = frozenset(
    {
        "enabled_tiers",
        "confidence_threshold",
        "context_token_budget",
        "interim_binding",
        "cost_preference",
        "ambiguous_context_default",
        "max_concurrent_analyzer_calls",
        "spending_ceiling",
    }
)
_REQUIRED = (
    "enabled_tiers",
    "confidence_threshold",
    "context_token_budget",
    "interim_binding",
)


def build_policy(
    raw: object, bindings: Mapping[Tier, TierBinding]
) -> tuple[Policy, tuple[str, ...]]:
    """Validate a declared Policy against the Tier Bindings. Returns it and any advisories.

    Messages name the field and never echo a value, which could be a secret.
    """
    if not isinstance(raw, Mapping):
        raise BadRequestError("policy must be a mapping of Policy fields")
    # One snapshot, so what is checked is exactly what is stored even for an odd Mapping.
    raw = dict(raw)
    unknown = [k for k in raw if k not in _FIELDS]
    if unknown:
        # The key is not echoed, since it may be a secret; the valid names let a typo be found.
        raise BadRequestError(
            f"policy has {len(unknown)} unrecognised field(s); valid fields are "
            f"{', '.join(sorted(_FIELDS))}"
        )
    for name in _REQUIRED:
        if name not in raw:
            raise BadRequestError(f"policy needs {name}")
    tiers = _enabled(raw["enabled_tiers"])
    for tier in tiers:
        if tier not in bindings:
            raise BadRequestError(f"policy enables {tier.name} but it has no Tier Binding")
    interim = _enabled_tier("interim_binding", raw["interim_binding"], tiers)
    # Not defaulted past MEDIUM: a hidden fallback would route somewhere the Caller never named.
    ambiguous = _enabled_tier(
        "ambiguous_context_default", raw.get("ambiguous_context_default", "MEDIUM"), tiers
    )
    preference = raw.get("cost_preference", "balanced")
    if not isinstance(preference, str) or preference not in COST_PREFERENCES:
        raise BadRequestError(
            f"policy cost_preference must be one of {', '.join(COST_PREFERENCES)}"
        )
    cap = raw.get("max_concurrent_analyzer_calls")
    ceiling = raw.get("spending_ceiling")
    fields: dict[str, Any] = {
        "enabled_tiers": tiers,
        "confidence_threshold": _threshold(raw["confidence_threshold"]),
        "context_token_budget": _whole("context_token_budget", raw["context_token_budget"]),
        "interim_binding": interim,
        "cost_preference": preference,
        "ambiguous_context_default": ambiguous,
        "max_concurrent_analyzer_calls": (
            _whole("max_concurrent_analyzer_calls", cap)
            if "max_concurrent_analyzer_calls" in raw
            else None
        ),
        "spending_ceiling": (
            _above_zero("spending_ceiling", ceiling) if "spending_ceiling" in raw else None
        ),
    }
    advisories: tuple[str, ...] = ()
    if preference != "balanced":
        advisories = (
            f"policy cost_preference is {preference}, not the default balanced; it is accepted "
            "but has no effect yet: it is advice for later, and routing does not use it",
        )
    return Policy(version=_version(fields), **fields), advisories


def _enabled(value: object) -> tuple[Tier, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise BadRequestError("policy enabled_tiers must be a non-empty list of Tier labels")
    tiers: set[Tier] = set()
    for label in value:
        # Exact upper-case words only: a near miss such as "small" is a mistake to surface.
        if not isinstance(label, str) or label not in Tier.__members__:
            raise BadRequestError("policy enabled_tiers must hold SMALL, MEDIUM or POWERFUL")
        if Tier[label] in tiers:
            raise BadRequestError("policy enabled_tiers names a Tier more than once")
        tiers.add(Tier[label])
    return tuple(sorted(tiers))


def _enabled_tier(field: str, value: object, enabled: tuple[Tier, ...]) -> Tier:
    if not isinstance(value, str) or value not in Tier.__members__:
        raise BadRequestError(f"policy {field} must be SMALL, MEDIUM or POWERFUL")
    if Tier[value] not in enabled:
        raise BadRequestError(f"policy {field} names a Tier that is not in enabled_tiers")
    return Tier[value]


def _finite(field: str, value: object) -> float:
    # bool is an int subclass, and `True` as a number is a mistake worth catching.
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BadRequestError(f"policy {field} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        # An integer too large for a float is no usable number, and must not leak OverflowError.
        raise BadRequestError(f"policy {field} must be a finite number") from None
    if not math.isfinite(number):
        raise BadRequestError(f"policy {field} must be a finite number")
    return number


def _threshold(value: object) -> float:
    number = _finite("confidence_threshold", value)
    if not 0 < number <= 1:
        raise BadRequestError("policy confidence_threshold must be above 0 and at most 1")
    return number


def _above_zero(field: str, value: object) -> float:
    number = _finite(field, value)
    if number <= 0:
        raise BadRequestError(f"policy {field} must be above 0")
    return number


def _whole(field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise BadRequestError(f"policy {field} must be a whole number of 1 or more")
    return value


def _version(fields: Mapping[str, Any]) -> str:
    shown = {
        k: [t.name for t in v] if isinstance(v, tuple) else v.name if isinstance(v, Tier) else v
        for k, v in fields.items()
    }
    blob = json.dumps(shown, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
