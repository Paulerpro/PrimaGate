from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from primagate.schema.errors import BadRequestError
from primagate.schema.profile import Profile
from primagate.schema.tiers import FEATURES


class InMemoryProfileStore:
    """The latest Profile, held only for the life of the process.

    A Profile is frozen, so swapping the reference under a lock is enough for a reader to get
    the old Profile or the new one whole, and nothing needs copying.
    """

    non_blocking = True

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profile: Profile | None = None

    def read(self) -> Profile | None:
        with self._lock:
            return self._profile

    def write(self, profile: Profile) -> None:
        with self._lock:
            self._profile = profile


class InMemoryModelRegistry:
    """Features per `(provider, model)`, checked once when built and never changed after."""

    non_blocking = True

    def __init__(self, entries: Mapping[tuple[str, str], Iterable[str]]) -> None:
        if not isinstance(entries, Mapping):
            raise BadRequestError(
                "model registry must be a mapping of (provider, model) to features"
            )
        # A read-only copy, so a caller who changes their mapping later cannot change the answers.
        self._entries: Mapping[tuple[str, str], frozenset[str]] = MappingProxyType(
            {_key(key): _features(features) for key, features in entries.items()}
        )

    def supported_features(self, provider: str, model: str) -> frozenset[str] | None:
        return self._entries.get((provider, model))


def _key(key: object) -> tuple[str, str]:
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not all(isinstance(part, str) and part for part in key)
    ):
        raise BadRequestError("model registry keys must be (provider, model) pairs of text")
    return key[0], key[1]


def _features(features: object) -> frozenset[str]:
    problem = f"model registry features must be a collection drawn from {', '.join(FEATURES)}"
    # A bare string would be read letter by letter, and a mapping by its keys alone, so a
    # feature marked False would count as supported.
    if isinstance(features, str | bytes | Mapping) or not isinstance(features, Iterable):
        raise BadRequestError(problem)
    found = list(features)
    if not all(isinstance(name, str) and name in FEATURES for name in found):
        raise BadRequestError(problem)
    return frozenset(found)
