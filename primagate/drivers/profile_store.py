from __future__ import annotations

# Re-exported so the client can name the port without reaching into core.
from primagate.core.ports import ProfileStore as ProfileStore
from primagate.stores.memory import InMemoryProfileStore


def default_profile_store() -> ProfileStore:
    """A new, empty in-memory store; it lasts only as long as the process."""
    return InMemoryProfileStore()
